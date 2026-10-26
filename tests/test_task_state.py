"""Task files, anchors, intervals, review workflow and the per-task lock (gui/task_state.py)."""
import json
import os
from os import path

import pytest

from gui.annotators import annotator_for_display, display_port, save_annotators
from gui.edit_history import Edit, EditHistory, FrameState
from gui.task_state import (ATTENTION_FILE, TaskLock, TaskLocked, TaskState, attention_of,
                            check_intervals, discard_of, discard_task, intervals_from_payload,
                            intervals_payload, lock_holder, mark_done, submit_for_review,
                            summarize_task, task_status, utc_now)
from tests.helpers import make_task


def test_anchor_origin_follows_label_studio(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, ))
    state = TaskState(task.workspace, 'alice')
    state.set_anchor('000001', 'mask', edited=True)
    entry = state.anchor('000001')
    assert entry['origin'] == 'ls_refined' and entry['ls_file'] == 'label_0.json'
    assert entry['by'] == 'alice'
    state.set_anchor('000003', 'empty', edited=True)
    assert state.anchor('000003')['origin'] == 'cutie'
    # accepting the shown mask of an anchor of the same kind changes nothing
    before = dict(state.anchor('000003'))
    state.set_anchor('000003', 'empty', edited=False)
    assert state.anchor('000003') == before
    on_disk = json.load(open(path.join(task.workspace, 'anchors.json')))
    assert set(on_disk) == {'000001', '000003'}


def test_restore_and_remove_anchor(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, 2))
    state = TaskState(task.workspace, 'alice')
    entry = dict(state.anchor('000003'))
    state.remove_anchor('000003')
    assert state.anchor('000003') is None
    state.restore_anchor('000003', entry)
    assert state.anchor('000003') == entry


def test_review_workflow(tmp_path):
    task = make_task(str(tmp_path))
    workspace = task.workspace
    assert task_status(workspace) == 'not started'
    state = TaskState(workspace, 'alice')
    state.touch()
    assert task_status(workspace) == 'in progress'
    submit_for_review(workspace, 'alice')
    assert task_status(workspace) == 'under review'
    mark_done(workspace, 'bob')
    assert task_status(workspace) == 'done'
    # any change reopens the task, and edited_by keeps everyone in order
    TaskState(workspace, 'carol').touch()
    assert task_status(workspace) == 'in progress'
    assert json.load(open(path.join(workspace, 'progress.json')))['edited_by'] == ['alice', 'carol']


def test_the_quit_time_of_ctrl_q_goes_with_the_done_mark(tmp_path):
    task = make_task(str(tmp_path))
    workspace = task.workspace
    TaskState(workspace, 'alice').touch()
    submit_for_review(workspace, 'alice')
    quit_at = utc_now()
    mark_done(workspace, 'alice', quit_at=quit_at)
    progress = task.progress()
    assert progress['quit_at'] == quit_at and progress['done_by'] == 'alice'
    assert progress['done_at'] >= progress['quit_at']
    # a new submission and a later Mark done from the browser keep no quit time
    submit_for_review(workspace, 'alice')
    assert not {'done_at', 'done_by', 'quit_at'} & set(task.progress())
    mark_done(workspace, 'bob')
    assert task_status(workspace) == 'done' and 'quit_at' not in task.progress()
    # a discard drops it with the done mark
    submit_for_review(workspace, 'alice')
    mark_done(workspace, 'alice', quit_at=utc_now())
    discard_task(workspace, 'dave', True)
    assert not {'done_at', 'done_by', 'quit_at'} & set(task.progress())


def test_intervals(tmp_path):
    assert intervals_from_payload(intervals_payload([(3, 9)])) == [(3, 9)]
    payload = intervals_payload([(10, 12), (1, 4)])
    assert 'start' not in payload
    assert intervals_from_payload(payload) == [(1, 4), (10, 12)]
    assert check_intervals([(1, 5), (5, 9)], 10)  # overlap
    assert check_intervals([(1, 4), (5, 9)], 10) == []
    assert check_intervals([(0, 4)], 10)
    task = make_task(str(tmp_path), frame_count=10)
    state = TaskState(task.workspace)
    assert state.intervals(10) == ([(1, 10)], 'curated')
    state.save_intervals([(5, 6), (1, 2)])
    assert state.intervals(10) == ([(1, 2), (5, 6)], 'cutie')


def test_lock_excludes_a_second_window(tmp_path):
    task = make_task(str(tmp_path))
    lock = TaskLock.acquire(task.workspace, 'alice')
    with pytest.raises(TaskLocked) as error:
        TaskLock.acquire(task.workspace, 'bob')
    assert error.value.holder['labeler'] == 'alice'
    assert lock_holder(task.workspace)['pid'] == os.getpid()
    lock.release()
    assert lock_holder(task.workspace) is None
    TaskLock.acquire(task.workspace, 'bob').release()


def test_annotator_from_vnc_display(tmp_path):
    assert display_port(':4') == 5904
    assert display_port('host:3.0') == 5903
    assert display_port(':0') is None
    save_annotators(str(tmp_path), [{'port': 5903, 'name': 'Dana'}])
    assert annotator_for_display(str(tmp_path), ':3') == 'Dana'
    assert annotator_for_display(str(tmp_path), ':2') is None


def test_edit_history_limit_and_redo():
    history = EditHistory(limit=2)
    edits = [Edit(str(i), FrameState(None, None), FrameState(b'x', None)) for i in range(3)]
    for edit in edits:
        history.record(edit)
    assert history.undo() is edits[2]
    assert history.undo() is edits[1]
    assert history.undo() is None  # the first edit fell off the limit
    assert history.redo() is edits[1]
    history.record(edits[0])
    assert history.redo() is None


def test_need_attention_shows_as_the_status_and_stays_through_edits(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, ))
    state = TaskState(task.workspace, 'alice')
    assert state.attention is None and summarize_task(task.workspace)['status'] == 'not started'
    state.save_attention(True)
    assert attention_of(task.workspace)['by'] == 'alice'
    summary = summarize_task(task.workspace)
    assert summary['status'] == 'Attention' and summary['attention']['by'] == 'alice'
    assert task_status(task.workspace) == 'not started'  # the review status itself is unchanged
    # only the button clears it: an edit and a submission keep it
    state.set_anchor('000003', 'empty', edited=True)
    state.save()
    submit_for_review(task.workspace, 'bob')
    assert summarize_task(task.workspace)['status'] == 'Attention'
    assert TaskState(task.workspace, 'bob').attention['by'] == 'alice'
    state.save_attention(False)
    assert not path.exists(path.join(task.workspace, ATTENTION_FILE))
    assert summarize_task(task.workspace)['status'] == 'under review'


def test_discard_drops_the_done_mark_and_attention_and_restore_brings_the_review_back(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, ))
    state = TaskState(task.workspace, 'alice')
    state.set_anchor('000003', 'empty', edited=True)
    state.save()
    submit_for_review(task.workspace, 'bob')
    mark_done(task.workspace, 'carol')
    state.save_attention(True)
    anchors = task.anchors()
    discard_task(task.workspace, 'dave', True)
    summary = summarize_task(task.workspace)
    assert summary['status'] == 'discarded' and summary['discard']['by'] == 'dave'
    assert attention_of(task.workspace) is None  # Need Attention is turned off
    with open(path.join(task.workspace, 'progress.json')) as f:
        progress = json.load(f)
    assert 'done_at' not in progress and 'review_at' in progress  # never listed as done again
    assert task.anchors() == anchors  # the labels stay
    discard_task(task.workspace, 'dave', False)
    assert discard_of(task.workspace) is None
    assert summarize_task(task.workspace)['status'] == 'under review'  # Mark done again
