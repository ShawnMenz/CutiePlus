"""The work folder (masks/ and metadata.json), anchors, intervals, review workflow and the
per-task lock (gui/task_state.py)."""
import os
import shutil
from os import path

import pytest

from gui.edit_history import Edit, EditHistory, FrameState
from gui.task_input import read_sequence
from gui.task_state import (ATTENTION, INTERVALS, PROGRESS, SOURCE, TaskLock, TaskLocked,
                            TaskState, attention_of, read_metadata, read_section,
                            check_intervals, discard_of, discard_task, intervals_from_payload,
                            intervals_payload, labeler_name, lock_holder, mark_done,
                            submit_for_review, summarize_task, task_status, utc_now,
                            work_folder)
from tests.helpers import make_task, summary_of


def test_an_anchor_records_its_kind_author_and_time(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, ))
    state = TaskState(task.workspace, 'alice')
    state.set_anchor('000001', 'mask', edited=True)
    entry = state.anchor('000001')
    assert set(entry) == {'kind', 'by', 'updated_at'}  # nothing finer, such as an origin
    assert (entry['kind'], entry['by']) == ('mask', 'alice')
    state.set_anchor('000003', 'empty', edited=True)
    assert state.anchor('000003')['kind'] == 'empty'
    # accepting the shown mask of an anchor of the same kind changes nothing
    before = dict(state.anchor('000003'))
    state.set_anchor('000003', 'empty', edited=False)
    assert state.anchor('000003') == before
    assert set(task.anchors()) == {'000001', '000003'}


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
    assert read_section(workspace, PROGRESS)['edited_by'] == ['alice', 'carol']


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
    assert state.intervals(10) == ([(1, 10)], 'full')  # the whole sequence by default
    state.save_intervals([(5, 6), (1, 2)])
    assert state.intervals(10) == ([(1, 2), (5, 6)], 'cutie')


def test_lock_excludes_a_second_window(tmp_path):
    task = make_task(str(tmp_path))
    lock = TaskLock.acquire(task.workspace, 'alice')
    with pytest.raises(TaskLocked) as error:
        TaskLock.acquire(task.workspace, 'bob')
    assert error.value.holder['labeler'] == 'alice' and 'display' not in error.value.holder
    assert lock_holder(task.workspace)['pid'] == os.getpid()
    assert f'alice (pid {os.getpid()})' in str(error.value)
    lock.release()
    assert lock_holder(task.workspace) is None
    TaskLock.acquire(task.workspace, 'bob').release()


def test_the_labeler_is_the_argument_else_the_variable_else_the_account(monkeypatch):
    import getpass
    monkeypatch.setenv('CUTIE_LABELER', 'fallback')
    assert labeler_name('cli') == 'cli' and labeler_name() == 'fallback'
    monkeypatch.delenv('CUTIE_LABELER')
    assert labeler_name() == getpass.getuser()


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
    assert state.attention is None and summary_of(task.workspace)['status'] == 'not started'
    state.save_attention(True)
    assert attention_of(task.workspace)['by'] == 'alice'
    summary = summary_of(task.workspace)
    assert summary['status'] == 'Attention' and summary['attention']['by'] == 'alice'
    assert task_status(task.workspace) == 'not started'  # the review status itself is unchanged
    # only the button clears it: an edit and a submission keep it
    state.set_anchor('000003', 'empty', edited=True)
    state.save()
    submit_for_review(task.workspace, 'bob')
    assert summary_of(task.workspace)['status'] == 'Attention'
    assert TaskState(task.workspace, 'bob').attention['by'] == 'alice'
    state.save_attention(False)
    assert ATTENTION not in read_metadata(task.workspace)
    assert summary_of(task.workspace)['status'] == 'under review'


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
    summary = summary_of(task.workspace)
    assert summary['status'] == 'discarded' and summary['discard']['by'] == 'dave'
    assert attention_of(task.workspace) is None  # Need Attention is turned off
    progress = read_section(task.workspace, PROGRESS)
    assert 'done_at' not in progress and 'review_at' in progress  # never listed as done again
    assert task.anchors() == anchors  # the labels stay
    discard_task(task.workspace, 'dave', False)
    assert discard_of(task.workspace) is None
    assert summary_of(task.workspace)['status'] == 'under review'  # Mark done again


def test_a_work_folder_reads_its_sequence_from_the_input(tmp_path):
    task = make_task(str(tmp_path), prefill=(2, ))
    source = read_section(task.workspace, SOURCE)
    assert source == {'input': path.join('..', '..', '..', 'tasks'), 'tab': 'main',
                      'sequence': '0001'}  # relative: input and work move together
    state = TaskState(task.workspace)
    assert (state.split, state.sequence_id, state.anchor_modality) == ('main', '0001', 'visual')
    assert state.frame_dirs == task.frame_dirs
    assert (state.category, state.description) == ('car', 'The red car moving right.')
    # the input's pregenerated masks, shown on frames without a mask of their own
    assert state.prefill == {'000003': {'source': 'pregenerated', 'path': path.join(
        task.root, 'main', 'annotation', '0001', '000003.png')}}
    state.save_description('Tuned.')
    assert TaskState(task.workspace).description == 'Tuned.'
    # Reset all clears the anchors; the pregenerated masks are input, and stay
    state.clear_mask_references()
    assert state.anchors == {} and list(state.prefill) == ['000003']


def test_without_a_target_the_category_is_null_and_the_description_empty(tmp_path):
    task = make_task(str(tmp_path), in_targets=False)
    state = TaskState(task.workspace)
    assert (state.category, state.description) == (None, '')
    summary = summary_of(task.workspace)
    assert (summary['category'], summary['target']) == ('null', 'null: ')


def test_a_sequence_never_opened_is_not_started_and_has_no_work_folder(tmp_path):
    task = make_task(str(tmp_path))
    make_task(str(tmp_path), sequence='0002', frame_count=7)
    shutil.rmtree(path.join(path.dirname(task.workspace), '0002'))  # never opened
    folder = work_folder(task.root, 'main', '0002')
    summary = summarize_task(read_sequence(task.root, 'main', '0002'), folder)
    assert (summary['status'], summary['frames'], summary['clips']) == ('not started', 7, [(1, 7)])
    assert summary['open_by'] is None and not path.exists(folder)


def test_each_added_interval_keeps_a_category_of_its_own(tmp_path):
    from gui.task_state import interval_categories
    task = make_task(str(tmp_path), frame_count=30)
    state = TaskState(task.workspace)
    state.save_intervals([(1, 10), (12, 20), (22, 30)],
                         {(12, 20): 'Second.', (22, 30): 'Third.'}, {(12, 20): 'person'})
    entries = read_section(task.workspace, INTERVALS)['intervals']
    assert entries[1] == {'start': 12, 'end': 20, 'description': 'Second.', 'category': 'person'}
    assert 'category' not in entries[0] and 'category' not in entries[2]
    # the first interval's is the sequence's, and so is an added one's saved without one
    assert state.interval_categories(30) == {(1, 10): 'car', (12, 20): 'person', (22, 30): 'car'}
    assert state.added_interval_categories(30) == {(12, 20): 'person', (22, 30): 'car'}
    state.save_category('truck')
    assert TaskState(task.workspace).interval_categories(30)[(1, 10)] == 'truck'
    assert interval_categories([(1, 30)], {}, None) == {(1, 30): None}
