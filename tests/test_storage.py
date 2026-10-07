"""
The coordinator's storage and commit protocol (cutieplus/coordinator/tasks.py): import of the
legacy files, leases and epochs, the per-task file lock shared with the original window, the
commit rules, revisions, idempotent requests, failed transactions, and the legacy files kept as
views.
"""
import json
import os
from os import path

import numpy as np
import pytest

from cutieplus.coordinator.sessions import SessionManager
from cutieplus.coordinator.tasks import TaskError, TaskService, decode_mask, encode_mask
from cutieplus.storage.db import Database
from cutieplus.storage.objects import ObjectStore
from cutieplus.storage.views import TaskViews
from gui.task_state import TaskLock, TaskLocked, TaskState, lock_holder, task_status
from tests.helpers import ellipse_mask, make_task, read_mask


class Setup:
    def __init__(self, tmp_path, **task_args):
        args = dict(frame_count=10, mask_anchors=(0, ), empty_anchors=(8, ), prefill=(5, ))
        args.update(task_args)
        self.task = make_task(str(tmp_path), **args)
        state = path.join(str(tmp_path), 'state')
        self.db = Database(path.join(state, 'service.db'))
        self.objects = ObjectStore(path.join(state, 'objects'))
        self.tasks = TaskService(self.db, self.objects, self.task.root)
        self.sessions = SessionManager(self.db, timeout=30)
        self.alice = self.sessions.register('alice', ':2', 111, 'lab')['session_id']
        self.bob = self.sessions.register('bob', ':3', 222, 'lab')['session_id']

    def open(self, session=None):
        return self.tasks.open_task(self.task.workspace, session or self.alice)

    def revisions(self):
        row = self.tasks.task_row('main/0001')
        return row['input_revision'], row['content_revision']


@pytest.fixture
def setup(tmp_path):
    return Setup(tmp_path)


def test_open_imports_the_task_and_takes_both_locks(setup):
    reply = setup.open()
    assert reply['sync'] == 'imported' and reply['lease_epoch'] == 1
    frames = setup.tasks.frames('main/0001')
    assert frames['000001']['anchor']['kind'] == 'mask' and frames['000001']['mask_object']
    assert frames['000009']['anchor']['kind'] == 'empty'
    holder = lock_holder(setup.task.workspace)
    assert holder['labeler'] == 'alice' and holder['pid'] == 111 and holder['display'] == ':2'
    with pytest.raises(TaskError) as error:
        setup.open(setup.bob)
    assert error.value.code == 'locked'
    with pytest.raises(TaskLocked):  # the original window cannot open it either
        TaskLock.acquire(setup.task.workspace, 'carol')
    assert setup.tasks.close_task('main/0001', setup.alice, 1)
    assert lock_holder(setup.task.workspace) is None
    assert setup.open(setup.bob)['lease_epoch'] == 2  # epochs only grow


def test_a_task_open_in_the_original_window_is_refused(setup):
    lock = TaskLock.acquire(setup.task.workspace, 'carol')
    with pytest.raises(TaskError) as error:
        setup.open()
    assert error.value.code == 'locked' and 'carol' in str(error.value)
    lock.release()
    assert setup.open()['sync'] == 'imported'


def test_a_click_edit_makes_an_anchor_and_bumps_both_revisions(setup):
    setup.open()
    before = setup.revisions()
    mask = ellipse_mask(3, 10)
    reply = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000004', mask=mask,
                                     anchor='edit', request_id='r1', op='click')
    assert reply['changed'] and reply['anchor']['origin'] == 'cutie' and reply['anchor']['by'] == 'alice'
    assert setup.revisions() == (before[0] + 1, before[1] + 1)
    assert np.array_equal(setup.task.mask('000004'), mask)  # the legacy file, palette PNG
    assert setup.task.anchors()['000004']['kind'] == 'mask'
    assert setup.task.progress()['edited_by'] == ['alice']
    assert task_status(setup.task.workspace) == 'in progress'
    # editing a Label Studio anchor keeps its source
    reply = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000001',
                                     mask=np.zeros_like(mask), anchor='edit')
    assert reply['anchor'] == dict(reply['anchor'], kind='empty', origin='ls_refined',
                                   ls_file='label_0.json')


def test_the_same_request_is_applied_once(setup):
    setup.open()
    mask = ellipse_mask(2, 10)
    first = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000003', mask=mask,
                                     anchor='edit', request_id='same')
    revisions = setup.revisions()
    again = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000003', mask=mask,
                                     anchor='edit', request_id='same')
    assert again['replayed'] and again['content_revision'] == first['content_revision']
    assert setup.revisions() == revisions
    edits = setup.db.query("SELECT * FROM edits WHERE request_id = 'same'")
    assert len(edits) == 1


def test_propagation_never_overwrites_anchors_and_stale_results_are_rejected(setup):
    setup.open()
    input_revision, content_revision = setup.revisions()
    mask = ellipse_mask(4, 10)

    def commit(frame, revision=input_revision, current=True, epoch=1):
        return setup.tasks.commit_propagated('main/0001', setup.alice, epoch, revision, frame,
                                             mask, 'job', 'job.1', lambda: current)

    reply = commit('000005')
    assert setup.revisions() == (input_revision, content_revision + 1)  # inputs unchanged
    assert np.array_equal(setup.task.mask('000005'), mask)
    assert setup.tasks.frames('main/0001')['000005']['anchor'] is None
    for frame, kwargs, reason in (('000001', {}, 'anchor'), ('000006', {'revision': 0}, 'inputs'),
                                  ('000006', {'current': False}, 'obsolete')):
        with pytest.raises(TaskError) as error:
            commit(frame, **kwargs)
        assert error.value.code == 'stale', reason
    assert setup.task.mask('000006') is None  # nothing written
    setup.tasks.close_task('main/0001', setup.alice, 1)
    with pytest.raises(TaskError) as error:
        commit('000006')
    assert error.value.code == 'lease_lost'


def test_a_failed_transaction_changes_nothing(setup):
    setup.open()
    revisions = setup.revisions()
    original = setup.task.mask('000001')
    setup.db.fail_next_commit = OSError('disk full')
    with pytest.raises(OSError):
        setup.tasks.commit_frame('main/0001', setup.alice, 1, '000001',
                                 mask=np.zeros_like(original), anchor='edit', request_id='r9')
    assert setup.revisions() == revisions
    assert np.array_equal(setup.task.mask('000001'), original)  # files follow the database
    assert setup.db.query("SELECT * FROM requests WHERE request_id = 'r9'") == []
    # the retry of the same request is applied normally
    setup.tasks.commit_frame('main/0001', setup.alice, 1, '000001', mask=np.zeros_like(original),
                             anchor='edit', request_id='r9')
    assert setup.task.mask('000001').max() == 0


def test_objects_are_immutable_and_written_atomically(setup):
    data = encode_mask(ellipse_mask(1, 10))
    sha = setup.objects.put(data)
    assert setup.objects.put(data) == sha  # same content, same object
    file_path = setup.objects.path(sha)
    assert os.stat(file_path).st_mode & 0o777 == 0o444
    assert np.array_equal(decode_mask(setup.objects.get(sha)), ellipse_mask(1, 10))
    assert not [name for name in os.listdir(path.dirname(file_path)) if name.endswith('.tmp')]


def test_changes_by_the_original_window_are_imported_again(setup):
    setup.open()
    setup.tasks.commit_frame('main/0001', setup.alice, 1, '000003', mask=ellipse_mask(2, 10),
                             anchor='edit')
    setup.tasks.close_task('main/0001', setup.alice, 1)
    # the original window edits the task while the service does not hold it
    lock = TaskLock.acquire(setup.task.workspace, 'carol')
    state = TaskState(setup.task.workspace, 'carol')
    state.set_anchor('000007', 'empty', edited=True)
    state.remove_anchor('000003')
    os.remove(path.join(setup.task.masks_dir, '000003.png'))
    lock.release()
    before = setup.revisions()
    reply = setup.open(setup.bob)
    assert reply['sync'] == 'reimported'
    anchors = setup.tasks.anchors('main/0001')
    assert anchors['000007']['by'] == 'carol' and '000003' not in anchors
    assert '000003' not in setup.tasks.frames('main/0001')
    assert setup.revisions() == (before[0] + 1, before[1] + 1)
    # nothing changed outside since: the next open uses the database as it is
    setup.tasks.close_task('main/0001', setup.bob, reply['lease_epoch'])
    assert setup.open()['sync'] == 'current'


def test_an_interrupted_view_update_is_completed(setup, monkeypatch):
    setup.open()
    mask = ellipse_mask(6, 10)

    def broken(self, frame, data):
        raise OSError('disk full')

    monkeypatch.setattr(TaskViews, 'write_mask', broken)
    reply = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000007', mask=mask,
                                     anchor='edit')
    assert reply['views_written'] is False  # committed; the file is behind
    monkeypatch.undo()
    row = setup.tasks.task_row('main/0001')
    assert row['view_revision'] < row['content_revision']
    assert setup.task.mask('000007') is None
    setup.tasks.close_task('main/0001', setup.alice, 1)
    assert setup.open()['sync'] == 'resynced'
    assert np.array_equal(setup.task.mask('000007'), mask)


def test_remove_and_restore_a_frame(setup):
    setup.open()
    with open(path.join(setup.task.masks_dir, '000001.png'), 'rb') as f:
        original_bytes = f.read()
    original_anchor = setup.task.anchors()['000001']
    setup.tasks.commit_frame('main/0001', setup.alice, 1, '000001', remove_mask=True,
                             anchor='remove')
    assert setup.task.mask('000001') is None and '000001' not in setup.task.anchors()
    reply = setup.tasks.commit_frame('main/0001', setup.alice, 1, '000001',
                                     mask_bytes=original_bytes, anchor='restore',
                                     restore_anchor=original_anchor)
    assert reply['anchor'] == original_anchor
    with open(path.join(setup.task.masks_dir, '000001.png'), 'rb') as f:
        assert f.read() == original_bytes  # undo restores the very file


def test_task_data_review_and_reset(setup):
    setup.open()
    setup.tasks.set_intervals('main/0001', setup.alice, 1, [[5, 8], [1, 3]])
    with open(path.join(setup.task.workspace, 'interval.json')) as f:
        assert json.load(f) == {'intervals': [{'start': 1, 'end': 3}, {'start': 5, 'end': 8}]}
    setup.tasks.set_description('main/0001', setup.alice, 1, ' The car. ')
    with open(path.join(setup.task.workspace, 'description.json')) as f:
        assert json.load(f)['text'] == 'The car.'
    revision = setup.revisions()[1]
    review = setup.tasks.submit_for_review('main/0001', setup.alice, 1, [1])
    assert review['review_revision'] == revision
    assert task_status(setup.task.workspace) == 'under review'
    binary = read_mask(path.join(setup.task.workspace, 'binary_masks', '000001.png'))
    assert set(np.unique(binary)) <= {0, 255}
    assert open(path.join(setup.task.workspace, 'binary_masks', 'interval.txt')).read().startswith(
        'interval_count=2')
    # any change reopens the task
    setup.tasks.commit_frame('main/0001', setup.alice, 1, '000002', mask=ellipse_mask(1, 10),
                             anchor='edit')
    assert task_status(setup.task.workspace) == 'in progress'
    reply = setup.tasks.reset_all('main/0001', setup.alice, 1, [])
    assert reply['frames_cleared'] >= 3
    assert setup.task.anchors() == {} and os.listdir(setup.task.masks_dir) == []
    assert json.load(open(path.join(setup.task.workspace, 'prefill.json'))) == {}


def test_the_database_must_be_local(tmp_path, monkeypatch):
    from cutieplus.storage import db as db_module
    from cutieplus.storage.db import StorageError
    monkeypatch.setattr(db_module, 'filesystem_type', lambda directory: 'nfs4')
    with pytest.raises(StorageError, match='network filesystem'):
        Database(str(tmp_path / 'x.db'))
