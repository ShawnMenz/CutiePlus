"""
A sequence's work folder holds masks/ and metadata.json only (gui/task_state.py): every record of
the task is a section of metadata.json, written as it is labeled, and the lock that keeps a task
in one window lives in a folder every window of the account shares, outside the output.
"""
import os
from os import path

import pytest

from gui.task_state import (DISCARD, METADATA_FILE, PROGRESS, SECTIONS, TaskLock, TaskState,
                            discard_task, lock_path, read_metadata, write_sections)
from tests.conftest import compose_gui_config
from tests.helpers import make_task


def test_sections_are_written_in_order_and_none_removes_one(tmp_path):
    task = make_task(str(tmp_path))
    write_sections(task.workspace, attention={'at': 'now', 'by': 'alice'})
    write_sections(task.workspace, progress={'last_edited': 'now'})
    metadata = read_metadata(task.workspace)
    assert list(metadata) == [key for key in SECTIONS if key in metadata]  # the file's order
    assert list(metadata) == ['source', 'progress', 'attention', 'anchors']
    write_sections(task.workspace, attention=None)
    assert 'attention' not in read_metadata(task.workspace)
    with pytest.raises(ValueError):
        write_sections(task.workspace, notes='x')


def test_a_discard_from_the_browser_survives_the_windows_next_save(tmp_path):
    # the window's TaskState was loaded before the discard; its next anchor rereads the file
    task = make_task(str(tmp_path))
    window = TaskState(task.workspace, 'alice')
    discard_task(task.workspace, 'bob', True)
    window.set_anchor('000003', 'empty', edited=True)
    metadata = read_metadata(task.workspace)
    assert metadata[DISCARD]['by'] == 'bob'
    assert '000003' in metadata['anchors'] and metadata[PROGRESS]['last_edited_by'] == 'alice'


def test_the_lock_lives_outside_the_output(tmp_path, lock_folder):
    task = make_task(str(tmp_path))
    lock = TaskLock.acquire(task.workspace, 'alice')
    try:
        held = lock_path(task.workspace)
        assert path.dirname(held) == str(lock_folder)
        assert os.stat(held).st_mode & 0o777 == 0o600
        assert path.basename(held).startswith('0001-') and path.getsize(held) > 0
        assert sorted(os.listdir(task.workspace)) == ['masks', METADATA_FILE]
    finally:
        lock.release()
    # two sequences of the same name in two tabs have a lock each
    other = make_task(str(tmp_path), split='long')
    assert lock_path(other.workspace) != lock_path(task.workspace)


@pytest.mark.models
@pytest.mark.slow
def test_labeling_and_saving_leave_masks_and_metadata_only(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=8)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        assert gui.export_binary_button.text() == 'Save (Ctrl+S)'
        assert gui.save_visualization_combo.isHidden() and gui.save_soft_mask_checkbox.isHidden()
        gui.tl_slider.setValue(5)
        controller.on_reset_mask()  # an edit: an empty anchor
        gui.tl_slider.setValue(0)
        controller.on_forward_propagation()  # up to the empty anchor
        assert controller.on_save()
        assert 'is under review' in gui.console.toPlainText()
    finally:
        gui.close()
    assert sorted(os.listdir(task.workspace)) == ['masks', METADATA_FILE]
    assert len(os.listdir(task.masks_dir)) > 2  # the anchors and the propagated frames
    metadata = read_metadata(task.workspace)
    assert list(metadata) == ['source', 'intervals', 'progress', 'anchors']
    assert metadata['anchors']['000006']['kind'] == 'empty'
    assert metadata[PROGRESS]['review_by'] == 'tester'


def test_every_window_of_the_account_shares_the_lock_folder(monkeypatch):
    # not $HOME or $XDG_RUNTIME_DIR: a remote desktop's service has other values for them
    import pwd
    from gui.task_state import lock_folder
    monkeypatch.delenv('CUTIEPLUS_LOCK_DIR')
    monkeypatch.setenv('HOME', '/nowhere')
    monkeypatch.setenv('XDG_RUNTIME_DIR', '/nowhere/run')
    assert lock_folder() == path.join(pwd.getpwuid(os.getuid()).pw_dir, '.cache', 'cutieplus',
                                      'task-locks')
