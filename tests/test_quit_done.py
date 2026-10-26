"""
Ctrl+Q in the labeling window, for a labeler who labels and reviews at the same time: after a
warning, it saves the binary PNGs (submitting the task), exports the task, marks it done with the
time Ctrl+Q was pressed (quit_at), and closes the window. The shared service's side is in
test_export_review.py.
"""
import time
from os import path

import pytest

from gui.task_state import TaskState, task_status
from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


def open_window(task, networks, events=None, answer=True, asked=None):
    from gui.main_controller import MainController
    controller = MainController(
        compose_gui_config(task.workspace), networks=networks,
        on_task_event=None if events is None else lambda _, event: events.append(event))
    if asked is not None:
        asked.clear()
    controller.gui.confirm_quit_done = lambda: (asked.append(True) if asked is not None
                                                else None) or answer
    return controller


def wait_for_export(qapp, controller, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while controller.export_process is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert controller.export_process is None, controller.gui.console.toPlainText()


def test_ctrl_q_saves_exports_marks_done_and_closes(tmp_path, qapp, cpu_networks):
    from PySide6.QtGui import QKeySequence, QShortcut
    from gui.gui_utils import TextBox
    task = make_task(str(tmp_path), frame_count=6)
    events, asked = [], []
    controller = open_window(task, cpu_networks, events, asked=asked)
    keys = [shortcut.key() for shortcut in controller.gui.findChildren(QShortcut)]
    assert QKeySequence('Ctrl+Q') in keys and QKeySequence('Q') in keys
    assert QKeySequence('Ctrl+Q') in TextBox.WINDOW_KEYS  # also while typing a description

    before = time.strftime('%Y')
    controller.on_quit_done()
    assert asked == [True]
    assert task_status(task.workspace) == 'under review'  # submitted, the export running
    wait_for_export(qapp, controller)

    assert controller.closed, controller.gui.console.toPlainText()
    assert events[-2:] == ['done', 'closed']  # the browser refreshes the row
    progress = task.progress()
    assert task_status(task.workspace) == 'done' and progress['done_by'] == 'tester'
    assert progress['quit_at'].startswith(before) and progress['quit_at'] <= progress['done_at']
    assert path.isfile(path.join(task.workspace, 'binary_masks', 'interval.txt'))
    assert path.isfile(path.join(str(tmp_path), 'exports', 'cutie_masks_visual', task.sequence,
                                 'metadata.json'))


def test_cancel_leaves_everything_as_it_was(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    asked = []
    controller = open_window(task, cpu_networks, answer=False, asked=asked)
    try:
        controller.on_quit_done()
        assert asked == [True] and controller.export_process is None
        assert not controller.closed and task_status(task.workspace) == 'not started'
        assert not path.exists(path.join(task.workspace, 'binary_masks'))
    finally:
        controller.gui.close()


def test_a_task_that_needs_attention_is_refused_without_asking(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    TaskState(task.workspace, 'alice').save_attention(True)
    asked = []
    controller = open_window(task, cpu_networks, asked=asked)
    try:
        controller.on_quit_done()
        assert asked == [] and not controller.closed
        assert 'needs attention' in controller.gui.console.toPlainText()
        assert task_status(task.workspace) == 'not started'
    finally:
        controller.gui.close()


def test_an_unfinished_interval_is_refused_without_asking(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=12)
    asked = []
    controller = open_window(task, cpu_networks, asked=asked)
    try:
        controller.gui.add_interval_pair(start=8)  # no end yet
        controller.on_quit_done()
        assert asked == [] and not controller.closed
        assert 'Not marked done: 1 interval not saved' in controller.gui.console.toPlainText()
        assert task_status(task.workspace) != 'done'
    finally:
        controller.gui.close()


def test_an_edit_during_the_export_keeps_the_window_open_and_not_done(tmp_path, qapp,
                                                                      cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    events = []
    controller = open_window(task, cpu_networks, events)
    try:
        controller.on_quit_done()
        controller.task.touch()  # an edit while the export runs
        wait_for_export(qapp, controller)
        assert not controller.closed and 'done' not in events
        assert task_status(task.workspace) == 'in progress'
        assert 'quit_at' not in task.progress()
        assert 'Not marked done: the task is in progress' in controller.gui.console.toPlainText()
    finally:
        controller.gui.close()


def test_the_warning_names_the_risk_and_cancel_is_the_default(tmp_path, qapp, cpu_networks,
                                                              monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    shown = []
    monkeypatch.setattr(QMessageBox, 'exec', lambda dialog: shown.append(dialog) or 0)
    try:
        assert controller.gui.confirm_quit_done() is False  # closed without a click
        dialog, = shown
        assert dialog.text() == "You are about to quit and save the intervals of 0001 as 'done'."
        assert 'Intervals: 1-6.' in dialog.informativeText()
        assert 'no separate review' in dialog.informativeText()
        assert dialog.icon() == QMessageBox.Icon.Warning
        assert dialog.defaultButton().text() == 'Cancel'
    finally:
        controller.gui.close()
