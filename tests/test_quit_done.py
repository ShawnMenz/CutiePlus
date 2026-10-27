"""
Ctrl+Q in the labeling window, for a labeler who labels and reviews at the same time: after a
warning, it saves the task (submitting it), marks it done with the time Ctrl+Q was pressed
(quit_at), and closes the window. The work folder keeps masks/ and metadata.json only.
"""
import os
import time

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


def test_ctrl_q_saves_marks_done_and_closes(tmp_path, qapp, cpu_networks):
    from PySide6.QtGui import QKeySequence, QShortcut
    from gui.gui_utils import TextBox
    task = make_task(str(tmp_path), frame_count=6)
    events, asked = [], []
    controller = open_window(task, cpu_networks, events, asked=asked)
    keys = [shortcut.key() for shortcut in controller.gui.findChildren(QShortcut)]
    assert QKeySequence('Ctrl+Q') in keys and QKeySequence('Q') in keys
    assert QKeySequence('Ctrl+Q') in TextBox.WINDOW_KEYS  # also while typing a description

    assert not hasattr(controller.gui, 'export_task_button')  # the browser's Export does it
    before = time.strftime('%Y')
    controller.on_quit_done()
    assert asked == [True]
    qapp.processEvents()
    assert controller.closed, controller.gui.console.toPlainText()
    assert events[-2:] == ['done', 'closed']  # the browser refreshes the row
    progress = task.progress()
    assert task_status(task.workspace) == 'done' and progress['done_by'] == 'tester'
    assert progress['quit_at'].startswith(before) and progress['quit_at'] <= progress['done_at']
    assert sorted(os.listdir(task.workspace)) == ['masks', 'metadata.json']


def test_cancel_leaves_everything_as_it_was(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    asked = []
    controller = open_window(task, cpu_networks, answer=False, asked=asked)
    try:
        controller.on_quit_done()
        assert asked == [True]
        assert not controller.closed and task_status(task.workspace) == 'not started'
        assert sorted(os.listdir(task.workspace)) == ['masks', 'metadata.json']
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
