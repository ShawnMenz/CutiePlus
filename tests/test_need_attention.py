"""
Need Attention in the labeling window (a toggle, orange) and the task browser's Attention status
(red); Mark done waits until the flag is off. The shared service's side is in
test_export_review.py.
"""
import pytest

from gui.task_state import ATTENTION_STATUS, TaskState, attention_of, task_status
from tests.conftest import compose_gui_config
from tests.helpers import make_task


@pytest.mark.models
@pytest.mark.slow
def test_the_button_toggles_the_flag(tmp_path, qapp, cpu_networks):
    from gui.gui import ATTENTION_COLOR
    from gui.style import style_sheet
    from gui.main_controller import MainController
    task = make_task(str(tmp_path))
    events = []

    def open_window():
        return MainController(compose_gui_config(task.workspace), networks=cpu_networks,
                              on_task_event=lambda _, event: events.append(event))

    controller = open_window()
    button = controller.gui.attention_button
    assert button.text() == 'Need Attention' and button.isCheckable() and not button.isChecked()
    # orange, filled while on (the attention variant of gui/style.py)
    assert button.property('variant') == 'attention' and ATTENTION_COLOR in style_sheet(False)
    button.click()
    assert button.isChecked() and attention_of(task.workspace)['by'] == 'tester'
    assert events == ['attention']  # the browser refreshes the row
    assert 'Marked as needing attention.' in controller.gui.console.toPlainText()
    assert 'flagged by tester' in button.toolTip()
    controller.gui.close()

    controller = open_window()  # a reopened window shows the flag
    button = controller.gui.attention_button
    assert button.isChecked()
    button.click()
    assert not button.isChecked() and attention_of(task.workspace) is None
    assert 'Attention cleared.' in controller.gui.console.toPlainText()
    controller.gui.close()


def test_the_browser_shows_attention_in_red_and_does_not_mark_it_done(tmp_path, qapp):
    from gui.task_browser import STATUS_COLUMN, STATUS_COLORS, TaskBrowser
    task = make_task(str(tmp_path))
    TaskState(task.workspace, 'alice').save_attention(True)
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    try:
        table = browser.tables['main']
        item = table.item(0, STATUS_COLUMN)
        assert item.text() == ATTENTION_STATUS
        assert item.foreground().color() == STATUS_COLORS[ATTENTION_STATUS]
        assert 'needs attention, flagged by alice' in item.toolTip()
        assert f'{ATTENTION_STATUS} 1' in browser.stats.text()
        table.selectRow(0)
        browser.on_mark_done()
        assert 'needs attention' in browser.console.toPlainText()
        assert task_status(task.workspace) == 'not started'
    finally:
        browser.close()
