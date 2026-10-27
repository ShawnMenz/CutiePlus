"""
The task browser's keys: D marks the selected sequence done (as the Mark done button), except
while the filter box has the focus, where it is just a letter.
"""
from gui.task_state import submit_for_review, task_status
from tests.conftest import compose_gui_config
from tests.helpers import make_task


def test_d_marks_the_selected_sequence_done_but_types_in_the_filter(tmp_path, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui.task_browser import TaskBrowser
    task = make_task(str(tmp_path))
    submit_for_review(task.workspace, 'alice')
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    try:
        browser.show()
        browser.activateWindow()
        assert QTest.qWaitForWindowActive(browser, 5000)

        # in the filter box, D is a letter
        browser.search.setFocus()
        QTest.keyClick(browser.search, Qt.Key.Key_D)
        assert browser.search.text() == 'd'
        assert task_status(task.workspace) == 'under review'
        browser.search.clear()

        # on the list: the selected sequence is marked done
        table = browser.tables['main']
        table.selectRow(0)
        table.setFocus()
        QTest.keyClick(table, Qt.Key.Key_D)
        assert task_status(task.workspace) == 'done', browser.console.toPlainText()
    finally:
        browser.close()
