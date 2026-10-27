"""
The task browser's Discard button: it asks first, shows the sequence as discarded
(dark gray, italics), hides it with Hide done / discarded, refuses Mark done, and reads Restore
on a discarded sequence. Export leaves a discarded sequence out (tests/test_dataset_export.py).
"""
from gui.task_state import discard_of, submit_for_review, task_status
from tests.conftest import compose_gui_config
from tests.helpers import make_task


def test_discard_and_restore_from_the_browser(tmp_path, qapp):
    from gui.task_browser import STATUS_COLUMN, TaskBrowser
    task = make_task(str(tmp_path))
    submit_for_review(task.workspace, 'alice')
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    asked = []
    answer = [False]
    browser.confirm_discard = lambda split, sequence: asked.append(sequence) or answer[0]
    try:
        table = browser.tables['main']
        table.selectRow(0)
        assert browser.discard_button.text() == 'Discard'

        browser.on_discard()  # declined: nothing changes
        assert asked == ['0001'] and discard_of(task.workspace) is None

        answer[0] = True
        browser.on_discard()
        item = table.item(0, STATUS_COLUMN)
        assert item.text() == 'discarded' and item.font().italic()
        assert item.foreground().color() == browser.status_color('discarded')
        assert 'discarded by tester' in item.toolTip()
        assert browser.discard_button.text() == 'Restore'
        assert 'is discarded: it is left out of the dataset' in browser.console.toPlainText()

        browser.on_mark_done()
        assert 'is discarded: restore it first' in browser.console.toPlainText()
        assert task_status(task.workspace) == 'under review'

        browser.hide_done.setChecked(True)
        assert table.isRowHidden(0)
        browser.hide_done.setChecked(False)

        table.selectRow(0)
        browser.on_discard()  # Restore: no question asked
        assert asked == ['0001', '0001'] and discard_of(task.workspace) is None
        assert table.item(0, STATUS_COLUMN).text() == 'under review'
        assert browser.discard_button.text() == 'Discard'
    finally:
        browser.close()
