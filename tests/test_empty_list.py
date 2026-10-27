"""
The task browser says why its list is empty, in the middle of the list (both ways): no subset
imported yet; every sequence done or discarded, with Hide done / discarded ticked; or none
matching the filter. A list with a sequence to show has no message.
"""
import os
from os import path

from PySide6.QtCore import Qt

from gui.task_state import discard_task, mark_done, submit_for_review
from tests.conftest import compose_gui_config
from tests.helpers import make_task

CENTERED = Qt.AlignmentFlag.AlignCenter


def browser_on(root):
    from gui.task_browser import TaskBrowser
    return TaskBrowser(compose_gui_config(device='cpu'), root)


def test_without_any_subset_the_list_says_no_data_imported(tmp_path, qapp):
    root = str(tmp_path / 'cutie_tasks')
    os.makedirs(root)
    browser = browser_on(root)
    try:
        assert browser.task_pages.currentWidget() is browser.no_data
        assert browser.no_data.text() == 'No data imported.\nPress Import to add a subset.'
        assert browser.no_data.alignment() == CENTERED
    finally:
        browser.close()
    os.makedirs(path.join(root, 'visual', '0001'))  # input waiting in the root
    browser = browser_on(root)
    try:
        assert 'Press Import, then Save' in browser.no_data.text()
    finally:
        browser.close()


def test_a_tab_with_everything_done_or_discarded_says_so_when_they_are_hidden(tmp_path, qapp):
    done = make_task(str(tmp_path), sequence='0001', frame_count=4)
    submit_for_review(done.workspace, 'alice')
    mark_done(done.workspace, 'alice')
    discarded = make_task(str(tmp_path), sequence='0002', frame_count=4)
    discard_task(discarded.workspace, 'alice', True)
    browser = browser_on(done.root)
    table = browser.tables['main']
    try:
        browser.show()
        assert browser.task_pages.currentWidget() is browser.tabs
        assert table.notice.isHidden()  # the done rows are listed
        browser.hide_done.setChecked(True)
        assert not table.notice.isHidden()
        assert table.notice.text().startswith('Every sequence in main is done or discarded')
        assert table.notice.alignment() == CENTERED
        assert table.notice.geometry() == table.viewport().rect()  # the middle of the list
        browser.hide_done.setChecked(False)
        assert table.notice.isHidden()
        browser.search.setText('zzz')
        assert table.notice.text() == 'No sequence matches "zzz".'
    finally:
        browser.close()


def test_a_sequence_left_to_label_keeps_the_list(tmp_path, qapp):
    done = make_task(str(tmp_path), sequence='0001', frame_count=4)
    submit_for_review(done.workspace, 'alice')
    mark_done(done.workspace, 'alice')
    make_task(str(tmp_path), sequence='0002', frame_count=4)  # not started
    browser = browser_on(done.root)
    try:
        browser.hide_done.setChecked(True)
        assert browser.tables['main'].notice.isHidden()
    finally:
        browser.close()
