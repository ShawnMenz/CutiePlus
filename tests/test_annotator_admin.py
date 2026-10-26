"""
The admin of the annotator list (admin_port in annotators.json): only a session on the admin's port
may change the list; every other session sees the page read-only, as the Annotator list. The
admin's row and the row of the screen the page is on cannot be removed, and a list without the
admin's port is never written.
"""
import pytest

from gui.annotators import (load_admin_port, load_annotators, may_manage, save_annotators,
                            set_admin)
from tests.conftest import compose_gui_config
from tests.helpers import make_task


class NoScreens:
    def sync(self, saved, removed):
        return {}


def team(root):
    save_annotators(root, [{'port': 5902, 'name': 'jenny'}, {'port': 5903, 'name': 'abhignya'},
                           {'port': 5904, 'name': 'zexuan'}])
    set_admin(root, 5904)


def open_manager(root):
    from gui.annotator_manager import AnnotatorManager
    return AnnotatorManager(root, screens=NoScreens())


def row_of(dialog, port):
    return next(row for row in dialog.rows if row.port_box.value() == port)


def test_the_admin_is_kept_and_cannot_be_dropped(tmp_path):
    root = str(tmp_path)
    assert may_manage(root, ':3')  # no admin yet: everyone may, as before
    team(root)
    assert load_admin_port(root) == 5904
    assert may_manage(root, ':4') and not may_manage(root, ':3') and not may_manage(root, ':1')
    save_annotators(root, [{'port': 5904, 'name': 'zexuan'}])  # others removed: the admin stays
    assert load_admin_port(root) == 5904
    with pytest.raises(ValueError, match='Port 5904 is the admin and cannot be removed'):
        save_annotators(root, [{'port': 5902, 'name': 'jenny'}])
    with pytest.raises(ValueError, match='not on the annotator list'):
        set_admin(root, 5909)


def test_users_see_a_read_only_list(tmp_path, qapp, monkeypatch):
    root = str(tmp_path)
    team(root)
    monkeypatch.setenv('DISPLAY', ':3')
    dialog = open_manager(root)
    assert dialog.windowTitle() == 'Annotator list'
    assert dialog.add_button.isHidden() and dialog.save_button.isHidden()
    for row in dialog.rows:
        assert row.name_box.isReadOnly() and row.port_box.isReadOnly()
        assert row.remove_button.isHidden()
    assert not dialog.save()  # even if called: nothing written
    assert 'only the admin' in dialog.status.text()
    assert len(load_annotators(root)) == 3


def test_the_admin_manages_but_cannot_remove_their_own_row(tmp_path, qapp, monkeypatch):
    root = str(tmp_path)
    team(root)
    monkeypatch.setenv('DISPLAY', ':4')
    dialog = open_manager(root)
    assert dialog.windowTitle() == 'Manage annotators'
    assert not dialog.save_button.isHidden()
    admin, user = row_of(dialog, 5904), row_of(dialog, 5902)
    # no Remove: the Admin label takes its place, as wide as the Remove of the other rows
    assert admin.remove_button.isHidden() and not admin.port_box.isEnabled()
    assert admin.port_box.toolTip() == 'The admin cannot be removed.'
    dialog.show()
    qapp.processEvents()
    assert admin.admin_label.geometry().left() == user.remove_button.geometry().left()
    assert admin.admin_label.width() == user.remove_button.width()
    assert user.remove_button.isEnabled() and user.admin_label is None
    dialog.remove_row(admin)  # however it is taken out, Save refuses
    assert 'Port 5904 is the admin and cannot be removed.' in dialog.status.text()
    assert not dialog.save() and len(load_annotators(root)) == 3


def test_without_an_admin_the_own_screen_is_locked(tmp_path, qapp, monkeypatch):
    root = str(tmp_path)
    save_annotators(root, [{'port': 5902, 'name': 'jenny'}, {'port': 5903, 'name': 'abhignya'}])
    monkeypatch.setenv('DISPLAY', ':2')
    dialog = open_manager(root)
    own = row_of(dialog, 5902)
    assert not own.remove_button.isEnabled()
    assert 'cannot remove itself' in own.remove_button.toolTip()
    assert row_of(dialog, 5903).remove_button.isEnabled()


def test_the_browser_button_follows_the_role(tmp_path, qapp, monkeypatch):
    from gui.task_browser import TaskBrowser
    task = make_task(str(tmp_path))
    team(task.root)
    monkeypatch.setenv('DISPLAY', ':3')
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    try:
        assert browser.manage_button.text() == 'Annotator list'
        monkeypatch.setenv('DISPLAY', ':4')
        browser.refresh_identity()
        assert browser.manage_button.text() == 'Manage annotators'
    finally:
        browser.close()
