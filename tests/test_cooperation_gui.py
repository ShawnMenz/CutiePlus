"""
The windows on a host with the cooperation service (gui/login_dialog.py, service_link.py,
manage_annotators.py, the task browser's account area): signing in and the first-time setup, the
offline way when the service is down, Manage Annotators for supervisors and admins only, what each
of them is offered, and a browser that asks for a new sign-in once its user's role changed. The
service runs in the test process on a real socket; the permission checks themselves are in
test_cooperation_authz.py.
"""
import time

import pytest

from cutieplus.cooperation.client import ServiceClient
from cutieplus.cooperation.server import Server
from tests.conftest import compose_gui_config
from tests.cooperation_harness import PASSWORD, Harness

pytestmark = pytest.mark.slow


def wait_for(qapp, predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError('the condition never held')
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def served(tmp_path):
    harness = Harness(tmp_path)
    socket_path = str(tmp_path / 'app.sock')
    server = Server(harness.service.handle, socket_path, mode=0o600)
    server.start()
    harness.socket_path = socket_path
    harness.owner_token = harness.owner()
    yield harness
    server.stop()


def signed_in(served, login: str):
    from gui.service_link import ServiceLink
    client = ServiceClient(served.socket_path, timeout=5)
    session = client.call('login', login_name=login, password=PASSWORD)
    return ServiceLink(client, session)


def browser_for(served, qapp, tmp_path, link=None, offline=False):
    from gui.task_browser import TaskBrowser
    root = tmp_path / 'tasks'
    root.mkdir(exist_ok=True)
    return TaskBrowser(compose_gui_config(), str(root), link=link, offline=offline)


def test_signing_in_and_setting_up_the_first_password(served, qapp):
    from gui.login_dialog import LoginDialog
    created = served.call('create_user', served.owner_token, login_name='alice',
                          display_name='Alice')
    dialog = LoginDialog(ServiceClient(served.socket_path, timeout=5))
    dialog.login_edit.setText('alice')
    dialog.password_edit.setText('not set yet')
    dialog.sign_in()
    assert dialog.session is None and 'setup token' in dialog.message.text()
    dialog.toggle_setup()
    dialog.token_edit.setText(created['setup_token'])
    dialog.new_password_edit.setText('her own password')
    dialog.repeat_edit.setText('a different one')
    dialog.sign_in()
    assert dialog.session is None and 'differ' in dialog.message.text()
    dialog.new_password_edit.setText('her own password')
    dialog.repeat_edit.setText('her own password')
    dialog.sign_in()
    assert dialog.session['user']['login_name'] == 'alice'
    assert not dialog.new_password_edit.text()  # nothing kept in the boxes


def test_a_service_that_is_down_offers_to_work_offline(tmp_path, qapp):
    from gui.login_dialog import LoginDialog
    dialog = LoginDialog(ServiceClient(str(tmp_path / 'nothing.sock'), timeout=1))
    assert not dialog.sign_in_button.isEnabled()
    assert not dialog.offline_button.isHidden() and 'not reachable' in dialog.message.text()
    dialog.work_offline()
    assert dialog.offline


def test_only_supervisors_and_admins_get_manage_annotators(served, qapp, tmp_path):
    served.add(served.owner_token, 'alice')
    served.add(served.owner_token, 'sam', 'supervisor')
    browsers = {}
    for login in ('alice', 'sam', 'owner'):
        browsers[login] = browser = browser_for(served, qapp, tmp_path, signed_in(served, login))
        wait_for(qapp, lambda: browser.link.whoami is not None)
    try:
        assert browsers['alice'].manage_button.isHidden()
        assert browsers['alice'].labeler == 'alice'
        assert 'Annotator' in browsers['alice'].account_label.text()
        for login in ('sam', 'owner'):
            assert not browsers[login].manage_button.isHidden()
            assert browsers[login].manage_button.isEnabled()
        # a forced click does nothing either (and the service would refuse it)
        browsers['alice'].on_manage()
        assert browsers['alice'].manage_window is None
    finally:
        for browser in browsers.values():
            browser.close()


def test_a_supervisor_manages_annotators_without_a_role_selector(served, qapp):
    from gui.manage_annotators import AddUserDialog, ManageAnnotators
    alice = served.add(served.owner_token, 'alice')
    served.add(served.owner_token, 'sue', 'supervisor')
    served.add(served.owner_token, 'sam', 'supervisor')
    window = ManageAnnotators(signed_in(served, 'sam'))
    asked, secrets = [], []
    window.confirm = lambda title, text: asked.append(text) or True
    window.show_secret = lambda title, login, reply: secrets.append(reply)
    try:
        wait_for(qapp, lambda: window.users)
        assert [user['login_name'] for user in window.users] == ['alice']
        assert AddUserDialog(admin=False).role_combo is None
        window.selected_id = alice['id']
        window.load_details()
        wait_for(qapp, lambda: window.details is not None)
        texts = [button.text() for button in window.findChildren(type(window.add_button))]
        assert 'Suspend' in texts and 'Remove' in texts
        assert 'Make Supervisor' not in texts and 'Revoke Supervisor' not in texts
        next(button for button in window.findChildren(type(window.add_button))
             if button.text() == 'Suspend').click()
        assert 'their queued GPU jobs are cancelled' in asked[-1]
        assert served.service.user_row(alice['id'])['status'] == 'suspended'
        # adding: the reply's setup token and desktop are shown once
        AddUserDialog.exec = lambda self: True
        AddUserDialog.values = lambda self: {'login_name': 'eve', 'display_name': 'Eve'}
        window.on_add_user()
        assert secrets[-1]['setup_token'] and secrets[-1]['desktop']['vnc_password']
    finally:
        del AddUserDialog.exec, AddUserDialog.values
        window.close()


def test_an_admin_appoints_supervisors_after_entering_the_password_again(served, qapp):
    from gui.manage_annotators import ManageAnnotators
    alice = served.add(served.owner_token, 'alice')
    window = ManageAnnotators(signed_in(served, 'owner'))
    passwords = []
    window.confirm = lambda title, text: True
    window.ask_password = lambda: passwords.append(1) or PASSWORD
    try:
        wait_for(qapp, lambda: window.users)
        assert window.tabs.count() == 3  # Users, Settings, Audit
        served.clock.advance(minutes=30)  # the password entered at sign-in is not fresh now
        served.call('whoami', window.link.client.token)  # keeps the session from going idle
        window.selected_id = alice['id']
        window.load_details()
        wait_for(qapp, lambda: window.details is not None and
                 window.details['id'] == alice['id'])
        next(button for button in window.findChildren(type(window.add_button))
             if button.text() == 'Make Supervisor').click()
        assert passwords == [1]
        assert served.service.user_row(alice['id'])['role'] == 'supervisor'
    finally:
        window.close()


def test_a_role_change_sends_the_browser_back_to_sign_in(served, qapp, tmp_path):
    sam = served.add(served.owner_token, 'sam', 'supervisor')
    browser = browser_for(served, qapp, tmp_path, signed_in(served, 'sam'))
    reasons = []
    browser.sign_in_again.connect(reasons.append)
    try:
        wait_for(qapp, lambda: not browser.manage_button.isHidden())
        served.call('set_role', served.owner_token, user_id=sam['id'], role='annotator')
        browser.link.poll()
        wait_for(qapp, lambda: reasons)
        assert 'sign in again' in reasons[0] or 'ended' in reasons[0]
        assert not browser.isVisible()
    finally:
        browser.close()


def test_offline_edits_are_recorded_as_offline(served, qapp, tmp_path):
    browser = browser_for(served, qapp, tmp_path, offline=True)
    try:
        assert browser.labeler.endswith('(offline)')
        assert browser.manage_button.isHidden() and browser.logout_button.isHidden()
    finally:
        browser.close()


def test_a_session_that_ends_during_an_export_disables_the_browser_at_once(served, qapp,
                                                                          tmp_path):
    import threading
    from types import SimpleNamespace
    from PySide6.QtWidgets import QProgressDialog
    served.add(served.owner_token, 'alice')
    browser = browser_for(served, qapp, tmp_path, signed_in(served, 'alice'))
    reasons = []
    browser.sign_in_again.connect(reasons.append)
    cancel = threading.Event()
    browser.export_worker = SimpleNamespace(cancel=cancel)
    browser.export_progress = QProgressDialog(browser)
    try:
        browser.end_session('your session ended')
        # not closed yet (the export is stopping), but nothing more runs under the session
        assert cancel.is_set() and reasons == [] and browser.revoked
        assert not browser.tabs.isEnabled() and not browser.logout_button.isEnabled()
        browser.on_open()  # refused
        assert browser.controller is None
        browser.on_exported(None, 'cancelled')
        assert reasons == ['your session ended'] and not browser.isVisible()
    finally:
        browser.export_worker = None
        browser.close()
