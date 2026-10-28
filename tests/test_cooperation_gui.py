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


def test_the_sign_in_switches_cleanly_between_password_and_setup_token(served, qapp):
    import os
    from gui.login_dialog import LoginDialog
    from gui.style import DANGER
    created = served.call('create_user', served.owner_token, login_name='ann',
                          display_name='Ann')
    # the window runs at Ann's remote desktop
    served.units[os.getpid()] = served.config.desktop_unit_of(created['desktop']['display'])
    dialog = LoginDialog(ServiceClient(served.socket_path, timeout=5))
    rows = {'password': dialog.password_edit, 'token': dialog.token_edit,
            'new': dialog.new_password_edit, 'repeat': dialog.repeat_edit}

    def shown() -> dict:
        qapp.processEvents()
        return {name: dialog.form.isRowVisible(edit) for name, edit in rows.items()}

    try:
        dialog.show()
        qapp.processEvents()
        first = dialog.size()
        assert 'remote desktop :2' in dialog.message.text()
        assert shown() == {'password': True, 'token': False, 'new': False, 'repeat': False}
        dialog.setup_toggle.click()
        # no second, greyed-out password box: the setup token's rows instead
        assert shown() == {'password': False, 'token': True, 'new': True, 'repeat': True}
        assert dialog.sign_in_button.text() == 'Set password and sign in'
        assert dialog.size().height() > first.height()
        dialog.login_edit.setText('ann')
        dialog.token_edit.setText('AAAAA-BBBBB-CCCCC-DDDDD')  # not hers
        dialog.new_password_edit.setText('her own password')
        dialog.repeat_edit.setText('her own password')
        dialog.sign_in()
        assert dialog.session is None and dialog.setting_up and shown()['token']
        assert 'setup token' in dialog.message.text()
        assert any(color in dialog.message.styleSheet() for color in DANGER)
        dialog.setup_toggle.click()
        assert shown()['password'] and dialog.sign_in_button.text() == 'Sign in'
        qapp.processEvents()
        assert dialog.size() == first  # back to its size: nothing left stretched
    finally:
        dialog.close()
        served.units.pop(os.getpid(), None)


def test_recent_activity_says_why_a_sign_in_failed(served, qapp):
    from gui.manage_annotators import describe_event
    base = {'created_at': '2026-10-27T20:19:00.000000Z', 'actor_login': None,
            'actor_source': 'system', 'result': 'failed'}
    assert describe_event(dict(base, action='setup_credential',
                               error_code='invalid_credentials')).endswith(
        'setup credential  failed (wrong, replaced or expired setup token)')
    assert describe_event(dict(base, action='login', error_code='wrong_desktop')).endswith(
        "(at another user's desktop)")
    assert describe_event(dict(base, action='login', error_code='invalid_credentials')).endswith(
        '(wrong password)')
    assert describe_event(dict(base, action='create_user', result='succeeded',
                               error_code=None)).endswith('create user  succeeded')
    # a real refusal, as the user's details show it to whoever manages them
    created = served.call('create_user', served.owner_token, login_name='ann',
                          display_name='Ann')
    assert served.code('setup_credential', login_name='ann', setup_token='AAAAA-BBBBB-CCCCC-DDDDD',
                       new_password='her own password') == 'invalid_credentials'
    details = served.call('get_user', served.owner_token, user_id=created['user']['id'])
    lines = [describe_event(event) for event in details['recent_activity']]
    assert any(line.endswith('setup credential  failed (wrong, replaced or expired setup token)')
               for line in lines), lines


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


def test_a_supervisor_manages_annotators_without_a_role_selector(served, qapp, monkeypatch):
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
        monkeypatch.setattr(AddUserDialog, 'exec', lambda self: True)
        monkeypatch.setattr(AddUserDialog, 'values',
                            lambda self: {'login_name': 'eve', 'display_name': 'Eve'})
        window.on_add_user()
        assert secrets[-1]['setup_token'] and secrets[-1]['desktop']['vnc_password']
    finally:
        window.close()


def test_the_details_open_on_a_click_and_close(served, qapp):
    from PySide6.QtWidgets import QPushButton
    from gui.manage_annotators import ManageAnnotators
    alice = served.add(served.owner_token, 'alice')
    window = ManageAnnotators(signed_in(served, 'owner'))

    def click_alice() -> None:
        window.user_table.selectRow(next(row for row, user in enumerate(window.users)
                                         if user['id'] == alice['id']))
        wait_for(qapp, lambda: window.details is not None)

    def refreshed() -> None:
        window.refresh()
        wait_for(qapp, lambda: not window.loading)

    try:
        wait_for(qapp, lambda: window.users)
        assert window.details_panel.isHidden()  # the list takes the whole width
        click_alice()
        assert not window.details_panel.isHidden()
        assert window.details_title.text() == 'Alice (alice)'
        next(button for button in window.details_panel.findChildren(QPushButton)
             if button.text() == '×').click()
        assert window.details_panel.isHidden() and window.selected_id is None
        assert not window.user_table.selectionModel().selectedRows()
        refreshed()
        assert window.details_panel.isHidden()  # the refresh leaves it closed
        click_alice()
        assert not window.details_panel.isHidden()
        # the selected user leaves the list (removed, removed users hidden): the panel closes
        served.call('remove_user', served.owner_token, user_id=alice['id'])
        refreshed()
        assert window.details_panel.isHidden() and window.selected_id is None
    finally:
        window.close()


def test_a_reply_after_the_window_closed_changes_nothing(served, qapp):
    """Before: a closed Manage Annotators stayed alive, and a late reply redrew its details,
    deleting widgets of a closed window: a crash in Qt."""
    import shiboken6
    from gui.gpu_queue import GpuQueueWindow
    from gui.manage_annotators import ManageAnnotators
    alice = served.add(served.owner_token, 'alice')
    link = signed_in(served, 'owner')
    window = ManageAnnotators(link)
    wait_for(qapp, lambda: window.users)
    window.selected_id = alice['id']
    window.load_details()
    wait_for(qapp, lambda: window.details is not None)
    details, users = window.details, window.users
    window.close()
    window.on_users(users, None)  # replies on their way when it closed
    window.on_details(details, None)
    wait_for(qapp, lambda: not shiboken6.isValid(window))  # deleted
    window.on_details(details, None)
    window.on_identity(link.whoami)
    queue = GpuQueueWindow(link)
    queue.close()
    queue.on_status({'worker': {'state': 'idle'}, 'queue': [], 'mine': []}, None)
    wait_for(qapp, lambda: not shiboken6.isValid(queue))


def test_connection_info_any_time_and_new_secrets_on_request(served, qapp):
    from PySide6.QtWidgets import QPushButton
    from gui.manage_annotators import ManageAnnotators, SecretCard
    alice = served.add(served.owner_token, 'alice')
    window = ManageAnnotators(signed_in(served, 'owner'))
    shown, asked = [], []
    window.show_info = lambda *args: shown.append(args)
    window.confirm = lambda title, text: asked.append(title) or True
    try:
        wait_for(qapp, lambda: window.users)
        window.selected_id = alice['id']
        window.load_details()
        wait_for(qapp, lambda: window.details is not None)
        assert not window.info_button.isHidden()
        window.info_button.click()
        title, login, reply, actions = shown[-1]
        assert title == 'Connection info: alice' and login == 'alice'
        command = reply['desktop']['ssh_command']
        port = reply['desktop']['vnc_port']
        assert command.startswith('ssh -N -o ExitOnForwardFailure=yes -J cutieplus-jump@')
        assert f' -L {port}:localhost:{port} ' in command and command.endswith(' labeler@localhost')
        assert reply['desktop']['vnc_address'] == f'localhost:{port}'
        assert 'vnc_password' not in reply['desktop']  # never kept: made on request
        assert [label for label, _ in actions] == ['New VNC password', 'New setup token']
        card = SecretCard(title, login, reply, actions=actions)
        assert list(card.fields) == ['Login name', 'SSH tunnel', 'VNC viewer']
        assert 'shown only when they are made' in card.warning.text()
        for label in ('New VNC password', 'New setup token'):
            next(button for button in card.findChildren(QPushButton)
                 if button.text() == label).click()
        assert asked == ['New VNC password', 'Reset credential']  # each asked first
        assert list(card.fields) == ['Login name', 'SSH tunnel', 'VNC viewer', 'VNC password',
                                     'Setup token']
        assert card.fields['VNC password'].text() and card.fields['Setup token'].text()
        assert card.warning.text().startswith('Shown once')
        card.close()
    finally:
        window.close()


def test_copy_says_what_it_copied_in_green(qapp):
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QPushButton
    from gui.manage_annotators import SecretCard
    from gui.style import PRIMARY, SECONDARY
    card = SecretCard('alice is added', 'alice', {
        'setup_token': 'AAAAA-BBBBB-CCCCC-DDDDD',
        'setup_token_expires_at': '2026-10-30T00:00:00.000000Z',
        'desktop': {'ssh_command': ('ssh -N -o ExitOnForwardFailure=yes -J '
                                    'cutieplus-jump@203.0.113.10 -p 2222 '
                                    '-L 5902:localhost:5902 labeler@localhost'),
                    'vnc_address': 'localhost:5902', 'display': 2, 'vnc_password': 'abcd2345'}})
    try:
        assert card.copied.text() == ''
        said = {}
        for button in card.findChildren(QPushButton):
            if button.text() == 'Copy':
                button.click()
                said[QGuiApplication.clipboard().text()] = card.copied.text()
        assert said == {
            'alice': 'Login name is copied.',
            'AAAAA-BBBBB-CCCCC-DDDDD': 'Setup token is copied.',
            ('ssh -N -o ExitOnForwardFailure=yes -J cutieplus-jump@203.0.113.10 -p 2222 '
             '-L 5902:localhost:5902 labeler@localhost'):
                'SSH tunnel is copied.',
            'localhost:5902': 'VNC viewer is copied.',
            'abcd2345': 'VNC password is copied.'}
        assert PRIMARY in card.copied.styleSheet() or SECONDARY in card.copied.styleSheet()
    finally:
        card.close()


def fill_each_time(monkeypatch, dialog_class, fill):
    """Opens dialog_class as usual, with fill(form) run in each of its exec loops."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog

    def exec_and_fill(self):
        QTimer.singleShot(0, lambda: fill(self))
        return QDialog.exec(self)

    monkeypatch.setattr(dialog_class, 'exec', exec_and_fill)


def test_enter_in_the_add_form_adds_and_a_refusal_keeps_what_was_typed(served, qapp,
                                                                       monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui.manage_annotators import AddUserDialog, ManageAnnotators
    window = ManageAnnotators(signed_in(served, 'owner'))
    secrets, seen = [], []
    window.show_secret = lambda title, login, reply: secrets.append(login)

    def fill(form):
        seen.append({'error': form.error.text(), 'name': form.name_edit.text()})
        if len(seen) == 1:
            QTest.keyClick(form.login_edit, Qt.Key.Key_Return)  # nothing typed: nothing happens
            seen[-1]['open'] = form.isVisible()
            form.login_edit.setText('supervisor')  # a reserved name: refused
            form.name_edit.setText('Sam')
            form.role_combo.setCurrentIndex(form.role_combo.findData('supervisor'))
        elif len(seen) == 2:
            form.login_edit.setText('sam')
        else:
            form.reject()
            return
        QTest.keyClick(form.name_edit, Qt.Key.Key_Return)

    fill_each_time(monkeypatch, AddUserDialog, fill)
    try:
        wait_for(qapp, lambda: window.users)
        window.on_add_user()
        assert seen[0]['open']
        assert 'reserved' in seen[1]['error'] and seen[1]['name'] == 'Sam'
        assert secrets == ['sam'] and len(seen) == 2
        wait_for(qapp, lambda: any(user['login_name'] == 'sam' for user in window.users))
        assert {user['login_name']: user['role'] for user in window.users}['sam'] == 'supervisor'
    finally:
        window.close()


def test_a_refused_key_keeps_the_key_form_open(served, qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui.manage_annotators import AddKeyDialog, ManageAnnotators
    from tests.cooperation_harness import ed25519_key
    alice = served.add(served.owner_token, 'alice')
    window = ManageAnnotators(signed_in(served, 'owner'))
    seen = []

    def fill(form):
        seen.append(form.error.text())
        if len(seen) == 1:
            form.key_edit.setPlainText('ssh-dss AAAAB3NzaC1kc3M= old')
            form.label_edit.setText('laptop')
        elif len(seen) == 2:
            form.key_edit.setPlainText(ed25519_key(7))
        else:
            form.reject()
            return
        QTest.keyClick(form.label_edit, Qt.Key.Key_Return)

    fill_each_time(monkeypatch, AddKeyDialog, fill)
    try:
        window.add_key(served.service.user_row(alice['id']))
        assert 'not an accepted key type' in seen[1] and len(seen) == 2
        keys = served.service.db.query('SELECT device_label FROM ssh_keys WHERE user_id = ?',
                                       (alice['id'], ))
        assert [key['device_label'] for key in keys] == ['laptop']
    finally:
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


def test_an_admin_deletes_a_user_with_the_password_every_time(served, qapp):
    from PySide6.QtWidgets import QPushButton
    from gui.manage_annotators import ManageAnnotators
    alice = served.add(served.owner_token, 'alice')
    window = ManageAnnotators(signed_in(served, 'owner'))
    asked, passwords = [], []
    window.confirm = lambda title, text: asked.append(text) or True
    window.ask_password = lambda: passwords.append(1) or PASSWORD

    def details_of(user_id: str) -> list:
        window.selected_id = user_id
        window.load_details()
        wait_for(qapp, lambda: window.details is not None and window.details['id'] == user_id)
        qapp.processEvents()  # the previous user's buttons are gone
        return [button for button in window.findChildren(QPushButton) if button.isVisible()]

    try:
        wait_for(qapp, lambda: window.users)
        assert 'Delete permanently…' not in [button.text() for button in
                                             details_of(served.service.owner_id())]
        assert window.link.service_version == 2  # from the service's hello
        window.link.service_version = 1  # a service older than this window
        older = next(button for button in details_of(alice['id'])
                     if button.text() == 'Delete permanently…')
        assert not older.isEnabled() and 'sudo deploy/install.sh' in older.toolTip()
        window.link.service_version = 2
        window.details = None
        delete = next(button for button in details_of(alice['id'])
                      if button.text() == 'Delete permanently…')
        assert delete.isEnabled()
        delete.click()  # signed in a moment ago, and asked anyway
        assert passwords == [1] and 'cannot be undone' in asked[-1]
        assert served.service.user_row(alice['id'])['status'] == 'archived'
    finally:
        window.close()


LINK_RACE = '''
import time
from PySide6.QtWidgets import QApplication
app = QApplication([])
from cutieplus.cooperation.client import ServiceUnavailable
from gui.service_link import ServiceLink

class GoneService:
    token = None
    def copy(self):
        return GoneService()
    def call(self, op, **args):
        time.sleep(0.0005)
        raise ServiceUnavailable('gone')
    def close(self):
        pass

def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()

for _ in range(200):
    link = ServiceLink(GoneService(), {'token': 't', 'user': {'login_name': 'x', 'role': 'admin'}})
    link.poll_timer.start(1)
    spin(0.02)
    del link  # while its requests are still coming back
    spin(0.02)
print('survived')
'''


def test_a_link_let_go_while_its_requests_run_is_destroyed_on_the_gui_thread(tmp_path):
    """Before: a request thread could drop the last reference, and Qt destroyed the link there
    with its poll timer running: a crash at the timer's next turn."""
    import os
    import subprocess
    import sys
    script = tmp_path / 'race.py'
    script.write_text(LINK_RACE)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH=root)
    done = subprocess.run([sys.executable, str(script)], cwd=root, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and 'survived' in done.stdout, done.stderr[-2000:]
    assert 'another thread' not in done.stderr


def test_quit_at_the_first_sign_in_ends_the_program(qapp, tmp_path, monkeypatch):
    """interactive_demo.py signs in before the event loop runs, where quit() alone is ignored."""
    import signal
    from PySide6.QtCore import QTimer
    from cutieplus.cooperation.config import ClientConfig
    from gui.cooperation_app import CooperationApp
    from gui.login_dialog import LoginDialog
    monkeypatch.setattr(LoginDialog, 'exec', lambda self: 0)  # Quit
    monkeypatch.setattr(signal, 'signal', lambda *args: None)
    program = CooperationApp(compose_gui_config(),
                             ClientConfig('cooperation', str(tmp_path / 'app.sock')))
    stayed = QTimer()
    stayed.setSingleShot(True)
    stayed.timeout.connect(lambda: qapp.exit(9))
    try:
        program.start()
        stayed.start(3000)
        assert qapp.exec() == 0  # 9: the program stayed, with no window
    finally:
        stayed.stop()
        program.signal_timer.stop()
        qapp.setQuitOnLastWindowClosed(True)


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
