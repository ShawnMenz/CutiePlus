"""
The Manage Annotators window (the task browser's button, for supervisors and admins). It shows what
the cooperation service lets its user manage and sends their changes to it; the service checks
every request again (cutieplus/cooperation/authz.py), so the buttons here only spare refusals.

    Users     the list (search, removed users on request): role, status, desktop, and SSH, VNC
              and app activity apart; the selected user's profile, actions, desktop, SSH keys,
              connections, operations in progress and recent activity
    Settings  (admins) the settings an admin may change, within their bounds
    Audit     (admins) every audit record

A supervisor manages annotators: add, edit, suspend, reactivate, remove, reset their credential,
their keys, desktop and sessions; there is no role selector. An admin manages supervisors too, and
makes or revokes them. Changes to the system (the SSH policy, connections, desktops) are applied by
the service afterwards: an operation shows Applying until it is done, or Failed with the reason
while the service keeps trying.
"""
import html
from typing import Callable, List, Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFormLayout,
                               QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from cutieplus.cooperation.client import ClientError
from cutieplus.cooperation.errors import REAUTH_REQUIRED
from gui.gui_utils import fit_window
from gui.service_link import ROLE_NAMES, ServiceLink
from gui.style import GAP, Card, set_pill, set_variant

REFRESH_MS = 5000
PREFERRED_SIZE = QSize(1240, 760)
DETAILS_WIDTH = 460
USER_COLUMNS = ('Name', 'Login', 'Role', 'Status', 'Desktop', 'SSH', 'VNC', 'App', 'Operations')
STATUS_PILLS = {'active': 'accent', 'pending': 'null', 'suspended': 'null', 'archived': 'null'}


def cell(text: str, tip: str = '') -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        item.setToolTip(tip)
    return item


def short_time(text: Optional[str]) -> str:
    return text[:16].replace('T', ' ') if text else '—'


def table(headers) -> QTableWidget:
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(list(headers))
    widget.verticalHeader().hide()
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    widget.horizontalHeader().setStretchLastSection(True)
    widget.setWordWrap(False)
    return widget


class SecretCard(QDialog):
    """What a new or reset user needs, shown once: their setup token, desktop, VNC password and the
    SSH command. Nothing here is kept: copy it now."""

    def __init__(self, title: str, login: str, reply: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        rows = []
        if reply.get('setup_token'):
            rows.append(('Setup token', reply['setup_token'],
                         f'valid until {short_time(reply.get("setup_token_expires_at"))} UTC; '
                         f'{login} signs in with "First time?" and sets their password'))
        desktop = reply.get('desktop')
        if desktop:
            rows.append(('SSH tunnel', desktop['ssh_command'],
                         'run on their computer, with the key you registered for them'))
            rows.append(('VNC viewer', desktop['vnc_address'], f'desktop :{desktop["display"]}'))
            if desktop.get('vnc_password'):
                rows.append(('VNC password', desktop['vnc_password'], 'asked by the VNC viewer'))
        grid = QGridLayout()
        for row, (name, value, note) in enumerate(rows):
            field = QLineEdit(value)
            field.setReadOnly(True)
            field.setMinimumWidth(420)
            copy = QPushButton('Copy')
            copy.clicked.connect(lambda _=False, text=value: QGuiApplication.clipboard().setText(
                text))
            grid.addWidget(QLabel(name), 2 * row, 0)
            grid.addWidget(field, 2 * row, 1)
            grid.addWidget(copy, 2 * row, 2)
            hint = QLabel(note)
            hint.setProperty('muted', True)
            grid.addWidget(hint, 2 * row + 1, 1)
        warning = QLabel('Shown once: CutiePlus keeps no copy. Hand them on privately.')
        done = set_variant(QPushButton('Done'), 'primary')
        done.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(Card(grid))
        layout.addWidget(warning)
        layout.addWidget(done, alignment=Qt.AlignmentFlag.AlignRight)


class AddUserDialog(QDialog):

    def __init__(self, admin: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add user' if admin else 'Add Annotator')
        self.login_edit = QLineEdit()
        self.login_edit.setPlaceholderText('e.g. alice (letters, digits, . _ -)')
        self.name_edit = QLineEdit()
        self.contact_edit = QLineEdit()
        self.contact_edit.setPlaceholderText('optional: e-mail, phone')
        form = QFormLayout()
        form.addRow('Login name', self.login_edit)
        form.addRow('Display name', self.name_edit)
        form.addRow('Contact', self.contact_edit)
        self.role_combo: Optional[QComboBox] = None
        if admin:  # a supervisor creates annotators only: no role to pick
            self.role_combo = QComboBox()
            self.role_combo.addItem('Annotator', 'annotator')
            self.role_combo.addItem('Supervisor', 'supervisor')
            form.addRow('Role', self.role_combo)
        add = set_variant(QPushButton('Add'), 'primary')
        add.clicked.connect(self.accept)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(add)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('They get a remote desktop and a one-time setup token; add their '
                                'SSH public key next.'))
        layout.addLayout(form)
        layout.addLayout(buttons)

    def values(self) -> dict:
        values = {'login_name': self.login_edit.text().strip(),
                  'display_name': self.name_edit.text().strip()}
        if self.contact_edit.text().strip():
            values['contact'] = self.contact_edit.text().strip()
        if self.role_combo is not None:
            values['role'] = self.role_combo.currentData()
        return values


class AddKeyDialog(QDialog):

    def __init__(self, login: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Add an SSH key for {login}')
        self.key_edit = QPlainTextEdit()
        self.key_edit.setPlaceholderText('ssh-ed25519 AAAA… (the one line of their .pub file; '
                                         'never a private key)')
        self.key_edit.setFixedHeight(90)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText('which device, e.g. home laptop')
        self.expiry_edit = QLineEdit()
        self.expiry_edit.setPlaceholderText('optional: YYYY-MM-DD')
        form = QFormLayout()
        form.addRow('Public key', self.key_edit)
        form.addRow('Device', self.label_edit)
        form.addRow('Expires', self.expiry_edit)
        add = set_variant(QPushButton('Add key'), 'primary')
        add.clicked.connect(self.accept)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(add)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel('The key can open their own desktop only, and nothing else.'))
        layout.addLayout(buttons)

    def values(self) -> dict:
        values = {'public_key': self.key_edit.toPlainText().strip(),
                  'device_label': self.label_edit.text().strip()}
        if self.expiry_edit.text().strip():
            values['expires_at'] = self.expiry_edit.text().strip()
        return values


class ManageAnnotators(QDialog):

    def __init__(self, link: ServiceLink, parent=None):
        super().__init__(parent)
        self.link = link
        self.setWindowTitle('Manage Annotators')
        self.users: List[dict] = []
        self.selected_id: Optional[str] = None
        self.details: Optional[dict] = None
        self.loading = False

        self.search = QLineEdit()
        self.search.setPlaceholderText('Search by login, name or contact')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refresh)
        self.show_removed = QCheckBox('Show removed')
        self.show_removed.toggled.connect(self.refresh)
        self.add_button = set_variant(QPushButton('Add Annotator'), 'primary')
        self.add_button.clicked.connect(self.on_add_user)
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.show_removed)
        top.addWidget(self.add_button)
        top.addWidget(refresh_button)

        self.user_table = table(USER_COLUMNS)
        self.user_table.itemSelectionChanged.connect(self.on_select)
        self.details_box = QVBoxLayout()
        self.details_box.setSpacing(GAP)
        details_widget = QWidget()
        details_widget.setLayout(self.details_box)
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setWidget(details_widget)
        details_scroll.setMinimumWidth(DETAILS_WIDTH)
        splitter = QSplitter()
        splitter.addWidget(self.user_table)
        splitter.addWidget(details_scroll)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 620])
        users_page = QWidget()
        users_layout = QVBoxLayout(users_page)
        users_layout.setContentsMargins(0, 0, 0, 0)
        users_layout.addLayout(top)
        users_layout.addWidget(splitter, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(users_page, 'Users')
        self.settings_table: Optional[QTableWidget] = None
        self.audit_table: Optional[QTableWidget] = None
        self.status = QLabel('Loading your permissions…')
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)

        # nothing is offered until the service has said what this user may do
        self.set_enabled(False)
        link.identity.connect(self.on_identity)
        if link.whoami is not None:
            self.on_identity(link.whoami)
        else:
            link.poll()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_MS)
        fit_window(self, PREFERRED_SIZE)

    # permissions
    @property
    def is_admin(self) -> bool:
        return self.link.permission('manage_supervisors')

    def set_enabled(self, enabled: bool) -> None:
        for widget in (self.search, self.show_removed, self.add_button, self.user_table):
            widget.setEnabled(enabled)

    def on_identity(self, whoami: dict) -> None:
        if not self.link.permission('manage_annotators'):
            self.set_enabled(False)
            self.status.setText('You do not manage users (any more).')
            return
        self.set_enabled(True)
        self.add_button.setText('Add user' if self.is_admin else 'Add Annotator')
        if self.is_admin and self.settings_table is None:
            self.build_admin_tabs()
        self.status.setText('')
        self.refresh()

    # requests
    def confirm(self, title: str, text: str) -> bool:
        """Overridden by the tests."""
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self)
        yes = box.addButton(title, QMessageBox.ButtonRole.AcceptRole)
        box.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is yes

    def ask_password(self) -> Optional[str]:
        """Overridden by the tests."""
        text, accepted = QInputDialog.getText(
            self, 'Your password', 'This change needs your password again:',
            QLineEdit.EchoMode.Password)
        return text if accepted else None

    def show_secret(self, title: str, login: str, reply: dict) -> None:
        """Overridden by the tests."""
        SecretCard(title, login, reply, self).exec()

    def run(self, what: str, op: str, **args) -> Optional[dict]:
        """A management request; asks for the password again when the service wants it fresh."""
        for attempt in range(2):
            try:
                result = self.link.call(op, **args)
                self.status.setText(f'{what}: sent.')
                self.refresh()
                return result
            except ClientError as exc:
                if exc.code == REAUTH_REQUIRED and attempt == 0:
                    password = self.ask_password()
                    if password is None:
                        break
                    try:
                        self.link.call('reauthenticate', password=password)
                        continue
                    except ClientError as again:
                        exc = again
                self.status.setText(f'{what}: {exc}')
                return None
        self.status.setText(f'{what}: cancelled.')
        return None

    # the list
    def refresh(self) -> None:
        if not self.link.permission('manage_annotators') or self.loading:
            return
        self.loading = True
        self.link.request('list_users', self.on_users, query=self.search.text().strip(),
                          include_archived=self.show_removed.isChecked())

    def on_users(self, users, error) -> None:
        self.loading = False
        if error is not None:
            self.status.setText(f'The list could not be loaded: {error}')
            return
        self.users = users
        self.user_table.blockSignals(True)
        self.user_table.setRowCount(len(users))
        for row, user in enumerate(users):
            role = ROLE_NAMES.get(user['role'], user['role']) + (' · owner' if user['is_owner']
                                                                 else '')
            desktop = f':{user["desktop_display"]}' if user['desktop_display'] else '—'
            vnc = user['desktop_state'] or '—'
            if user['desktop_desired'] and user['desktop_state'] and \
                    (user['desktop_desired'] == 'running') != (user['desktop_state'] == 'running'):
                vnc += f' → {user["desktop_desired"]}'
            operations = 'Failed' if user['operations_failed'] else (
                'Applying…' if user['operations_running'] else '')
            values = (user['display_name'], user['login_name'], role, user['status'], desktop,
                      f'connected ({user["ssh_connections"]})' if user['ssh_connections'] else
                      '—', vnc, 'signed in' if user['app_sessions'] else '—', operations)
            for column, value in enumerate(values):
                self.user_table.setItem(row, column, cell(
                    value, user['desktop_error'] or '' if column == 6 else ''))
            if user['id'] == self.selected_id:
                self.user_table.selectRow(row)
        self.user_table.resizeColumnsToContents()
        self.user_table.blockSignals(False)
        if self.selected_id is not None:
            self.load_details()

    def on_select(self) -> None:
        rows = self.user_table.selectionModel().selectedRows()
        self.selected_id = self.users[rows[0].row()]['id'] if rows else None
        self.load_details()

    def load_details(self) -> None:
        if self.selected_id is None:
            self.show_details(None)
            return
        self.link.request('get_user', self.on_details, user_id=self.selected_id)

    def on_details(self, details, error) -> None:
        if error is not None:
            self.status.setText(f'The user could not be loaded: {error}')
            return
        if details['id'] == self.selected_id:
            self.show_details(details)

    # the selected user
    def clear_details(self) -> None:
        while self.details_box.count():
            item = self.details_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def button(self, text: str, action: Callable[[], None], variant: Optional[str] = None,
               tip: str = '') -> QPushButton:
        widget = set_variant(QPushButton(text), variant)
        widget.clicked.connect(action)
        if tip:
            widget.setToolTip(tip)
        return widget

    def show_details(self, user: Optional[dict]) -> None:
        self.details = user
        self.clear_details()
        if user is None:
            self.details_box.addWidget(QLabel('Select a user.'))
            self.details_box.addStretch(1)
            return
        me = user['id'] == self.link.user['id']
        managed = not me or self.is_admin
        self.details_box.addWidget(self.profile_card(user))
        if not me and not user['is_owner']:
            self.details_box.addWidget(self.actions_card(user))
        if managed:
            self.details_box.addWidget(self.desktop_card(user))
            self.details_box.addWidget(self.keys_card(user))
        self.details_box.addWidget(self.connections_card(user))
        self.details_box.addWidget(self.operations_card(user))
        if user.get('recent_activity') is not None:
            self.details_box.addWidget(self.activity_card(user))
        self.details_box.addStretch(1)

    def profile_card(self, user: dict) -> Card:
        name_edit = QLineEdit(user['display_name'])
        contact_edit = QLineEdit(user['contact'] or '')
        form = QFormLayout()
        pills = QHBoxLayout()
        role = set_pill(QLabel(ROLE_NAMES.get(user['role'], user['role'])), 'accent')
        status = set_pill(QLabel(user['status']), STATUS_PILLS.get(user['status'], 'null'))
        pills.addWidget(role)
        pills.addWidget(status)
        if user['is_owner']:
            pills.addWidget(set_pill(QLabel('owner'), 'key'))
        if user['credential_setup_required']:
            pills.addWidget(set_pill(QLabel('password not set up'), 'null'))
        pills.addStretch(1)
        form.addRow(pills)
        form.addRow('Login', QLabel(user['login_name']))
        form.addRow('Display name', name_edit)
        form.addRow('Contact', contact_edit)
        form.addRow('Last sign-in', QLabel(short_time(user['last_login_at'])))
        form.addRow('Windows signed in', QLabel(str(len(user.get('sessions', [])))))
        editable = user['status'] != 'archived'
        name_edit.setEnabled(editable)
        contact_edit.setEnabled(editable)
        save = self.button('Save profile', lambda: self.run(
            'Profile', 'update_user', user_id=user['id'], row_version=user['row_version'],
            display_name=name_edit.text(), contact=contact_edit.text()))
        save.setEnabled(editable)
        sign_out = self.button('Sign out everywhere', lambda: self.confirm(
            'Sign out', f'{user["login_name"]} is signed out of every CutiePlus window and signs '
            'in again. Their desktop keeps running.') and self.run(
                'Sign out', 'revoke_sessions', user_id=user['id']))
        row = QHBoxLayout()
        row.addWidget(save)
        row.addWidget(sign_out)
        row.addStretch(1)
        form.addRow(row)
        return Card(form, title=user['display_name'])

    def actions_card(self, user: dict) -> Card:
        login = user['login_name']
        desktop = user.get('desktop')
        where = f' desktop :{desktop["display_number"]}' if desktop else ' desktop'
        row = QHBoxLayout()
        if user['status'] in ('pending', 'active'):
            row.addWidget(self.button('Suspend', lambda: self.confirm(
                'Suspend', f'Suspend {login}? They are signed out of every window, their SSH '
                f'access stops and open connections are cut, their{where} stops (its window '
                'saves and closes), and their queued GPU jobs are cancelled. Reactivate undoes '
                'it.') and self.run('Suspend', 'suspend_user', user_id=user['id']), 'attention'))
            row.addWidget(self.button('Reset credential', self.reset_credential(user)))
        elif user['status'] == 'suspended':
            row.addWidget(self.button('Reactivate', lambda: self.run(
                'Reactivate', 'reactivate_user', user_id=user['id']), 'tonal'))
        if user['status'] != 'archived':
            row.addWidget(self.button('Remove', lambda: self.confirm(
                'Remove', f'Remove {login}? Their access is revoked for good: sign-in, SSH keys '
                f'and their{where}. Their annotation files, jobs and audit records are kept.')
                and self.run('Remove', 'remove_user', user_id=user['id']), 'danger'))
            if self.is_admin and user['role'] == 'annotator':
                row.addWidget(self.button('Make Supervisor', lambda: self.confirm(
                    'Make Supervisor', f'{login} signs in again and can then manage '
                    'annotators.') and self.run('Make Supervisor', 'set_role', user_id=user['id'],
                                                role='supervisor')))
            elif self.is_admin and user['role'] == 'supervisor':
                row.addWidget(self.button('Revoke Supervisor', lambda: self.confirm(
                    'Revoke Supervisor', f'{login} signs in again as an annotator; their desktop '
                    'and work stay.') and self.run('Revoke Supervisor', 'set_role',
                                                   user_id=user['id'], role='annotator')))
        row.addStretch(1)
        return Card(row, title='Account')

    def reset_credential(self, user: dict) -> Callable[[], None]:

        def reset() -> None:
            if not self.confirm('Reset credential', f'{user["login_name"]}\'s password stops '
                                'working and they are signed out; they set a new one with the '
                                'setup token you get now.'):
                return
            reply = self.run('Reset credential', 'reset_credential', user_id=user['id'])
            if reply:
                self.show_secret('New setup token', user['login_name'], reply)

        return reset

    def desktop_card(self, user: dict) -> Card:
        desktop = user.get('desktop')
        layout = QVBoxLayout()
        if desktop is None:
            if user['status'] in ('pending', 'active'):
                layout.addWidget(self.button('Give a desktop', lambda: self.give_desktop(user)))
            else:
                layout.addWidget(QLabel('No desktop.'))
            return Card(layout, title='Desktop')
        state = desktop['observed_state']
        text = (f'Display :{desktop["display_number"]} (VNC port {desktop["vnc_port"]}): '
                f'<b>{state}</b>')
        if desktop['desired_state'] != ('running' if state == 'running' else desktop[
                'desired_state']):
            text += f' → {desktop["desired_state"]}'
        if not desktop['vnc_password_set']:
            text += ' · no VNC password yet'
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)
        if desktop['last_error']:
            error = QLabel(html.escape(desktop['last_error']))
            error.setWordWrap(True)
            layout.addWidget(error)
        tunnel = QLabel(f'SSH: <code>{html.escape(desktop["card"]["ssh_command"])}</code>, then '
                        f'VNC to {desktop["card"]["vnc_address"]}')
        tunnel.setTextFormat(Qt.TextFormat.RichText)
        tunnel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        tunnel.setWordWrap(True)
        layout.addWidget(tunnel)
        row = QHBoxLayout()
        display = desktop['display_number']
        if user['status'] in ('pending', 'active'):
            row.addWidget(self.button('Start', lambda: self.run(
                'Start desktop', 'start_desktop', user_id=user['id'])))
            row.addWidget(self.button('Restart', lambda: self.confirm(
                'Restart', f'Restart desktop :{display}? Its window saves its work and closes, '
                'then opens again.') and self.run('Restart desktop', 'restart_desktop',
                                                 user_id=user['id'])))
        row.addWidget(self.button('Stop', lambda: self.confirm(
            'Stop', f'Stop desktop :{display}? Its window saves its work and closes; '
            f'{user["login_name"]} cannot use it until it is started again.') and self.run(
                'Stop desktop', 'stop_desktop', user_id=user['id'])))
        row.addWidget(self.button('New VNC password', lambda: self.new_vnc_password(user)))
        row.addStretch(1)
        layout.addLayout(row)
        return Card(layout, title='Desktop')

    def give_desktop(self, user: dict) -> None:
        reply = self.run('Desktop', 'assign_desktop', user_id=user['id'])
        if reply:
            self.show_secret('Their desktop', user['login_name'], reply)

    def new_vnc_password(self, user: dict) -> None:
        if not self.confirm('New VNC password', 'The desktop restarts to use it: its window '
                            'saves its work and closes, then opens again.'):
            return
        reply = self.run('VNC password', 'reset_vnc_password', user_id=user['id'])
        if reply:
            self.show_secret('New VNC password', user['login_name'], reply)

    def keys_card(self, user: dict) -> Card:
        keys = user.get('keys', [])
        layout = QVBoxLayout()
        grid = table(('Device', 'Fingerprint', 'Status', 'Expires', 'Last used', ''))
        grid.setRowCount(len(keys))
        for row, key in enumerate(keys):
            for column, value in enumerate((key['device_label'], key['fingerprint_sha256'],
                                            key['status'], short_time(key['expires_at']),
                                            short_time(key['last_authenticated_at']))):
                grid.setItem(row, column, cell(value))
            if key['status'] != 'revoked':
                grid.setCellWidget(row, 5, self.button('Revoke', lambda _=False, key=key: (
                    self.confirm('Revoke key', f'Revoke {user["login_name"]}\'s key '
                                 f'"{key["device_label"]}"? It stops working once published and '
                                 'its open connections are cut; their other keys keep working.')
                    and self.run('Revoke key', 'revoke_key', key_id=key['id']))))
        grid.setFixedHeight(min(60 + 32 * len(keys), 220))
        layout.addWidget(grid)
        if user['status'] != 'archived':
            layout.addWidget(self.button('Add key…', lambda: self.add_key(user)),
                             alignment=Qt.AlignmentFlag.AlignLeft)
        return Card(layout, title='SSH keys')

    def add_key(self, user: dict) -> None:
        dialog = AddKeyDialog(user['login_name'], self)
        if dialog.exec():
            self.run('Add key', 'add_key', user_id=user['id'], **dialog.values())

    def connections_card(self, user: dict) -> Card:
        connections = [row for row in user.get('connections', []) if row['status'] != 'closed']
        layout = QVBoxLayout()
        if not connections:
            layout.addWidget(QLabel('No SSH connection.'))
        for connection in connections:
            row = QHBoxLayout()
            row.addWidget(QLabel(f'{connection["source_address"]}:{connection["source_port"]} '
                                 f'since {short_time(connection["connected_at"])} '
                                 f'({connection["status"]})'))
            row.addStretch(1)
            row.addWidget(self.button('Disconnect', lambda _=False, connection=connection: (
                self.confirm('Disconnect', 'End this SSH connection? Their desktop and work stay, '
                             'and they can connect again.')
                and self.run('Disconnect', 'disconnect', connection_id=connection['id']))))
            layout.addLayout(row)
        return Card(layout, title='SSH connections')

    def operations_card(self, user: dict) -> Card:
        operations = user.get('operations', [])[:8]
        layout = QVBoxLayout()
        if not operations:
            layout.addWidget(QLabel('Nothing in progress.'))
        for operation in operations:
            status = {'running': 'Applying', 'pending': 'Pending', 'failed': 'Failed',
                      'succeeded': 'Done', 'superseded': 'Superseded'}[operation['status']]
            text = f'{operation["operation_type"].replace("_", " ")}: <b>{status}</b>'
            if operation['current_step']:
                text += f' ({html.escape(operation["current_step"])})'
            if operation['error_summary']:
                text += f'<br>{html.escape(operation["error_summary"])}'
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            layout.addWidget(label)
        return Card(layout, title='Operations')

    def activity_card(self, user: dict) -> Card:
        layout = QVBoxLayout()
        for event in user['recent_activity'][:10]:
            layout.addWidget(QLabel(f'{short_time(event["created_at"])}  '
                                    f'{event.get("actor_login") or event["actor_source"]}  '
                                    f'{event["action"].replace("_", " ")}  {event["result"]}'))
        return Card(layout, title='Recent activity')

    def on_add_user(self) -> None:
        dialog = AddUserDialog(self.is_admin, self)
        if not dialog.exec():
            return
        values = dialog.values()
        reply = self.run('Add', 'create_user', **values)
        if reply:
            self.selected_id = reply['user']['id']
            self.show_secret(f'{values["login_name"]} is added', values['login_name'], reply)

    # admins: settings and the audit
    def build_admin_tabs(self) -> None:
        self.settings_table = table(('Setting', 'Value', 'What it does', ''))
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.addWidget(self.settings_table)
        self.fixed_label = QLabel()
        self.fixed_label.setWordWrap(True)
        settings_layout.addWidget(self.fixed_label)
        self.tabs.addTab(settings_page, 'Settings')
        self.audit_table = table(('Time (UTC)', 'Actor', 'Source', 'Action', 'Target', 'Result',
                                  'Error'))
        audit_page = QWidget()
        audit_layout = QVBoxLayout(audit_page)
        reload = QPushButton('Refresh')
        reload.clicked.connect(self.load_audit)
        audit_layout.addWidget(reload, alignment=Qt.AlignmentFlag.AlignRight)
        audit_layout.addWidget(self.audit_table, 1)
        self.tabs.addTab(audit_page, 'Audit')
        self.tabs.currentChanged.connect(self.on_tab)

    def on_tab(self, index: int) -> None:
        name = self.tabs.tabText(index)
        if name == 'Settings':
            self.link.request('get_settings', self.on_settings)
        elif name == 'Audit':
            self.load_audit()

    def on_settings(self, settings, error) -> None:
        if error is not None:
            self.status.setText(f'Settings: {error}')
            return
        editable = settings['editable']
        self.settings_table.setRowCount(len(editable))
        for row, (key, item) in enumerate(editable.items()):
            box = QSpinBox()
            box.setRange(item['low'], item['high'])
            box.setValue(item['value'])
            self.settings_table.setItem(row, 0, cell(key))
            self.settings_table.setCellWidget(row, 1, box)
            self.settings_table.setItem(row, 2, cell(item['description']))
            self.settings_table.setCellWidget(row, 3, self.button(
                'Save', lambda _=False, key=key, box=box: self.run(
                    f'Setting {key}', 'update_setting', key=key, value=box.value())))
        fixed = settings['fixed']
        self.fixed_label.setText(
            f'Fixed at installation: owner {fixed["owner"]}, workspace {fixed["workspace_root"]}, '
            f'desktops :{fixed["displays"][0]}-:{fixed["displays"][1]}, SSH port '
            f'{fixed["ssh_port"]}, {fixed["gpu_jobs_at_once"]} GPU job at a time.')

    def load_audit(self) -> None:
        self.link.request('list_audit', self.on_audit, limit=500)

    def on_audit(self, events, error) -> None:
        if error is not None:
            self.status.setText(f'Audit: {error}')
            return
        self.audit_table.setRowCount(len(events))
        for row, event in enumerate(events):
            values = (short_time(event['created_at']), event.get('actor_login') or '—',
                      event['actor_source'], event['action'],
                      f'{event["target_type"] or ""} {event["target_id"] or ""}'.strip(),
                      event['result'], event['error_code'] or '')
            for column, value in enumerate(values):
                self.audit_table.setItem(row, column, cell(value))

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
