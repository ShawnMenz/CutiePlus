"""
Signing in to CutiePlus on a host that runs the cooperation service (every window, at every start:
the session lives in this program only). A user's first sign-in, and the one after a reset, uses
the one-time setup token their supervisor or admin handed them, and sets their password.

When the service cannot be reached, the window says so and offers to work offline: the tasks can
be browsed, edited and saved, but nothing that needs the GPU runs, and the edits are recorded under
the account's name marked (offline).
"""
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QLabel, QLayout, QLineEdit,
                               QPushButton, QVBoxLayout)

from cutieplus.cooperation.client import ClientError, ServiceClient, ServiceUnavailable
from gui.gui_utils import is_dark
from gui.style import DANGER, GAP, PRIMARY, SECONDARY, Card, set_variant

MESSAGE_WIDTH = 440  # px: the text above the form wraps at about this width
UNREACHABLE = ('The CutiePlus service is not reachable ({reason}). Ask the owner to check it '
               '(systemctl status cutieplus-service), then Try again; or work offline, without '
               'clicks or propagation.')


class LoginDialog(QDialog):

    def __init__(self, client: ServiceClient, reason: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.client = client
        self.session: Optional[dict] = None  # login's reply once signed in
        self.hello: Optional[dict] = None  # the service's hello reply, once it answered
        self.offline = False
        self.setWindowTitle('Sign in to CutiePlus')
        self.setMinimumWidth(460)

        self.reason = reason  # why the sign-in is asked again, shown until the user acts
        self.reachable = False
        self.setting_up = False  # the setup token's rows, else the password's

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setMinimumWidth(MESSAGE_WIDTH)
        self.login_edit = QLineEdit()
        self.login_edit.setPlaceholderText('login name')
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText('password')
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText('XXXXX-XXXXX-XXXXX-XXXXX')
        self.new_password_edit = QLineEdit()
        self.new_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.new_password_edit.setPlaceholderText('at least 10 characters')
        self.repeat_edit = QLineEdit()
        self.repeat_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.repeat_edit.setPlaceholderText('the same again')

        # one form: the password's row, or the setup token's (set_mode)
        self.form = QFormLayout()
        self.form.addRow('Login name', self.login_edit)
        self.form.addRow('Password', self.password_edit)
        self.form.addRow('Setup token', self.token_edit)
        self.form.addRow('New password', self.new_password_edit)
        self.form.addRow('Repeat it', self.repeat_edit)

        self.setup_toggle = QPushButton()
        self.setup_toggle.setFlat(True)
        self.setup_toggle.setAutoDefault(False)
        self.setup_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        # a link: the theme's green, underlined
        self.setup_toggle.setStyleSheet(
            f'QPushButton {{ color: {(PRIMARY, SECONDARY)[is_dark(self)]}; border: none; '
            'background: transparent; text-decoration: underline; padding: 2px 4px; }')
        self.setup_toggle.clicked.connect(self.toggle_setup)
        self.sign_in_button = set_variant(QPushButton('Sign in'), 'primary')
        self.sign_in_button.setDefault(True)
        self.sign_in_button.clicked.connect(self.sign_in)
        self.retry_button = QPushButton('Try again')
        self.retry_button.setAutoDefault(False)
        self.retry_button.clicked.connect(self.check_service)
        self.offline_button = QPushButton('Work offline (CPU only)')
        self.offline_button.setAutoDefault(False)
        self.offline_button.clicked.connect(self.work_offline)
        quit_button = QPushButton('Quit')
        quit_button.setAutoDefault(False)
        quit_button.clicked.connect(self.reject)

        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.setup_toggle)
        toggle_row.addStretch(1)
        buttons = QHBoxLayout()
        for button in (self.offline_button, self.retry_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(quit_button)
        buttons.addWidget(self.sign_in_button)
        layout = QVBoxLayout(self)
        layout.setSpacing(GAP)
        # the window always fits what it shows: it shrinks again when the setup rows go
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        layout.addWidget(self.message)
        layout.addWidget(Card(self.form))
        layout.addLayout(toggle_row)
        layout.addLayout(buttons)
        self.set_mode(False)
        self.check_service()

    def show_message(self, text: str, error: bool = False) -> None:
        self.message.setText(text)
        self.message.setStyleSheet(f'color: {DANGER[is_dark(self)]};' if error else '')

    def check_service(self) -> None:
        try:
            hello = self.hello = self.client.call('hello')
        except ServiceUnavailable as exc:
            self.set_reachable(False, str(exc))
            return
        self.set_reachable(True)
        if not hello['owner_configured']:
            self.show_message('CutiePlus has no owner yet: the owner runs sudo cutieplus-admin '
                              'bootstrap-owner on this computer first.', error=True)
        else:
            self.show_hint()

    def set_reachable(self, reachable: bool, reason: str = '') -> None:
        self.reachable = reachable
        for widget in (self.login_edit, self.password_edit, self.token_edit,
                       self.new_password_edit, self.repeat_edit, self.sign_in_button,
                       self.setup_toggle):
            widget.setEnabled(reachable)
        self.retry_button.setVisible(not reachable)
        self.offline_button.setVisible(not reachable)
        if not reachable:
            self.show_message(UNREACHABLE.format(reason=reason), error=True)

    def show_hint(self) -> None:
        """What to type in this mode; at a remote desktop, which one it is (only its user signs
        in there: a sign-in refused there is often one at another's desktop)."""
        if self.reason:
            text = self.reason
        elif self.setting_up:
            text = ('First time, or after a reset: your login name, the setup token you were '
                    'given, and a new password of your own.')
        else:
            text = 'Sign in with your CutiePlus login name and password.'
        display = ((self.hello or {}).get('origin') or {}).get('display')
        if display is not None:
            text += f' This is remote desktop :{display}: only its own user signs in here.'
        self.show_message(text)

    def toggle_setup(self) -> None:
        self.set_mode(not self.setting_up)

    def set_mode(self, setting_up: bool) -> None:
        """The setup token's rows (first time, or after a reset), else the password's."""
        self.setting_up = setting_up
        self.form.setRowVisible(self.password_edit, not setting_up)
        for edit in (self.token_edit, self.new_password_edit, self.repeat_edit):
            self.form.setRowVisible(edit, setting_up)
        self.setup_toggle.setText('Have a password already? Sign in with it' if setting_up else
                                  'First time, or after a reset? Use your setup token')
        self.sign_in_button.setText('Set password and sign in' if setting_up else 'Sign in')
        if self.reachable:
            self.reason = None
            self.show_hint()
        if not self.login_edit.text().strip():
            self.login_edit.setFocus()
        else:
            (self.token_edit if setting_up else self.password_edit).setFocus()

    def sign_in(self) -> None:
        login = self.login_edit.text().strip()
        try:
            if not self.setting_up:
                self.session = self.client.call('login', login_name=login,
                                                password=self.password_edit.text())
            else:
                if self.new_password_edit.text() != self.repeat_edit.text():
                    self.show_message('The two new passwords differ.', error=True)
                    return
                self.session = self.client.call('setup_credential', login_name=login,
                                                setup_token=self.token_edit.text(),
                                                new_password=self.new_password_edit.text())
        except ServiceUnavailable as exc:
            self.set_reachable(False, str(exc))
            return
        except ClientError as exc:
            self.show_message(str(exc), error=True)
            self.password_edit.clear()
            return
        finally:
            for edit in (self.new_password_edit, self.repeat_edit):
                edit.clear()
        self.accept()

    def work_offline(self) -> None:
        self.offline = True
        self.accept()
