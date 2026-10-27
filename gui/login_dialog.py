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
from PySide6.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout)

from cutieplus.cooperation.client import ClientError, ServiceClient, ServiceUnavailable
from gui.gui_utils import is_dark
from gui.style import DANGER, GAP, Card, set_variant

UNREACHABLE = ('The CutiePlus service is not reachable ({reason}). Ask the owner to check it '
               '(systemctl status cutieplus-service), then Try again; or work offline, without '
               'clicks or propagation.')


class LoginDialog(QDialog):

    def __init__(self, client: ServiceClient, reason: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.client = client
        self.session: Optional[dict] = None  # login's reply once signed in
        self.offline = False
        self.setWindowTitle('Sign in to CutiePlus')
        self.setMinimumWidth(460)

        self.message = QLabel(reason or 'Sign in with your CutiePlus login name.')
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
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

        form = QFormLayout()
        form.addRow('Login name', self.login_edit)
        form.addRow('Password', self.password_edit)
        self.setup_form = QFormLayout()
        self.setup_form.addRow('Setup token', self.token_edit)
        self.setup_form.addRow('New password', self.new_password_edit)
        self.setup_form.addRow('Repeat it', self.repeat_edit)
        self.setup_card = Card(self.setup_form, title='First time, or after a reset')
        self.setup_card.hide()

        self.setup_toggle = QPushButton('First time? Use a setup token')
        self.setup_toggle.setFlat(True)
        self.setup_toggle.clicked.connect(self.toggle_setup)
        self.sign_in_button = set_variant(QPushButton('Sign in'), 'primary')
        self.sign_in_button.setDefault(True)
        self.sign_in_button.clicked.connect(self.sign_in)
        self.retry_button = QPushButton('Try again')
        self.retry_button.clicked.connect(self.check_service)
        self.offline_button = QPushButton('Work offline (CPU only)')
        self.offline_button.clicked.connect(self.work_offline)
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self.setup_toggle)
        buttons.addStretch(1)
        for button in (self.offline_button, self.retry_button, quit_button,
                       self.sign_in_button):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.setSpacing(GAP)
        layout.addWidget(self.message)
        layout.addWidget(Card(form))
        layout.addWidget(self.setup_card)
        layout.addLayout(buttons)
        self.check_service()

    def show_message(self, text: str, error: bool = False) -> None:
        self.message.setText(text)
        self.message.setStyleSheet(f'color: {DANGER[is_dark(self)]};' if error else '')

    def check_service(self) -> None:
        try:
            hello = self.client.call('hello')
        except ServiceUnavailable as exc:
            self.set_reachable(False, str(exc))
            return
        self.set_reachable(True)
        if not hello['owner_configured']:
            self.show_message('CutiePlus has no owner yet: the owner runs sudo cutieplus-admin '
                              'bootstrap-owner on this computer first.', error=True)

    def set_reachable(self, reachable: bool, reason: str = '') -> None:
        for widget in (self.login_edit, self.password_edit, self.sign_in_button,
                       self.setup_toggle):
            widget.setEnabled(reachable)
        self.retry_button.setVisible(not reachable)
        self.offline_button.setVisible(not reachable)
        if not reachable:
            self.show_message(UNREACHABLE.format(reason=reason), error=True)

    def toggle_setup(self) -> None:
        setting_up = self.setup_card.isHidden()
        self.setup_card.setVisible(setting_up)
        self.password_edit.setEnabled(not setting_up)
        self.setup_toggle.setText('Sign in with a password instead' if setting_up else
                                  'First time? Use a setup token')
        self.sign_in_button.setText('Set password and sign in' if setting_up else 'Sign in')

    def sign_in(self) -> None:
        login = self.login_edit.text().strip()
        try:
            if self.setup_card.isHidden():
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
