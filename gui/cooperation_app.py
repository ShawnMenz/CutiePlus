"""
The program on a host that runs the cooperation service (/etc/cutieplus/client.toml): sign in
(gui/login_dialog.py), then the task browser as that user; when they sign out, or their session
ends (a role change, a suspension), the browser closes, its open task saving first, and the
sign-in comes back. Closing the browser otherwise quits.
"""
from typing import Optional

from omegaconf import DictConfig
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from cutieplus.cooperation.client import ServiceClient
from cutieplus.cooperation.config import ClientConfig
from gui.login_dialog import LoginDialog
from gui.service_link import ServiceLink


class CooperationApp(QObject):

    def __init__(self, cfg: DictConfig, client: ClientConfig):
        super().__init__()
        self.cfg = cfg
        self.client_config = client
        self.browser = None
        self.link: Optional[ServiceLink] = None
        QApplication.instance().setQuitOnLastWindowClosed(False)

    def start(self, reason: Optional[str] = None) -> None:
        from gui.task_browser import TaskBrowser
        client = ServiceClient(self.client_config.app_socket)
        dialog = LoginDialog(client, reason)
        if not dialog.exec():
            QApplication.instance().quit()
            return
        if dialog.offline:
            self.link = None
            self.browser = TaskBrowser(self.cfg, self.cfg['tasks'], self.cfg['task_split'],
                                       offline=True)
        else:
            self.link = ServiceLink(client, dialog.session)
            self.browser = TaskBrowser(self.cfg, self.cfg['tasks'], self.cfg['task_split'],
                                       link=self.link)
        self.browser.sign_in_again.connect(self.on_sign_in_again)
        self.browser.closed.connect(self.on_closed)

    def on_sign_in_again(self, reason: str) -> None:
        if self.link is not None:
            self.link.close()
            self.link = None
        self.browser = None
        QTimer.singleShot(0, lambda: self.start(reason))

    def on_closed(self) -> None:
        if self.browser is not None and not self.browser.signing_out:
            if self.link is not None:
                self.link.logout()
            QApplication.instance().quit()
