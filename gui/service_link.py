"""
A window's link to the cooperation service (cutieplus/cooperation): the session its user signed in
with, requests sent off the GUI thread (request), and a poll of whoami that notices when the
session ends, because the user's role changed, they were suspended, or the session expired:
signed_out then asks the window to have its user sign in again. Nothing here decides a permission;
the service checks every request.
"""
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from cutieplus.cooperation.client import ClientError, ServiceClient, ServiceUnavailable
from cutieplus.cooperation.errors import SIGN_IN_AGAIN

POLL_MS = 10_000
ROLE_NAMES = {'admin': 'Admin', 'supervisor': 'Supervisor', 'annotator': 'Annotator'}

Done = Callable[[object, Optional[ClientError]], None]


class ServiceLink(QObject):
    signed_out = Signal(str)  # why
    identity = Signal(dict)  # whoami's reply, at start and whenever it changes
    reachable = Signal(bool)  # the service answered (True) or not (False)
    _finished = Signal(object, object, object)  # done, result, error: back on the GUI thread

    def __init__(self, client: ServiceClient, session: dict):
        super().__init__()
        self.client = client
        self.client.token = session['token']
        self.background = client.copy()  # the threads' connection, not the GUI's
        self.user = session['user']
        self.whoami: Optional[dict] = None
        self.ended = False
        self.online = True
        self._finished.connect(self._deliver)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(POLL_MS)
        self.poll()

    @property
    def login_name(self) -> str:
        return self.user['login_name']

    @property
    def role(self) -> str:
        return self.user['role']

    @property
    def role_name(self) -> str:
        return ROLE_NAMES.get(self.role, self.role)

    def permission(self, name: str) -> bool:
        """What the window may offer, once whoami has answered (False before)."""
        return bool(self.whoami and self.whoami['permissions'].get(name))

    # requests
    def call(self, op: str, **args):
        """On the GUI thread, for quick requests; raises ClientError."""
        try:
            return self.client.call(op, **args)
        except ClientError as exc:
            self._noticed(exc)
            raise

    def request(self, op: str, done: Optional[Done] = None, **args) -> None:
        """In a thread; done(result, error) runs on the GUI thread."""

        def run() -> None:
            try:
                result, error = self.background.call(op, **args), None
            except ClientError as exc:
                result, error = None, exc
            self._finished.emit(done, result, error)

        threading.Thread(target=run, name=f'service-{op}', daemon=True).start()

    def _deliver(self, done, result, error) -> None:
        if error is not None:
            self._noticed(error)
        elif not self.online:
            self.online = True
            self.reachable.emit(True)
        if done is not None and not self.ended:
            done(result, error)

    def _noticed(self, error: ClientError) -> None:
        if error.code in SIGN_IN_AGAIN and not self.ended:
            self.ended = True
            self.poll_timer.stop()
            self.signed_out.emit(str(error))
        elif isinstance(error, ServiceUnavailable) and self.online:
            self.online = False
            self.reachable.emit(False)

    def poll(self) -> None:
        self.request('whoami', self._on_whoami)

    def _on_whoami(self, result, error) -> None:
        if error is None and result != self.whoami:
            self.whoami = result
            self.user = result['user']
            self.identity.emit(result)

    def logout(self) -> None:
        self.poll_timer.stop()
        self.ended = True
        try:
            self.client.call('logout')
        except ClientError:
            pass  # the session ends on its own
        self.client.token = self.background.token = None

    def close(self) -> None:
        self.poll_timer.stop()
        self.ended = True
        self.client.close()
        self.background.close()
