"""
A labeling window's connection to the coordinator: requests over a Unix domain socket, a session
registered with the annotator and VNC display of the window, and heartbeats from a thread of their
own (a window busy with a long operation stays alive). After a restart of the coordinator the
client reconnects and resumes its session. Nothing here imports torch or Qt.
"""
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable, Optional

from cutieplus.protocol import receive_message, send_message

log = logging.getLogger(__name__)


class ClientError(Exception):
    """The coordinator refused a request; code says why (see cutieplus.coordinator.service)."""

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ServiceUnavailable(ClientError):
    def __init__(self, message: str):
        super().__init__('unavailable', message)


def new_request_id() -> str:
    """Repeating a request with the same id never applies it twice."""
    return uuid.uuid4().hex


class _Connection:

    def __init__(self, socket_path: str, timeout: float):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(socket_path)

    def request(self, message: dict) -> dict:
        send_message(self.sock, message)
        return receive_message(self.sock)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class ServiceClient:

    def __init__(self, socket_path: str, annotator_name: str, vnc_display: Optional[str] = None,
                 request_timeout: float = 300.0, reconnect_seconds: float = 60.0):
        self.socket_path = socket_path
        self.annotator_name = annotator_name
        self.vnc_display = vnc_display if vnc_display is not None else os.environ.get('DISPLAY')
        self.request_timeout = request_timeout
        self.reconnect_seconds = reconnect_seconds
        self.session_id: Optional[str] = None
        self.connection: Optional[_Connection] = None
        self.lock = threading.Lock()
        self.heartbeat_seconds = 5.0
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.stopping = threading.Event()
        self.last_heartbeat: Optional[float] = None
        self.on_session_lost: Optional[Callable[[str], None]] = None
        self.hello: dict = {}

    # connection ------------------------------------------------------------------------------
    def _connect(self) -> _Connection:
        deadline = time.monotonic() + self.reconnect_seconds
        while True:
            try:
                return _Connection(self.socket_path, self.request_timeout)
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                if time.monotonic() >= deadline:
                    raise ServiceUnavailable(f'the CutiePlus service is not running at '
                                             f'{self.socket_path} ({exc})') from None
                time.sleep(0.5)

    def _register(self, connection: _Connection) -> None:
        reply = connection.request({'op': 'register_session', 'args': {
            'annotator_name': self.annotator_name, 'vnc_display': self.vnc_display,
            'process_id': os.getpid(), 'host': socket.gethostname(),
            'session_id': self.session_id}})
        if not reply['ok'] and self.session_id is not None:
            # the session expired while the window was away: a new one (its leases are gone)
            lost = self.session_id
            self.session_id = None
            if self.on_session_lost is not None:
                self.on_session_lost(lost)
            reply = connection.request({'op': 'register_session', 'args': {
                'annotator_name': self.annotator_name, 'vnc_display': self.vnc_display,
                'process_id': os.getpid(), 'host': socket.gethostname()}})
        if not reply['ok']:
            raise ClientError(reply['error']['code'], reply['error']['message'])
        result = reply['result']
        self.session_id = result['session_id']
        self.heartbeat_seconds = float(result.get('heartbeat_seconds', 5))

    def connect(self) -> dict:
        with self.lock:
            self._ensure_connected()
            reply = self.connection.request({'op': 'hello', 'args': {}})
        self.hello = reply['result']
        return self.hello

    def _ensure_connected(self) -> None:
        if self.connection is None:
            connection = self._connect()
            try:
                self._register(connection)
            except BaseException:
                connection.close()
                raise
            self.connection = connection

    def call(self, op: str, **args) -> Any:
        """The result of a request. Mutating requests carry a request_id, so the one retry after
        a broken connection (the coordinator restarted) is safe."""
        with self.lock:
            for attempt in range(2):
                try:
                    self._ensure_connected()
                    reply = self.connection.request(
                        {'op': op, 'args': dict(args, session_id=self.session_id)})
                    break
                except (ConnectionError, OSError, socket.timeout) as exc:
                    if self.connection is not None:
                        self.connection.close()
                    self.connection = None
                    if attempt == 1 or isinstance(exc, socket.timeout):
                        raise ServiceUnavailable(f'lost the CutiePlus service during {op}: '
                                                 f'{exc}') from None
            if not reply['ok'] and reply['error']['code'] == 'session_expired':
                # resumed by registering again (a new session when the old one expired)
                self.connection.close()
                self.connection = None
                self._ensure_connected()
                reply = self.connection.request(
                    {'op': op, 'args': dict(args, session_id=self.session_id)})
        if not reply['ok']:
            error = dict(reply['error'])
            raise ClientError(error.pop('code'), error.pop('message'), error)
        return reply['result']

    # heartbeats ------------------------------------------------------------------------------
    def start_heartbeat(self) -> None:
        if self.heartbeat_thread is not None:
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name='heartbeat',
                                                 daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        connection: Optional[_Connection] = None
        while not self.stopping.wait(self.heartbeat_seconds):
            if self.session_id is None:
                continue
            try:
                if connection is None:
                    connection = _Connection(self.socket_path, 10.0)
                reply = connection.request({'op': 'heartbeat',
                                            'args': {'session_id': self.session_id}})
                if reply['ok']:
                    self.last_heartbeat = time.time()
            except (ConnectionError, OSError, socket.timeout):
                if connection is not None:
                    connection.close()
                connection = None
        if connection is not None:
            connection.close()

    def close(self) -> None:
        self.stopping.set()
        with self.lock:
            if self.connection is not None:
                try:
                    self.connection.request({'op': 'close_session',
                                             'args': {'session_id': self.session_id}})
                except (ConnectionError, OSError, socket.timeout):
                    pass
                self.connection.close()
                self.connection = None
