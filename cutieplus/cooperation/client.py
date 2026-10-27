"""
A window's connection to the cooperation service. The window signs in (login) and keeps the
session token in this object only: never in a file, never shared with another window. Every later
request carries it. Nothing here imports torch or Qt.

A request whose connection breaks after it was sent is not repeated here: the service may have
applied it. Requests that may be repeated carry their own idempotency key (job submissions).
"""
import logging
import socket
import threading
from typing import Any, Optional

from cutieplus.cooperation.protocol import ProtocolError, receive_message, send_message

log = logging.getLogger(__name__)


class ClientError(Exception):
    """The service refused a request; code says why (cutieplus.cooperation.errors)."""

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ServiceUnavailable(ClientError):

    def __init__(self, message: str):
        super().__init__('unavailable', message)


class ServiceClient:

    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.token: Optional[str] = None
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()

    def copy(self) -> 'ServiceClient':
        """Another connection with the same session, e.g. for a thread that polls."""
        other = ServiceClient(self.socket_path, self.timeout)
        other.token = self.token
        return other

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            sock.close()
            raise ServiceUnavailable(f'the CutiePlus service is not reachable at '
                                     f'{self.socket_path} ({exc.strerror or exc})') from None
        return sock

    def call(self, op: str, timeout: Optional[float] = None, **args) -> Any:
        """The result of the request; raises ClientError, or ServiceUnavailable when the service
        cannot be reached or the connection broke."""
        message = {'op': op, 'args': args}
        if self.token is not None:
            message['token'] = self.token
        with self.lock:
            for attempt in range(2):
                sent = False
                try:
                    if self.sock is None:
                        self.sock = self._connect()
                    self.sock.settimeout(timeout or self.timeout)
                    send_message(self.sock, message)
                    sent = True
                    reply = receive_message(self.sock)
                    break
                except ServiceUnavailable:
                    raise
                except (OSError, ConnectionError, ProtocolError) as exc:
                    self._drop()
                    # a connection the service closed while idle: sent nothing, so try once more
                    if sent or attempt == 1:
                        raise ServiceUnavailable(f'lost the CutiePlus service during {op} '
                                                 f'({exc})') from None
        if not reply.get('ok'):
            details = dict(reply.get('error') or {})
            raise ClientError(details.pop('code', 'error'), details.pop('message', ''), details)
        return reply.get('result')

    def _drop(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def close(self) -> None:
        with self.lock:
            self._drop()
