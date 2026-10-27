"""
A Unix domain socket that answers requests, one thread per connection. The handler gets each
request with the kernel's word on the peer (procinfo.Peer): who connected is never taken from the
request itself.
"""
import logging
import os
import socket
import stat
import threading
from typing import Callable, List

from cutieplus.cooperation.procinfo import Peer, peer_of
from cutieplus.cooperation.protocol import ProtocolError, error, receive_message, send_message

log = logging.getLogger(__name__)

Handler = Callable[[dict, Peer], dict]


def bind_socket(socket_path: str, mode: int) -> socket.socket:
    """A listening socket at socket_path with the given permission bits; a socket file left by a
    process that died is replaced, any other file refused."""
    if os.path.lexists(socket_path):
        if not stat.S_ISSOCK(os.lstat(socket_path).st_mode):
            raise RuntimeError(f'{socket_path} exists and is not a socket')
        os.remove(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o177)  # private until chmod below
    try:
        listener.bind(socket_path)
    finally:
        os.umask(old_umask)
    os.chmod(socket_path, mode)
    listener.listen(64)
    return listener


class Server:

    def __init__(self, handler: Handler, socket_path: str, mode: int = 0o600, name: str = 'server'):
        self.handler = handler
        self.socket_path = socket_path
        self.mode = mode
        self.name = name
        self.listener: socket.socket = None
        self.connections: List[socket.socket] = []
        self.lock = threading.Lock()
        self.running = False

    def start(self) -> None:
        self.listener = bind_socket(self.socket_path, self.mode)
        self.running = True
        threading.Thread(target=self._accept_loop, name=f'{self.name}-accept',
                         daemon=True).start()
        log.info('%s listening on %s', self.name, self.socket_path)

    def _accept_loop(self) -> None:
        while self.running:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                break
            with self.lock:
                self.connections.append(connection)
            threading.Thread(target=self._serve, args=(connection, ), name=f'{self.name}-client',
                             daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        try:
            peer = peer_of(connection)
            while self.running:
                try:
                    request = receive_message(connection)
                except (ConnectionError, OSError):
                    break
                except ProtocolError as exc:
                    send_message(connection, error('bad_request', str(exc)))
                    break
                if not isinstance(request, dict) or not isinstance(request.get('op'), str):
                    response = error('bad_request', 'a request is {"op": ..., "args": {...}}')
                else:
                    try:
                        response = self.handler(request, peer)
                    except Exception:  # the connection survives an unexpected error
                        log.exception('%s: %s failed', self.name, request.get('op'))
                        response = error('internal', 'the service failed; see its log')
                try:
                    send_message(connection, response)
                except OSError:
                    break
        finally:
            with self.lock:
                if connection in self.connections:
                    self.connections.remove(connection)
            connection.close()

    def stop(self) -> None:
        self.running = False
        if self.listener is not None:
            try:
                self.listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.listener.close()
        with self.lock:
            for connection in self.connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if os.path.lexists(self.socket_path):
            os.remove(self.socket_path)
