"""
The coordinator's Unix domain socket: one thread per connected window (a window keeps two
connections, one for its requests and one for its heartbeats). The socket file is private to the
account (mode 0600); every collaborator uses that account, so it identifies nobody.
"""
import logging
import os
import socket
import stat
import threading
from typing import List

from cutieplus.protocol import ProtocolError, receive_message, send_message

log = logging.getLogger(__name__)


def remove_stale_socket(socket_path: str) -> None:
    """A socket file left by a coordinator that died (the instance lock is already ours)."""
    if os.path.exists(socket_path):
        if not stat.S_ISSOCK(os.stat(socket_path).st_mode):
            raise RuntimeError(f'{socket_path} exists and is not a socket')
        os.remove(socket_path)


class Server:

    def __init__(self, service, socket_path: str):
        self.service = service
        self.socket_path = socket_path
        self.listener: socket.socket = None
        self.threads: List[threading.Thread] = []
        self.connections: List[socket.socket] = []
        self.lock = threading.Lock()
        self.running = False

    def start(self) -> None:
        remove_stale_socket(self.socket_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            listener.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        listener.listen(64)
        self.listener = listener
        self.running = True
        thread = threading.Thread(target=self._accept_loop, name='accept', daemon=True)
        thread.start()
        self.threads.append(thread)
        log.info('listening on %s', self.socket_path)

    def _accept_loop(self) -> None:
        while self.running:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                break
            with self.lock:
                self.connections.append(connection)
            thread = threading.Thread(target=self._serve, args=(connection, ), name='client',
                                      daemon=True)
            thread.start()

    def _serve(self, connection: socket.socket) -> None:
        try:
            while self.running:
                try:
                    request = receive_message(connection)
                except (ConnectionError, OSError, ProtocolError, ValueError):
                    break
                response = self.service.handle(request)
                if 'request_tag' in request:
                    response['request_tag'] = request['request_tag']
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
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
