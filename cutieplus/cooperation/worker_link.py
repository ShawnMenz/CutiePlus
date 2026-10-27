"""
The service's end of the GPU worker (cutieplus/gpu_worker). The worker connects to the worker
socket; the service accepts it only from the cutieplus-gpu-worker system service, and only while
that process holds the GPU lock (both from the kernel: its control group and /proc/locks). One
worker at a time: another is refused while the first one's process still runs. Each accepted
connection is a new worker generation (gpu_state.worker_generation); nothing a previous generation
started is continued, and its results are refused.

Commands go one at a time (call). A worker that breaks the connection, or does not answer in time,
is dropped: the service never sends work to two workers, and when the worker hangs it asks the
root helper to restart its unit, which stops the whole process group before a new one starts.
"""
import json
import logging
import socket
import threading
from typing import Callable, Optional

from cutieplus.cooperation import audit
from cutieplus.cooperation.procinfo import (ProcessIdentity, flock_holders, peer_of,
                                            process_identity, same_process_alive,
                                            system_unit_of)
from cutieplus.cooperation.protocol import (ProtocolError, error, ok, receive_message,
                                            send_message)
from cutieplus.cooperation.server import bind_socket

log = logging.getLogger(__name__)
GPU_ID = 'gpu0'


class WorkerLost(Exception):
    """No worker, or it went away during the command."""


class CommandFailed(Exception):
    """The worker refused or failed a command; kind: 'invalid' | 'oom' | 'failed'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class WorkerLink:

    def __init__(self, service, on_change: Callable[[], None] = lambda: None):
        self.service = service
        self.config = service.config
        self.on_change = on_change
        self.lock = threading.Lock()  # one command at a time
        self.sock: Optional[socket.socket] = None
        self.generation = 0
        self.instance_id: Optional[str] = None
        self.identity: Optional[ProcessIdentity] = None
        self.hello: dict = {}
        self.listener: Optional[socket.socket] = None
        self.running = False
        self.restart_worker: Optional[Callable[[], None]] = None  # the helper's, when installed
        with service.db.transaction() as cursor:
            cursor.execute("INSERT OR IGNORE INTO gpu_state (gpu_id, state, updated_at) "
                           "VALUES (?, 'offline', ?)", (GPU_ID, service.now()))
            # a worker of the previous run of the service is not ours any more
            cursor.execute("UPDATE gpu_state SET state = 'offline', active_job_id = NULL, "
                           'updated_at = ? WHERE gpu_id = ?', (service.now(), GPU_ID))

    @property
    def ready(self) -> bool:
        return self.sock is not None

    def start(self) -> None:
        # every local process may connect; only the worker's service is accepted (check_peer)
        self.listener = bind_socket(self.config.worker_socket, 0o666)
        self.running = True
        threading.Thread(target=self._accept_loop, name='worker-accept', daemon=True).start()

    def stop(self) -> None:
        self.running = False
        if self.listener is not None:
            try:
                self.listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.listener.close()
        self.drop('the service stopped', restart=False)

    def check_peer(self, pid: int) -> Optional[str]:
        """Why this process may not be the GPU worker, or None."""
        if not self.config.verify_peers:
            return None
        if system_unit_of(pid) != self.config.worker_unit:
            return f'not a process of {self.config.worker_unit}'
        if pid not in flock_holders(self.config.gpu_lock):
            return 'it does not hold the GPU lock'
        return None

    def _accept_loop(self) -> None:
        while self.running:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                break
            try:
                self._accept(connection)
            except (OSError, ConnectionError, ProtocolError) as exc:
                log.warning('a worker connection failed: %s', exc)
                connection.close()

    def _accept(self, connection: socket.socket) -> None:
        peer = peer_of(connection)
        connection.settimeout(30)
        hello = receive_message(connection)
        refusal = self.check_peer(peer.pid)
        if refusal is None and self.sock is not None and same_process_alive(self.identity):
            refusal = 'another worker is connected'
        if refusal is not None:
            log.warning('refused a GPU worker (pid %d): %s', peer.pid, refusal)
            send_message(connection, error('refused', refusal))
            connection.close()
            return
        if self.sock is not None:
            self.drop('replaced by a new worker', restart=False)
        now = self.service.now()
        identity = process_identity(peer.pid)
        with self.service.db.transaction() as cursor:
            cursor.execute('UPDATE gpu_state SET worker_generation = worker_generation + 1 '
                           'WHERE gpu_id = ?', (GPU_ID, ))
            generation = cursor.execute('SELECT worker_generation FROM gpu_state WHERE '
                                        'gpu_id = ?', (GPU_ID, )).fetchall()[0][
                                            'worker_generation']
            cursor.execute(
                "UPDATE gpu_state SET worker_instance_id = ?, worker_process_identity = ?, "
                "state = 'idle', active_job_id = NULL, heartbeat_at = ?, last_error = NULL, "
                'updated_at = ? WHERE gpu_id = ?',
                (hello.get('instance_id'), None if identity is None else
                 json.dumps(identity.as_dict(), sort_keys=True), now, now, GPU_ID))
            audit.record(cursor, now, 'gpu_worker_connected', audit.SUCCEEDED, None,
                         audit.SYSTEM, 'gpu', GPU_ID,
                         after={'generation': generation, 'pid': peer.pid,
                                'engine': (hello.get('hello') or {}).get('engine')})
        send_message(connection, ok({'generation': generation}))
        connection.settimeout(None)
        with self.lock:
            self.sock = connection
            self.generation = generation
            self.instance_id = hello.get('instance_id')
            self.identity = identity
            self.hello = hello.get('hello') or {}
        log.info('GPU worker generation %d connected (pid %d)', generation, peer.pid)
        self.on_change()

    def call(self, op: str, timeout: float = 60.0, **args):
        with self.lock:
            sock = self.sock
            if sock is None:
                raise WorkerLost('no GPU worker is connected')
            try:
                sock.settimeout(timeout)
                send_message(sock, {'op': op, 'args': args})
                reply = receive_message(sock)
            except socket.timeout:
                self._drop_locked(f'it did not answer {op} within {timeout:.0f} s', restart=True)
                raise WorkerLost(f'the GPU worker did not answer {op}') from None
            except (OSError, ConnectionError, ProtocolError) as exc:
                self._drop_locked(f'the connection broke during {op}: {exc}', restart=False)
                raise WorkerLost(f'the GPU worker went away during {op}') from None
        if not reply.get('ok'):
            details = reply.get('error') or {}
            raise CommandFailed(details.get('code', 'failed'), details.get('message', ''))
        return reply.get('result')

    def drop(self, reason: str, restart: bool) -> None:
        with self.lock:
            self._drop_locked(reason, restart)

    def _drop_locked(self, reason: str, restart: bool) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None
        log.warning('GPU worker generation %d dropped: %s', self.generation, reason)
        state = 'recovering' if restart else 'offline'
        with self.service.db.transaction() as cursor:
            cursor.execute('UPDATE gpu_state SET state = ?, active_job_id = NULL, '
                           'last_error = ?, updated_at = ? WHERE gpu_id = ?',
                           (state, reason, self.service.now(), GPU_ID))
        if restart and self.restart_worker is not None:
            # a hung worker: its unit is restarted, which ends its whole process group first
            threading.Thread(target=self.restart_worker, daemon=True).start()
        threading.Thread(target=self.on_change, daemon=True).start()
