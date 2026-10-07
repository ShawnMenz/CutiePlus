"""
The coordinator's handle on the GPU worker process: spawn, the ready handshake, one command at a
time with a timeout, and the detection of a dead or hung worker. Every start is a new generation;
results of an older generation are never used.
"""
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from typing import Any, Optional

log = logging.getLogger(__name__)


class WorkerError(Exception):
    """A command the worker refused or failed (kind: 'invalid' | 'rejected' | 'oom' | 'pressure' |
    'internal')."""

    def __init__(self, kind: str, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or {}


class WorkerDied(Exception):
    """The worker process exited, broke its pipe, or did not answer in time (it is then killed)."""


class WorkerProxy:

    def __init__(self, cfg_container: dict, service: dict, log_path: Optional[str] = None):
        self.cfg_container = cfg_container
        self.service = service
        self.log_path = log_path
        self.context = mp.get_context('spawn')
        self.process: Optional[mp.Process] = None
        self.conn = None
        self.generation = 0
        self.hello: dict = {}
        self.seq = 0
        self.lock = threading.Lock()
        self.started_at: Optional[float] = None
        self.start_seconds: Optional[float] = None

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self, timeout: float = 600.0) -> dict:
        """Spawns a worker and waits until it has loaded and profiled the models."""
        from cutieplus.worker.main import worker_main
        self.stop()
        parent, child = self.context.Pipe(duplex=True)
        start = time.perf_counter()
        process = self.context.Process(target=worker_main, name='cutieplus-gpu-worker',
                                       args=(child, self.cfg_container, self.service, self.log_path),
                                       daemon=True)
        # marks the process, so a restarted coordinator can recognize a worker left behind
        os.environ['CUTIEPLUS_WORKER'] = '1'
        try:
            process.start()
        finally:
            os.environ.pop('CUTIEPLUS_WORKER', None)
        child.close()
        self.process, self.conn = process, parent
        self.generation += 1
        if not parent.poll(timeout):
            self.kill()
            raise WorkerDied(f'the GPU worker did not start within {timeout:.0f} s')
        try:
            message = parent.recv()
        except (EOFError, OSError):
            self.kill()
            raise WorkerDied(f'the GPU worker exited while starting (exit code '
                             f'{process.exitcode})') from None
        if message.get('event') != 'ready':
            self.kill()
            raise WorkerDied(f'the GPU worker failed to start: {message.get("error")}')
        self.hello = message
        self.started_at = time.time()
        self.start_seconds = time.perf_counter() - start
        log.info('GPU worker %s ready (generation %d, %.1f s)', process.pid, self.generation,
                 self.start_seconds)
        return message

    def call(self, op: str, timeout: Optional[float] = None, **args) -> Any:
        """The command's result; raises WorkerError when the worker refuses it and WorkerDied
        when the worker is gone (a worker that does not answer in time is killed)."""
        with self.lock:
            if self.conn is None or not self.alive():
                raise WorkerDied('the GPU worker is not running')
            self.seq += 1
            seq = self.seq
            try:
                self.conn.send({'seq': seq, 'op': op, 'args': args})
                deadline = None if timeout is None else time.monotonic() + timeout
                while True:
                    wait = 1.0 if deadline is None else max(0.0, min(1.0, deadline - time.monotonic()))
                    if self.conn.poll(wait):
                        reply = self.conn.recv()
                        break
                    if not self.alive():
                        raise WorkerDied(f'the GPU worker exited (exit code {self.process.exitcode}) '
                                         f'during {op}')
                    if deadline is not None and time.monotonic() >= deadline:
                        self.kill()
                        raise WorkerDied(f'the GPU worker did not answer {op} within {timeout:.0f} s; '
                                         'it was stopped')
            except (EOFError, OSError, BrokenPipeError) as exc:
                self.kill()
                raise WorkerDied(f'the GPU worker connection broke during {op}: {exc}') from None
            if reply.get('seq') != seq:
                self.kill()
                raise WorkerDied(f'out-of-order reply from the GPU worker ({reply.get("seq")} != {seq})')
            if not reply['ok']:
                error = dict(reply['error'])
                raise WorkerError(error.pop('kind'), error.pop('message'), error)
            result = reply['result']
            if isinstance(result, dict):
                result.setdefault('_worker_seconds', reply.get('seconds'))
            return result

    def stop(self, timeout: float = 10.0) -> None:
        """Asks the worker to exit, then kills it if needed."""
        if self.process is None:
            return
        if self.alive() and self.conn is not None:
            try:
                with self.lock:
                    self.seq += 1
                    self.conn.send({'seq': self.seq, 'op': 'shutdown'})
                    if self.conn.poll(timeout):
                        self.conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                pass
        self.process.join(timeout)
        self.kill()

    def kill(self) -> None:
        process = self.process
        if process is not None and process.is_alive():
            log.warning('killing the GPU worker %s', process.pid)
            process.kill()
            process.join(10)
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None
        self.process = None


def stop_stale_worker(pid: Optional[int]) -> bool:
    """Stops a worker left by a coordinator that died (its pid recorded in the state directory),
    if it is still running and really a CutiePlus worker. Returns whether one was stopped."""
    if not pid:
        return False
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            command = f.read().replace(b'\0', b' ')
    except OSError:
        return False
    # a spawned worker runs multiprocessing's spawn_main from this program's interpreter
    if b'multiprocessing' not in command:
        return False
    try:
        with open(f'/proc/{pid}/environ', 'rb') as f:
            if b'CUTIEPLUS_WORKER=1' not in f.read():
                return False
    except OSError:
        return False
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not os.path.exists(f'/proc/{pid}'):
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    return True
