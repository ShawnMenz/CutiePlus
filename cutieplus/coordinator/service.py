"""
The coordinator: one per task root (an exclusive lock in its state directory), CPU only. It owns
the sessions, the task leases and every write of official annotation data; it schedules the GPU
work and drives the one GPU worker, restarting it when it fails.
"""
import fcntl
import inspect
import logging
import os
import socket
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from os import path
from typing import Any, Callable, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from cutieplus.coordinator.interactions import InteractionError, InteractionManager
from cutieplus.coordinator.jobs import JobConfig, JobError, JobManager
from cutieplus.coordinator.metrics import Metrics
from cutieplus.coordinator.scheduler import QueueFull, Scheduler, SchedulerConfig
from cutieplus.coordinator.sessions import SessionError, SessionManager
from cutieplus.coordinator.tasks import TaskError, TaskService
from cutieplus.coordinator.worker_proxy import (WorkerDied, WorkerError, WorkerProxy,
                                                stop_stale_worker)
from cutieplus.engine.settings import model_config
from cutieplus.paths import default_socket_path, default_state_dir
from cutieplus.storage.db import Database, dumps
from cutieplus.storage.objects import ObjectStore
from gui.task_state import TaskLock, TaskLocked

log = logging.getLogger(__name__)

VERSION = 1
WORKER_KEYS = ('device', 'allow_cpu', 'allow_precision_change', 'profile', 'profile_frame_size',
               'gpu', 'ritm_snapshot_mib', 'max_interactions', 'engine', 'fake',
               'allow_test_commands', 'torch_threads')


class ServiceError(Exception):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details


class AlreadyRunning(Exception):
    pass


def service_section(cfg: DictConfig, overrides: Optional[dict] = None) -> dict:
    section = OmegaConf.to_container(cfg.get('service', {}), resolve=True) if 'service' in cfg else {}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(section.get(key), dict):
            section[key] = dict(section[key], **value)
        else:
            section[key] = value
    return section




class Service:

    def __init__(self, cfg: DictConfig, task_root: str, state_dir: Optional[str] = None,
                 overrides: Optional[dict] = None):
        self.cfg = cfg
        self.settings = service_section(cfg, overrides)
        self.task_root = path.abspath(task_root)
        self.state_dir = path.abspath(state_dir or self.settings.get('state_dir') or
                                      default_state_dir(task_root))
        os.makedirs(path.join(self.state_dir, 'logs'), exist_ok=True)
        self.socket_path = self.settings.get('socket') or default_socket_path(self.state_dir)
        self.instance_lock = self._acquire_instance_lock()
        self.db = Database(path.join(self.state_dir, 'service.db'))
        self.objects = ObjectStore(path.join(self.state_dir, 'objects'))
        self.tasks = TaskService(self.db, self.objects, self.task_root,
                                 int(cfg.get('max_internal_size', 480)))
        self.scheduler = Scheduler(SchedulerConfig.from_config(self.settings.get('scheduler')))
        worker_settings = {key: self.settings[key] for key in WORKER_KEYS if key in self.settings}
        self.worker = WorkerProxy(OmegaConf.to_container(cfg, resolve=True), worker_settings,
                                  path.join(self.state_dir, 'logs', 'worker.log'))
        gpu = self.settings.get('gpu') or {}
        self.jobs = JobManager(self.db, self.tasks, self.scheduler, self._worker_call,
                               lambda: self.worker_hello,
                               JobConfig(queue_wait_seconds=float(gpu.get('queue_wait_seconds', 600)),
                                         max_oom_retries=int(gpu.get('max_oom_retries', 1))))
        self.interactions = InteractionManager(self.tasks, self.run_on_gpu, self.worker)
        self.sessions = SessionManager(self.db, float(self.settings.get('session_timeout_seconds',
                                                                        30)),
                                       on_expire=self._session_ended)
        self.metrics = Metrics()
        self.request_timeout = float(self.settings.get('request_timeout_seconds', 120))
        self.running = False
        self.threads = []
        self.worker_hello: dict = {}
        self.worker_problem: Optional[str] = None
        self.recovery_report: dict = {}

    # single instance -------------------------------------------------------------------------
    def _acquire_instance_lock(self):
        stream = open(path.join(self.state_dir, 'coordinator.lock'), 'a+', encoding='utf-8')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            holder = stream.read().strip()
            stream.close()
            raise AlreadyRunning(f'a coordinator already runs for {self.task_root} ({holder})') from None
        stream.seek(0)
        stream.truncate()
        stream.write(f'pid {os.getpid()} on {socket.gethostname()}\n')
        stream.flush()
        return stream

    # lifecycle -------------------------------------------------------------------------------
    def start(self) -> None:
        self.recovery_report = self._recover_on_start()
        self._start_worker()
        self.running = True
        for target, name in ((self._dispatch_loop, 'dispatcher'),
                             (self._maintenance_loop, 'maintenance')):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def _start_worker(self) -> None:
        timeout = float(self.settings.get('worker_start_timeout_seconds', 600))
        hello = self.worker.start(timeout)
        hello['model_configuration'] = model_config(self.cfg)
        self.worker_hello = hello
        self.worker_problem = None
        with self.db.transaction() as conn:
            conn.execute('INSERT INTO worker_starts (at, generation, pid, hello) VALUES (?, ?, ?, ?)',
                         (self.tasks.clock(), self.worker.generation, hello.get('pid'),
                          dumps({k: v for k, v in hello.items() if k != 'event'})))
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('worker_pid', ?)",
                         (str(hello.get('pid')), ))
        self._event('worker_started', {'generation': self.worker.generation,
                                       'pid': hello.get('pid'),
                                       'seconds': self.worker.start_seconds})

    def stop(self) -> None:
        if not self.running and not self.threads:
            self._close_storage()
            return
        self.running = False
        for job in self.jobs.active():
            self.jobs.cancel(job.job_id, reason='the service is stopping')
        self.scheduler.stop()
        for thread in self.threads:
            thread.join(30)
        self.threads = []
        self.jobs.shutdown()
        for context in list(self.tasks.leases.values()):
            self.tasks.close_task(context.task_id, context.session_id, context.epoch,
                                  status='interrupted')
        self.worker.stop()
        self._event('stopped', {})
        self._close_storage()

    def _close_storage(self) -> None:
        self.db.close()
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_UN)
            self.instance_lock.close()
        except (OSError, ValueError):
            pass

    def _event(self, kind: str, detail: dict) -> None:
        try:
            with self.db.transaction() as conn:
                conn.execute('INSERT INTO events (at, kind, detail) VALUES (?, ?, ?)',
                             (self.tasks.clock(), kind, dumps(detail)))
        except Exception:
            log.exception('recording event %s', kind)

    # recovery at start -----------------------------------------------------------------------
    def _recover_on_start(self) -> dict:
        report: Dict[str, Any] = {}
        row = self.db.query_one("SELECT value FROM meta WHERE key = 'worker_pid'")
        if row is not None and row['value'].isdigit():
            report['stale_worker_stopped'] = stop_stale_worker(int(row['value']))
        report['temporary_objects_removed'] = self.objects.remove_temporary_files()
        now = self.tasks.clock()
        with self.db.transaction() as conn:
            report['jobs_interrupted'] = conn.execute(
                "UPDATE jobs SET status = 'interrupted', stop_reason = 'the service restarted', "
                "updated_at = ? WHERE status NOT IN ('finished', 'cancelled', 'failed', "
                "'interrupted')", (now, )).rowcount
            report['leases_interrupted'] = conn.execute(
                "UPDATE leases SET status = 'interrupted', renewed_at = ? WHERE status = 'active'",
                (now, )).rowcount
            report['requests_pruned'] = conn.execute(
                "DELETE FROM requests WHERE created_at < datetime('now', '-7 days')").rowcount
        report['sessions_given_grace'] = self.sessions.grant_grace()
        # tasks whose last commit did not reach their files, when no other program holds them
        resynced, skipped = [], []
        for task in self.db.query('SELECT task_id, workspace FROM tasks WHERE view_revision < '
                                  'content_revision'):
            try:
                lock = TaskLock.acquire(task['workspace'], 'cutieplus-recovery',
                                        activity='recovering')
            except (TaskLocked, OSError):
                skipped.append(task['task_id'])
                continue
            try:
                info = self.tasks.info(task['workspace'])
                outcome = self.tasks.ensure_current(info)
                resynced.append((task['task_id'], outcome))
            finally:
                lock.release()
        report['views_resynced'] = resynced
        report['views_skipped'] = skipped
        report['orphan_objects_removed'] = self._remove_orphan_objects()
        self._event('recovered', report)
        log.info('start-up recovery: %s', report)
        return report

    def _remove_orphan_objects(self) -> int:
        """Objects published by a commit whose transaction never completed (the service stopped in
        between); nothing refers to them."""
        referenced = set()
        for row in self.db.query('SELECT mask_object FROM frames WHERE mask_object IS NOT NULL'):
            referenced.add(row['mask_object'])
        for row in self.db.query('SELECT before_object, after_object FROM edits'):
            referenced.update(sha for sha in (row['before_object'], row['after_object']) if sha)
        removed = 0
        for sha in list(self.objects.objects()):
            if sha not in referenced:
                self.objects.remove(sha)
                removed += 1
        return removed

    # the GPU -----------------------------------------------------------------------------------
    def _worker_call(self, op: str, **args):
        """Dispatcher thread only."""
        return self.worker.call(op, timeout=self.request_timeout, **args)

    def run_on_gpu(self, session_id: str, fn: Callable, label: str = '') -> Any:
        """Runs fn on the dispatcher thread when the scheduler picks it; waits for its result."""
        if not self.running:
            raise ServiceError('stopping', 'the service is stopping')
        if self.worker_problem is not None:
            raise ServiceError('worker', f'the GPU worker is not available: {self.worker_problem}')
        future: Future = Future()
        submitted = time.perf_counter()

        def run():
            if not future.set_running_or_notify_cancel():
                return
            started = time.perf_counter()
            try:
                result = fn()
            except BaseException as exc:
                future.set_exception(exc)
                if isinstance(exc, WorkerDied):
                    raise
                return
            finally:
                self.metrics.record(f'gpu.{label}', time.perf_counter() - started,
                                    started - submitted)
            if label in ('ritm_click', 'ritm_undo') and isinstance(result, dict):
                result['queue_seconds'] = started - submitted
            future.set_result(result)

        try:
            self.scheduler.submit(session_id, run, label)
        except QueueFull as exc:
            raise ServiceError('queue_full', str(exc)) from None
        try:
            return future.result(timeout=self.request_timeout)
        except FutureTimeout:
            future.cancel()
            raise ServiceError('timeout', f'the GPU did not serve the request within '
                               f'{self.request_timeout:.0f} s') from None

    def _dispatch_loop(self) -> None:
        while self.running:
            decision = self.scheduler.next(timeout=0.5)
            if decision is None:
                continue
            try:
                if decision.kind == 'interactive':
                    decision.item.run()
                else:
                    started = time.perf_counter()
                    self.jobs.run(decision)
                    self.metrics.record('gpu.propagation_step', time.perf_counter() - started,
                                        decision.waited)
            except WorkerDied as exc:
                self._worker_failed(str(exc))
            except Exception as exc:
                log.exception('dispatcher')
                if decision.kind == 'job':
                    try:
                        self.jobs.abort(decision.job_id, f'{type(exc).__name__}: {exc}')
                    except WorkerDied as died:
                        self._worker_failed(str(died))

    def _worker_failed(self, reason: str) -> None:
        log.error('GPU worker failed: %s', reason)
        failed_at = time.perf_counter()
        self._event('worker_failed', {'reason': reason, 'generation': self.worker.generation})
        self.worker.kill()
        self.jobs.worker_lost()
        for attempt in range(3):
            try:
                self._start_worker()
                self.metrics.record('worker_recovery', time.perf_counter() - failed_at, 0.0)
                return
            except WorkerDied as exc:
                log.error('GPU worker restart %d failed: %s', attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        self.worker_problem = reason
        for job in self.jobs.active():
            self.jobs._fail(job, f'the GPU worker could not be restarted: {reason}')

    def _maintenance_loop(self) -> None:
        last_health = last_prune = time.monotonic()
        while self.running:
            time.sleep(1.0)
            try:
                self.sessions.expire_stale()
            except Exception:
                log.exception('session expiry')
            now = time.monotonic()
            if now - last_health >= 5.0:
                last_health = now
                self._check_worker()
                try:
                    self.tasks.retry_stale_views()
                except Exception:
                    log.exception('rewriting task files')
            if now - last_prune >= 600.0:
                last_prune = now
                self.jobs.prune()
                with self.db.transaction() as conn:
                    conn.execute("DELETE FROM requests WHERE created_at < datetime('now', '-7 days')")

    def _check_worker(self) -> None:
        """A worker that died while idle is replaced now, not at the next request: a status
        request through the scheduler meets the dead worker on the dispatcher thread, which
        recovers it."""
        if self.worker_problem is not None or self.worker.alive():
            return
        log.warning('the GPU worker is not running; replacing it')
        try:
            self.scheduler.submit('_service', lambda: self.worker.call('status', timeout=30),
                                  'health')
        except QueueFull:
            pass

    def _session_ended(self, session_id: str) -> None:
        self.jobs.cancel_session(session_id, 'the session ended')
        self.interactions.close_session(session_id)
        self.scheduler.drop_session(session_id)
        for context in self.tasks.leases_of(session_id):
            self.tasks.close_task(context.task_id, session_id, context.epoch, status='expired')

    # requests --------------------------------------------------------------------------------
    def handle(self, request: dict) -> dict:
        op = request.get('op')
        args = request.get('args') or {}
        handler = getattr(self, f'op_{op}', None)
        started = time.perf_counter()
        try:
            if handler is None:
                raise ServiceError('invalid', f'unknown operation {op!r}')
            try:
                inspect.signature(handler).bind(**args)
            except TypeError as exc:
                raise ServiceError('invalid', f'{op}: {exc}') from None
            if op not in ('hello', 'status', 'register_session', 'shutdown'):
                session_id = args.get('session_id')
                if not session_id or not self.sessions.heartbeat(session_id):
                    raise ServiceError('session_expired', 'this window\'s session has expired; '
                                       'it reconnects')
            result = handler(**args)
            response = {'ok': True, 'result': result}
        except (ServiceError, TaskError, JobError) as exc:
            response = {'ok': False, 'error': dict(getattr(exc, 'details', {}),
                                                   code=exc.code, message=str(exc))}
        except InteractionError as exc:
            response = {'ok': False, 'error': {'code': 'invalid', 'message': str(exc)}}
        except SessionError as exc:
            response = {'ok': False, 'error': {'code': 'session_expired', 'message': str(exc)}}
        except WorkerError as exc:
            response = {'ok': False, 'error': {'code': exc.kind, 'message': str(exc)}}
        except WorkerDied as exc:
            response = {'ok': False, 'error': {'code': 'worker_restarted', 'message': str(exc)}}
        except Exception as exc:
            log.exception('request %s failed', op)
            response = {'ok': False, 'error': {'code': 'internal',
                                               'message': f'{type(exc).__name__}: {exc}'}}
        self.metrics.record(f'request.{op}', time.perf_counter() - started, 0.0)
        return response

    # operations ------------------------------------------------------------------------------
    def op_hello(self) -> dict:
        return {'version': VERSION, 'task_root': self.task_root,
                'heartbeat_seconds': float(self.settings.get('heartbeat_seconds', 5)),
                'session_timeout_seconds': self.sessions.timeout,
                'gpu': self.worker_hello.get('hardware', {}).get('name'),
                'worker_ready': self.worker_problem is None and self.worker.alive()}

    def op_status(self, session_id: Optional[str] = None) -> dict:
        status = {
            'service': {'pid': os.getpid(), 'task_root': self.task_root,
                        'state_dir': self.state_dir, 'socket': self.socket_path},
            'worker': {'pid': self.worker.pid, 'generation': self.worker.generation,
                       'alive': self.worker.alive(), 'problem': self.worker_problem,
                       'hardware': self.worker_hello.get('hardware'),
                       'model_bytes': self.worker_hello.get('model_bytes'),
                       'load_seconds': self.worker_hello.get('load_seconds'),
                       'start_seconds': self.worker.start_seconds,
                       'memory_model': {k: v for k, v in
                                        (self.worker_hello.get('memory_model') or {}).items()
                                        if k != 'samples'},
                       'quality_changes': self.worker_hello.get('quality_changes')},
            'scheduler': self.scheduler.snapshot(),
            'sessions': self.sessions.active(),
            'leases': [{'task_id': c.task_id, 'session_id': c.session_id, 'epoch': c.epoch}
                       for c in self.tasks.leases.values()],
            'jobs': [{'job_id': j.job_id, 'task_id': j.task_id, 'state': j.state,
                      'committed': len(j.committed), 'attempt': j.attempt}
                     for j in self.jobs.jobs.values() if not j.done],
            'metrics': self.metrics.summary(),
            'recovery': self.recovery_report,
        }
        return status

    def op_gpu_status(self, session_id: str) -> dict:
        """The worker's current memory (asked through the scheduler, like any GPU request)."""
        return self.run_on_gpu(session_id, lambda: self.worker.call('status'), 'status')

    def op_register_session(self, annotator_name: str, vnc_display: Optional[str] = None,
                            process_id: Optional[int] = None, host: Optional[str] = None,
                            session_id: Optional[str] = None) -> dict:
        reply = self.sessions.register(annotator_name, vnc_display, process_id, host, session_id)
        return dict(reply, heartbeat_seconds=float(self.settings.get('heartbeat_seconds', 5)),
                    session_timeout_seconds=self.sessions.timeout)

    def op_heartbeat(self, session_id: str) -> dict:
        return {'leases': [{'task_id': c.task_id, 'epoch': c.epoch}
                           for c in self.tasks.leases_of(session_id)]}

    def op_close_session(self, session_id: str) -> dict:
        self.sessions.close(session_id)
        return {}

    def op_open_task(self, session_id: str, workspace: str) -> dict:
        return self.tasks.open_task(workspace, session_id)

    def op_close_task(self, session_id: str, task_id: str, lease_epoch: int) -> dict:
        for job in self.jobs.active():
            if job.task_id == task_id and job.session_id == session_id:
                self.jobs.cancel(job.job_id, reason='the task was closed')
        for record in self.interactions.records():
            if record.task_id == task_id and record.session_id == session_id:
                self.interactions.close(record.interaction_id)
        return {'closed': self.tasks.close_task(task_id, session_id, lease_epoch)}

    def op_open_interaction(self, session_id: str, task_id: str, lease_epoch: int, frame: str,
                            modality: str, target_object: int) -> dict:
        return self.interactions.open(session_id, task_id, lease_epoch, frame, modality,
                                      target_object)

    def _interaction(self, session_id: str, interaction_id: str, task_id: Optional[str],
                     lease_epoch: Optional[int]) -> None:
        record = self.interactions.get(interaction_id, session_id)
        if (task_id is not None and task_id != record.task_id) or \
                (lease_epoch is not None and lease_epoch != record.lease_epoch):
            raise ServiceError('invalid', 'the interaction belongs to another task or lease')

    def op_click(self, session_id: str, interaction_id: str, x: int, y: int, is_negative: bool,
                 request_id: Optional[str] = None, task_id: Optional[str] = None,
                 lease_epoch: Optional[int] = None) -> dict:
        self._interaction(session_id, interaction_id, task_id, lease_epoch)
        return self.interactions.click(interaction_id, session_id, x, y, is_negative, request_id)

    def op_undo_click(self, session_id: str, interaction_id: str,
                      request_id: Optional[str] = None, task_id: Optional[str] = None,
                      lease_epoch: Optional[int] = None) -> dict:
        self._interaction(session_id, interaction_id, task_id, lease_epoch)
        return self.interactions.undo(interaction_id, session_id, request_id)

    def op_replay_interaction(self, session_id: str, interaction_id: str) -> dict:
        return self.interactions.replay(interaction_id, session_id)

    def op_close_interaction(self, session_id: str, interaction_id: str) -> dict:
        self.interactions.close(interaction_id, session_id)
        return {}

    def op_commit_frame(self, session_id: str, task_id: str, lease_epoch: int, frame: str,
                        mask=None, anchor: Optional[str] = None,
                        request_id: Optional[str] = None) -> dict:
        self._no_running_job(task_id)
        return self.tasks.commit_frame(task_id, session_id, lease_epoch, frame, mask=mask,
                                       anchor=anchor, request_id=request_id, op='edit')

    def op_remove_anchor(self, session_id: str, task_id: str, lease_epoch: int, frame: str,
                         request_id: Optional[str] = None) -> dict:
        self._no_running_job(task_id)
        return self.tasks.commit_frame(task_id, session_id, lease_epoch, frame, remove_mask=True,
                                       anchor='remove', request_id=request_id, op='remove_anchor')

    def op_restore_frame(self, session_id: str, task_id: str, lease_epoch: int, frame: str,
                         mask_png: Optional[bytes] = None, anchor_entry: Optional[dict] = None,
                         request_id: Optional[str] = None) -> dict:
        self._no_running_job(task_id)
        return self.tasks.commit_frame(task_id, session_id, lease_epoch, frame, mask_bytes=mask_png,
                                       remove_mask=mask_png is None, anchor='restore',
                                       restore_anchor=anchor_entry, request_id=request_id,
                                       op='restore')

    def _no_running_job(self, task_id: str) -> None:
        for job in self.jobs.active():
            if job.task_id == task_id:
                raise ServiceError('busy', 'pause the propagation first')

    def op_set_intervals(self, session_id: str, task_id: str, lease_epoch: int, intervals,
                         request_id: Optional[str] = None) -> dict:
        return self.tasks.set_intervals(task_id, session_id, lease_epoch, intervals, request_id)

    def op_set_description(self, session_id: str, task_id: str, lease_epoch: int, text: str,
                           request_id: Optional[str] = None) -> dict:
        return self.tasks.set_description(task_id, session_id, lease_epoch, text, request_id)

    def op_set_attention(self, session_id: str, task_id: str, lease_epoch: int, flagged: bool,
                         request_id: Optional[str] = None) -> dict:
        return self.tasks.set_attention(task_id, session_id, lease_epoch, flagged, request_id)

    def op_set_discard(self, session_id: str, task_id: str, lease_epoch: int, discarded: bool,
                       request_id: Optional[str] = None) -> dict:
        return self.tasks.set_discard(task_id, session_id, lease_epoch, discarded, request_id)

    def op_submit_for_review(self, session_id: str, task_id: str, lease_epoch: int,
                             target_objects, request_id: Optional[str] = None) -> dict:
        self._no_running_job(task_id)
        return self.tasks.submit_for_review(task_id, session_id, lease_epoch, target_objects,
                                            request_id)

    def op_reset_all_masks(self, session_id: str, task_id: str, lease_epoch: int, export_dirs,
                           request_id: Optional[str] = None) -> dict:
        self._no_running_job(task_id)
        return self.tasks.reset_all(task_id, session_id, lease_epoch, export_dirs, request_id)

    def op_start_propagation(self, session_id: str, task_id: str, lease_epoch: int,
                             start_frame: str, direction: str, memory: dict,
                             request_id: Optional[str] = None,
                             modality: Optional[str] = None) -> dict:
        try:
            return self.jobs.create(task_id, session_id, lease_epoch, start_frame, direction,
                                    memory, request_id, modality=modality)
        except QueueFull as exc:
            raise ServiceError('queue_full', str(exc)) from None

    def op_poll_propagation(self, session_id: str, job_id: str, since: int = 0,
                            messages_since: int = 0) -> dict:
        return self.jobs.poll(job_id, session_id, since, messages_since)

    def op_cancel_propagation(self, session_id: str, job_id: str) -> dict:
        self.jobs.cancel(job_id, session_id)
        return {}

    def op_begin_export(self, session_id: str, task_id: str, lease_epoch: int) -> dict:
        self._no_running_job(task_id)
        return self.tasks.begin_export(task_id, session_id, lease_epoch)

    def op_finish_export(self, session_id: str, export_id: str, exit_ok: bool) -> dict:
        return self.tasks.finish_export(export_id, session_id, exit_ok)

    def op_export_status(self, session_id: str, task_id: str) -> dict:
        self.tasks.info_by_id(task_id)
        if self.tasks.task_row(task_id) is None:
            return {'status': 'unknown', 'review_current': False, 'export_current': False}
        return self.tasks.export_status(task_id)

    def op_mark_done(self, session_id: str, task_id: str, lease_epoch: int,
                     request_id: Optional[str] = None) -> dict:
        return self.tasks.mark_done(task_id, session_id, lease_epoch, request_id)

    def op_test_worker(self, session_id: str, command: str, **args) -> dict:
        """Test hooks of the fake engine (allow_test_commands only)."""
        if not self.settings.get('allow_test_commands'):
            raise ServiceError('invalid', 'test commands are disabled')
        return self.run_on_gpu(session_id, lambda: self.worker.call(command, **args), command)

    def op_shutdown(self, session_id: Optional[str] = None) -> dict:
        threading.Thread(target=self._shutdown_soon, daemon=True).start()
        return {'stopping': True}

    def _shutdown_soon(self) -> None:
        time.sleep(0.2)
        if self.on_shutdown is not None:
            self.on_shutdown()

    on_shutdown: Optional[Callable[[], None]] = None
