"""
Propagation jobs of the shared service.

A job is created from an immutable snapshot of its inputs (frames, anchors and their committed
masks, the start frame and its mask, direction, processing size, memory settings, model identity)
taken at one input revision of the task. It runs in attempts: each attempt builds a fresh Cutie
context in the GPU worker and steps it. A propagated mask is committed before the job may take its
next step; a commit that is refused (the lease is gone, a human input changed, the attempt is
obsolete) stops the job. When an attempt fails (worker crash, out of memory), a new attempt
rebuilds the context from the snapshot and replays the frames already committed without writing
them again, then continues from the last confirmed frame.

Worker commands are only sent from the dispatcher thread; commits run on the commit thread.
"""
import hashlib
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from cutieplus.coordinator.scheduler import Decision, Scheduler
from cutieplus.coordinator.tasks import TaskError, TaskService, encode_mask
from cutieplus.coordinator.worker_proxy import WorkerDied, WorkerError
from cutieplus.storage.db import Database, dumps, loads

log = logging.getLogger(__name__)

FINAL_STATES = ('finished', 'cancelled', 'failed')


class JobError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class JobConfig:
    memory_retry_seconds: float = 2.0  # how often a job waiting for memory asks again
    queue_wait_seconds: float = 600.0  # then it is rejected
    max_oom_retries: int = 1
    max_worker_restarts: int = 2  # per job
    pressure_retries: int = 20  # steps postponed while memory is short, then the context is freed
    max_events: int = 100000


@dataclass
class Job:
    job_id: str
    task_id: str
    session_id: str
    lease_epoch: int
    input_revision: int
    snapshot: dict
    created: float = field(default_factory=time.monotonic)
    state: str = 'queued'  # queued | waiting_memory | running | committing | + FINAL_STATES
    attempt: int = 0
    attempt_id: str = ''
    stop_reason: Optional[str] = None
    stopped_at: Optional[int] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    release_needed: bool = False  # the worker holds a context to free
    committed: Dict[int, str] = field(default_factory=dict)  # ti -> mask object
    events: List[int] = field(default_factory=list)  # committed frames, in commit order
    messages: List[str] = field(default_factory=list)
    latest: Optional[tuple] = None  # (ti, mask) last committed, for display
    effective: List[dict] = field(default_factory=list)  # resource plan of every attempt
    forced_permanent: Optional[List[int]] = None  # a replay keeps the anchors of the first plan
    memory_wait_since: Optional[float] = None
    oom_retries: int = 0
    oom_budget: Optional[int] = None  # memory available when the last attempt ran out
    worker_restarts: int = 0
    pressure_count: int = 0
    replay_checked: int = 0
    replay_mismatches: int = 0
    content_revision: Optional[int] = None  # of the task after this job's last commit
    metrics: Dict[str, float] = field(default_factory=lambda: {
        'queue_seconds': 0.0, 'inference_seconds': 0.0, 'commit_seconds': 0.0,
        'commit_wait_seconds': 0.0, 'steps': 0, 'replayed_frames': 0})
    started: Optional[float] = None
    finished: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def done(self) -> bool:
        return self.state in FINAL_STATES

    def message(self, text: str) -> None:
        self.messages.append(text)
        log.info('job %s: %s', self.job_id, text)


class JobManager:

    def __init__(self, db: Database, tasks: TaskService, scheduler: Scheduler,
                 worker_call: Callable, worker_info: Callable[[], dict], config: JobConfig,
                 clock: Callable[[], float] = time.monotonic):
        self.db = db
        self.tasks = tasks
        self.scheduler = scheduler
        self.worker_call = worker_call  # (op, **args) on the dispatcher thread
        self.worker_info = worker_info  # the worker's hello (model identity)
        self.config = config
        self.clock = clock
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()
        self.committer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='commit')
        self.listeners: List[Callable[[Job], None]] = []

    # creation ------------------------------------------------------------------------------
    def create(self, task_id: str, session_id: str, epoch: int, start_frame: str, direction: str,
               memory: dict, request_id: Optional[str] = None) -> dict:
        if request_id is not None:
            stored = self.db.query_one('SELECT response FROM requests WHERE request_id = ?',
                                       (request_id, ))
            if stored is not None:
                return loads(stored['response'])
        context = self.tasks.context(task_id, session_id, epoch)
        info = self.tasks.info_by_id(task_id)
        if direction not in ('forward', 'backward'):
            raise JobError('invalid', f'unknown direction {direction!r}')
        if start_frame not in info.name_to_ti:
            raise JobError('invalid', f'{task_id} has no frame {start_frame}')
        with self.lock:
            for job in self.jobs.values():
                if job.task_id == task_id and not job.done:
                    raise JobError('busy', f'{task_id} is already propagating ({job.job_id})')
        row = self.tasks.task_row(task_id)
        frames = self.tasks.frames(task_id)
        start_entry = frames.get(start_frame)
        if start_entry is not None and (start_entry['anchor'] or {}).get('kind') == 'empty':
            raise JobError('invalid', 'the target is marked absent here (empty anchor); start from '
                           'a frame that has a mask')
        anchors = {frame: entry['anchor']['kind'] for frame, entry in frames.items()
                   if entry['anchor'] is not None}
        anchor_masks = {}
        for frame, kind in anchors.items():
            if kind == 'mask':
                source = self.tasks.mask_source(info, frame, frames, row['state'])
                if source is None:
                    raise JobError('invalid', f'anchor {frame} has no mask; remove the anchor or '
                                   'import its mask')
                anchor_masks[str(info.name_to_ti[frame])] = source
        hello = self.worker_info()
        snapshot = {
            'task_id': task_id,
            'session_id': session_id,
            'input_revision': row['input_revision'],
            'model_identity': hello.get('model_identity'),
            'model_configuration': hello.get('model_configuration'),
            'frame_names': list(info.names),
            'frame_paths': [info.image_path(name) for name in info.names],
            'anchors': anchors,
            'anchor_snapshot': {frame: {'kind': entry['anchor']['kind'],
                                        'mask_object': entry['mask_object']}
                                for frame, entry in frames.items() if entry['anchor'] is not None},
            'anchor_masks': anchor_masks,
            'anchor_modality': info.anchor_modality,
            'start_ti': info.name_to_ti[start_frame],
            'start_frame': start_frame,
            'direction': direction,
            'seed_mask': self.tasks.mask_source(info, start_frame, frames, row['state']),
            'height': info.height,
            'width': info.width,
            'num_objects': 1,
            'resolution': {'max_internal_size': context.resolution.max_internal_size,
                           'upscale_to_internal_size': context.resolution.upscale_to_internal_size},
            'memory': dict(memory),
            'click_frame': [info.height, info.width],
        }
        job_id = uuid.uuid4().hex[:16]
        job = Job(job_id, task_id, session_id, epoch, row['input_revision'], snapshot)
        now = self.tasks.clock()
        response = {'job_id': job_id}
        with self.db.transaction() as conn:
            conn.execute('INSERT INTO jobs (job_id, task_id, session_id, created_at, updated_at, '
                         'status, attempts, snapshot) VALUES (?, ?, ?, ?, ?, ?, 0, ?)',
                         (job_id, task_id, session_id, now, now, 'queued', dumps(snapshot)))
            if request_id is not None:
                conn.execute('INSERT INTO requests VALUES (?, ?, ?, ?, ?)',
                             (request_id, session_id, 'start_propagation', now, dumps(response)))
        with self.lock:
            self.jobs[job_id] = job
        self.scheduler.add_job(job_id, session_id)
        return response

    def get(self, job_id: str, session_id: Optional[str] = None) -> Job:
        job = self.jobs.get(job_id)
        if job is None or (session_id is not None and job.session_id != session_id):
            raise JobError('invalid', f'no propagation job {job_id}')
        return job

    # client view ---------------------------------------------------------------------------
    def poll(self, job_id: str, session_id: str, since: int = 0, messages_since: int = 0) -> dict:
        job = self.get(job_id, session_id)
        with job.lock:
            frames = job.events[since:]
            latest = job.latest if frames else None
            return {
                'state': job.state,
                'frames': frames,
                'next': len(job.events),
                'latest_ti': None if latest is None else latest[0],
                'latest_mask': None if latest is None else latest[1],
                'messages': job.messages[messages_since:],
                'next_message': len(job.messages),
                'stop_reason': job.stop_reason,
                'stopped_at': job.stopped_at,
                'error': job.error,
                'queue_position': (self.scheduler.queue_position(job_id)
                                   if job.state in ('queued', 'waiting_memory') else None),
                'attempt': job.attempt,
                'content_revision': job.content_revision,
                'metrics': dict(job.metrics, replay_checked=job.replay_checked,
                                replay_mismatches=job.replay_mismatches,
                                oom_retries=job.oom_retries, worker_restarts=job.worker_restarts,
                                run_seconds=(job.finished or self.clock()) - job.started
                                if job.started else None),
                'effective': job.effective,
            }

    def cancel(self, job_id: str, session_id: Optional[str] = None, reason: str = 'cancelled') -> None:
        job = self.get(job_id, session_id)
        with job.lock:
            if job.done:
                return
            job.cancel_requested = True
            job.stop_reason = reason
            holds_context = job.release_needed
        if holds_context:
            self.scheduler.job_ready(job_id)  # the dispatcher frees the context
        else:
            self._finish(job, 'cancelled', reason)

    def cancel_session(self, session_id: str, reason: str) -> None:
        for job in list(self.jobs.values()):
            if job.session_id == session_id and not job.done:
                self.cancel(job.job_id, reason=reason)

    # dispatcher ----------------------------------------------------------------------------
    def run(self, decision: Decision) -> None:
        """One scheduling decision for a job (dispatcher thread)."""
        job = self.jobs.get(decision.job_id)
        if job is None or job.done:
            self.scheduler.remove_job(decision.job_id)
            return
        if job.cancel_requested:
            self._release(job)
            self._finish(job, 'cancelled', job.stop_reason or 'cancelled')
            return
        if decision.admission:
            job.metrics['queue_seconds'] += decision.waited
            self._admit(job)
        else:
            self._step(job)

    def _admit(self, job: Job) -> None:
        if job.oom_budget is not None and job.forced_permanent is None and job.effective:
            # after running out of memory: the same work under the same conditions would fail
            # again, so it waits until the plan or the free memory changes
            probe = self.worker_call('prop_start', plan_only=True,
                                     job=dict(job.snapshot, job_id=job.job_id, attempt_id='plan',
                                              permanent_tis=None))
            plan = probe['plan']
            if probe['decision'] == 'run' and plan['budget']['available'] <= job.oom_budget and \
                    len(plan['permanent_tis']) >= len(job.effective[-1]['permanent_tis']):
                self._wait_for_memory(job, 'the last attempt ran out of GPU memory; waiting for '
                                      'more free memory')
                return
        job.attempt += 1
        job.attempt_id = f'{job.job_id}.{job.attempt}'
        payload = dict(job.snapshot, job_id=job.job_id, attempt_id=job.attempt_id,
                       permanent_tis=job.forced_permanent)
        try:
            reply = self.worker_call('prop_start', job=payload)
        except WorkerError as exc:
            if exc.kind == 'oom':
                self._out_of_memory(job, str(exc))
                return
            self._fail(job, f'the GPU worker refused the job: {exc}')
            return
        plan = reply['plan']
        if reply['decision'] == 'run':
            job.state = 'running'
            job.release_needed = True
            job.memory_wait_since = None
            job.started = job.started or self.clock()
            plan = dict(plan, attempt_id=job.attempt_id)
            job.effective.append(plan)
            mask_anchors = len(job.snapshot['anchor_masks'])
            if plan['limited_by'] == 'memory':
                job.message(f'GPU memory allows {len(plan["permanent_tis"])} of {mask_anchors} mask '
                            'anchors in the permanent memory (those nearest to the start frame); '
                            'the others guide the run when it reaches them.')
            if job.forced_permanent is None and job.attempt == 1:
                job.forced_permanent = list(plan['permanent_tis'])
            if job.committed:
                job.message(f'Rebuilding the propagation (attempt {job.attempt}): replaying '
                            f'{len(job.committed)} confirmed frames without writing them.')
            self._update_row(job)
            self.scheduler.job_ready(job.job_id)
        elif reply['decision'] == 'queue':
            self._wait_for_memory(job, plan['reason'])
        else:
            self._fail(job, f'not enough GPU memory: {plan["reason"]}')

    def _wait_for_memory(self, job: Job, reason: str) -> None:
        now = self.clock()
        if job.memory_wait_since is None:
            job.memory_wait_since = now
            job.message(f'Waiting for GPU memory: {reason}.')
        if now - job.memory_wait_since > self.config.queue_wait_seconds:
            self._fail(job, f'GPU memory stayed short for {self.config.queue_wait_seconds:.0f} s: '
                       f'{reason}')
            return
        job.state = 'waiting_memory'
        self.scheduler.requeue(job.job_id, now + self.config.memory_retry_seconds)

    def _step(self, job: Job) -> None:
        try:
            reply = self.worker_call('prop_step', job_id=job.job_id, attempt_id=job.attempt_id)
        except WorkerError as exc:
            if exc.kind == 'oom':
                job.release_needed = False  # the worker freed the context
                self._out_of_memory(job, str(exc))
            elif exc.kind == 'pressure':
                self._pressure(job, str(exc))
            else:
                self._release(job)
                self._fail(job, f'propagation step failed: {exc}')
            return
        job.metrics['steps'] += 1
        job.metrics['inference_seconds'] += float(reply.get('inference_seconds') or 0.0)
        job.pressure_count = 0
        for text in reply.get('messages') or ():
            job.message(text)
        kind, ti = reply['kind'], reply['ti']
        if reply['finished']:
            job.release_needed = False  # the worker frees a finished context itself
        if kind == 'stop':
            job.stopped_at = ti
            self._finish(job, 'finished', 'empty_anchor')
            return
        if kind == 'frame':
            if ti in job.committed:
                self._check_replayed(job, ti, reply['mask'])
            else:
                job.state = 'committing'
                self.scheduler.job_blocked(job.job_id)
                attempt_id = job.attempt_id
                submitted = self.clock()
                self.committer.submit(self._commit, job, attempt_id, ti, reply['mask'],
                                      reply['finished'], submitted)
                return
        if reply['finished']:
            self._finish(job, 'finished', reply.get('stop_reason') or 'end')
        else:
            self.scheduler.job_ready(job.job_id)

    def _check_replayed(self, job: Job, ti: int, mask: np.ndarray) -> None:
        """A frame recomputed while rebuilding a context: never written again; compared with what
        was committed."""
        job.metrics['replayed_frames'] += 1
        job.replay_checked += 1
        if hashlib.sha256(encode_mask(mask)).hexdigest() != job.committed[ti]:
            job.replay_mismatches += 1
            log.warning('job %s: replayed frame %d differs from its committed mask', job.job_id, ti)

    def _commit(self, job: Job, attempt_id: str, ti: int, mask: np.ndarray, finished: bool,
                submitted: float) -> None:
        """Commit thread: the step's mask must be confirmed before the next step."""
        start = self.clock()
        name = job.snapshot['frame_names'][ti]
        try:
            result = self.tasks.commit_propagated(
                job.task_id, job.session_id, job.lease_epoch, job.input_revision, name, mask,
                job.job_id, attempt_id,
                lambda: job.attempt_id == attempt_id and not job.cancel_requested)
        except TaskError as exc:
            if job.attempt_id != attempt_id:
                return  # an older attempt's result; the current attempt recomputes the frame
            job.message(f'Propagation stopped at {name}: {exc}')
            with job.lock:
                job.stop_reason = f'{exc.code}: {exc}'
                job.cancel_requested = True
            if job.release_needed:
                self.scheduler.job_ready(job.job_id)  # the dispatcher frees the context
            else:
                self._finish(job, 'cancelled', job.stop_reason)
            return
        except Exception as exc:  # storage failure: never continue from an unconfirmed state
            log.exception('job %s: commit of %s failed', job.job_id, name)
            with job.lock:
                job.stop_reason = f'storage: {exc}'
                job.cancel_requested = True
            self._fail_after_release(job, f'saving {name} failed: {exc}')
            return
        job.metrics['commit_seconds'] += self.clock() - start
        job.metrics['commit_wait_seconds'] += self.clock() - submitted
        with job.lock:
            job.committed[ti] = result['mask_object']
            job.events.append(ti)
            job.latest = (ti, mask)
            job.content_revision = result['content_revision']
            if not result.get('views_written', True):
                job.messages.append(f'{name} is saved, but its file in masks/ could not be '
                                    'updated yet.')
        if finished:
            self._finish(job, 'finished', 'end')
        else:
            job.state = 'running'
            self.scheduler.job_ready(job.job_id)

    def _fail_after_release(self, job: Job, error: str) -> None:
        job.error = error
        if job.release_needed:
            self.scheduler.job_ready(job.job_id)
        else:
            self._fail(job, error)

    def _pressure(self, job: Job, reason: str) -> None:
        job.pressure_count += 1
        if job.pressure_count == 1:
            job.message(f'GPU memory is short ({reason}); the next step waits.')
        if job.pressure_count <= self.config.pressure_retries:
            self.scheduler.job_ready(job.job_id, not_before=self.clock() + 0.5)
            return
        # still short: free the context and wait for memory as a queued job; the replay keeps
        # the confirmed frames
        self._release(job)
        job.pressure_count = 0
        job.forced_permanent = None  # memory limits may apply again (recorded per attempt)
        self._wait_for_memory(job, reason)

    def _out_of_memory(self, job: Job, reason: str) -> None:
        job.oom_retries += 1
        status = self.worker_call('status')
        job.oom_budget = status['budget']['available']
        if job.oom_retries > self.config.max_oom_retries:
            self._fail(job, f'out of GPU memory again after {job.oom_retries - 1} retries: {reason}')
            return
        job.message(f'Out of GPU memory ({reason}); the context was freed and the run is rebuilt.')
        job.forced_permanent = None  # a memory limit may now keep fewer anchors (recorded)
        job.state = 'waiting_memory'
        self.scheduler.requeue(job.job_id, self.clock() + self.config.memory_retry_seconds)

    def _release(self, job: Job) -> None:
        if job.release_needed:
            try:
                self.worker_call('prop_release', job_id=job.job_id)
            except (WorkerError, WorkerDied):
                pass
            job.release_needed = False
        self.scheduler.release(job.job_id)

    def _fail(self, job: Job, error: str) -> None:
        job.error = error
        job.message(f'Propagation failed: {error}')
        self._finish(job, 'failed', error)

    def _finish(self, job: Job, state: str, reason: Optional[str]) -> None:
        with job.lock:
            if job.done:
                return
            if job.error is not None and state == 'cancelled':
                state = 'failed'
            job.state = state
            job.stop_reason = job.stop_reason if state == 'cancelled' and job.stop_reason else reason
            job.finished = self.clock()
        self.scheduler.remove_job(job.job_id)
        self._update_row(job)
        for listener in self.listeners:
            listener(job)

    def _update_row(self, job: Job) -> None:
        metrics = dict(job.metrics, replay_checked=job.replay_checked,
                       replay_mismatches=job.replay_mismatches, oom_retries=job.oom_retries,
                       worker_restarts=job.worker_restarts)
        if job.started is not None and job.finished is not None:
            metrics['run_seconds'] = job.finished - job.started
        try:
            with self.db.transaction() as conn:
                conn.execute('UPDATE jobs SET updated_at = ?, status = ?, attempt_id = ?, '
                             'attempts = ?, effective = ?, committed_count = ?, '
                             'last_committed_ti = ?, stop_reason = ?, error = ?, metrics = ? '
                             'WHERE job_id = ?',
                             (self.tasks.clock(), job.state, job.attempt_id, job.attempt,
                              dumps(job.effective), len(job.committed),
                              job.events[-1] if job.events else None, job.stop_reason, job.error,
                              dumps(metrics), job.job_id))
        except Exception:
            log.exception('job %s: recording its state failed', job.job_id)

    # recovery ------------------------------------------------------------------------------
    def worker_lost(self) -> None:
        """The worker died: every job that held a context starts a new attempt (after the worker
        restarts), replaying its confirmed frames; jobs whose inputs changed stop instead."""
        for job in list(self.jobs.values()):
            if job.done or not job.release_needed and job.state not in ('running', 'committing'):
                continue
            job.release_needed = False
            self.scheduler.release(job.job_id)
            job.worker_restarts += 1
            row = self.tasks.task_row(job.task_id)
            if row is None or row['input_revision'] != job.input_revision:
                self._finish(job, 'cancelled', 'the task changed while the GPU worker restarted')
                continue
            if job.worker_restarts > self.config.max_worker_restarts:
                self._fail(job, f'the GPU worker failed {job.worker_restarts} times during this job')
                continue
            job.message('The GPU worker stopped; the propagation resumes from the last confirmed '
                        'frame once it is back.')
            # results of the old attempt are obsolete from now on
            job.attempt_id = f'{job.job_id}.{job.attempt}.lost'
            job.state = 'queued'
            self.scheduler.requeue(job.job_id, 0.0)

    def active(self) -> List[Job]:
        return [job for job in self.jobs.values() if not job.done]

    def prune(self, max_age: float = 3600.0) -> int:
        """Forgets jobs finished more than max_age seconds ago (their record stays in the
        database); the windows have read their end long before."""
        limit = self.clock() - max_age
        with self.lock:
            old = [job_id for job_id, job in self.jobs.items()
                   if job.done and job.finished is not None and job.finished < limit]
            for job_id in old:
                del self.jobs[job_id]
        return len(old)

    def shutdown(self) -> None:
        self.committer.shutdown(wait=True)
