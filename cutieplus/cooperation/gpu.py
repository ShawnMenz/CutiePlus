"""
The one GPU queue of the cooperation module. Every model inference of every window runs here, in
the one GPU worker (worker_link.py), one command at a time:

- Propagations are jobs (gpu_jobs), first in first out, one running at a time. A job's input is
  the snapshot taken when it was submitted (frames, anchors, anchor masks, the start frame's
  probabilities, memory settings), stored by the service and never changed. A job is claimed in
  one short transaction (queued -> running, with an execution token and the worker generation),
  then run a step at a time; no transaction is held during a step.
- Clicks are short jobs too (operation_type click / click_undo) and run between two steps of the
  running propagation, round-robin between users: a long run never makes a click wait for its end.
  No role goes first.

A propagated mask is staged by the worker under the job's folder; the service records its SHA-256
in the job's manifest (<state_dir>/results/<job>.json, appended as the run goes, so a crash keeps
what was done). The window that submitted the job applies the result to its task once done, if
the task did not change meanwhile, else it is held for review (cutieplus/backends/service.py).

Cancelling a queued job is immediate. A running one is marked cancel_requested; the service sends
no further step, asks the worker to release the run (the worker answers once its memory is freed
and the device synchronized), and only then marks it cancelled and moves on. A worker that goes
away leaves its running job interrupted, with the frames done so far as its result; nothing is
retried by itself: Retry makes a new job from the same snapshot.
"""
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from os import path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from cutieplus.cooperation import audit, authz, validate
from cutieplus.cooperation.api import Context, op
from cutieplus.cooperation.clock import new_id
from cutieplus.cooperation.errors import (CONFLICT, FORBIDDEN, INVALID, LIMIT, NOT_FOUND,
                                          UNAVAILABLE, Refused)
from cutieplus.cooperation.protocol import decode_message, encode_message
from cutieplus.cooperation.service import Service
from cutieplus.cooperation.worker_link import GPU_ID, CommandFailed, WorkerLink, WorkerLost

log = logging.getLogger(__name__)

CLICK_TIMEOUT_S = 60.0  # a click waits at most this long for the worker
STEP_TIMEOUT_S = 300.0  # the first step of a run compiles kernels
START_TIMEOUT_S = 300.0
MAX_BURST = 4  # clicks in a row before the running propagation gets a step
EVENT_BUFFER = 256  # frames kept with their masks for the window to fetch
HEARTBEAT_S = 5.0
MAX_FRAMES = 200_000
MAX_MASK_BYTES = 16 << 20
MAX_SEED_BYTES = 256 << 20
DIRECTIONS = ('forward', 'backward')
MEMORY_KEYS = ('mem_every', 'top_k', 'min_mem_frames', 'max_mem_frames', 'max_num_tokens')
FINAL = ('succeeded', 'failed', 'cancelled', 'interrupted')
RESULT_OUTCOMES = ('applied', 'held_for_review', 'discarded')


@dataclass
class Interaction:
    id: str
    session_id: str
    user_id: str
    workspace: str
    image_path: str
    mask_png: Optional[bytes]
    num_objects: int
    target_object: int
    clicks: List[tuple] = field(default_factory=list)
    generation: int = 0  # the worker generation that has it open; 0: none


@dataclass
class Waiter:
    """A click waiting for the GPU; the request's thread waits on done."""
    user_id: str
    job_id: str
    run: Callable[[], dict]
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict] = None
    error: Optional[Refused] = None


class Feed:
    """A job's progress for the window that watches it: frames with their masks (the oldest
    lose their mask beyond EVENT_BUFFER: the window reads the staged file then), messages, and
    its state."""

    def __init__(self):
        self.cv = threading.Condition()
        self.events: List[dict] = []
        self.seq = 0
        self.state: dict = {}

    def add(self, event: dict) -> None:
        with self.cv:
            self.seq += 1
            self.events.append(dict(event, seq=self.seq))
            for old in self.events[:-EVENT_BUFFER]:
                old['mask'] = None
            self.cv.notify_all()

    def update(self, **state) -> None:
        with self.cv:
            self.state.update(state)
            self.cv.notify_all()

    def take(self, after: int, wait_s: float) -> List[dict]:
        with self.cv:
            self.events = [event for event in self.events if event['seq'] > after]
            if not self.events and wait_s > 0 and self.state.get('status') not in FINAL:
                self.cv.wait(wait_s)
            return list(self.events)


class Running:
    """The propagation the worker runs now."""

    def __init__(self, job: dict, token: str, generation: int, total: int, journal: str):
        self.job = job
        self.token = token
        self.generation = generation
        self.total = max(1, total)
        self.done = 0
        self.journal = journal  # <results>/<job>.jsonl: a line per staged frame
        self.cancel = False
        self.last_write = 0.0


def snapshot_frames(snapshot: dict) -> int:
    """How many frames the run may read after its start frame."""
    from cutieplus.engine import policies
    bounds = snapshot.get('bounds')
    return 1 + len(policies.frames_after_start(int(snapshot['start_ti']), snapshot['direction'],
                                               len(snapshot['frame_names']),
                                               None if bounds is None else tuple(bounds)))


class GpuQueue:

    def __init__(self, service: Service):
        self.service = service
        self.db = service.db
        self.config = service.config
        self.link = WorkerLink(service, self.wake)
        self.cv = threading.Condition()
        self.clicks: 'OrderedDict[str, Deque[Waiter]]' = OrderedDict()  # user -> their clicks
        self.running: Optional[Running] = None
        self.feeds: Dict[str, Feed] = {}
        self.interactions: Dict[str, Interaction] = {}
        self.burst = 0
        self.queued_hint = True
        self.stopping = threading.Event()
        self.last_heartbeat = 0.0
        self.results_dir = path.join(self.config.state_dir, 'results')
        self.snapshots_dir = self.config.snapshots_dir
        for folder in (self.results_dir, self.snapshots_dir):
            os.makedirs(folder, mode=0o700, exist_ok=True)
        self.recover()
        service.suspension_hooks.append(self.on_user_suspended)

    # lifecycle
    def start(self) -> None:
        self.link.start()
        threading.Thread(target=self.loop, name='gpu-queue', daemon=True).start()

    def stop(self) -> None:
        self.stopping.set()
        self.wake()
        self.link.stop()

    def wake(self) -> None:
        with self.cv:
            self.cv.notify_all()

    def recover(self) -> None:
        """At start: the jobs a previous run of the service had running were interrupted (its
        worker connection, and the worker's state with it, are gone); what they staged stays as
        their result."""
        rows = self.db.query("SELECT * FROM gpu_jobs WHERE status IN ('running', "
                             "'cancel_requested')")
        for row in rows:
            if row['operation_type'] == 'propagate':
                self.finish(row, 'interrupted', 'service_restarted',
                            'the service restarted while it ran')
            else:
                self.finish_click(row['id'], 'interrupted', 'service_restarted',
                                  'the service restarted while it ran')

    # the loop
    def loop(self) -> None:
        while not self.stopping.is_set():
            with self.cv:
                if not self.has_work():
                    self.cv.wait(1.0)
                if self.stopping.is_set():
                    return
                idle = not self.has_work()
                waiter = None
                if idle:
                    pass
                elif self.clicks and (self.running is None or self.burst < MAX_BURST):
                    user, waiters = next(iter(self.clicks.items()))
                    waiter = waiters.popleft()
                    self.clicks.move_to_end(user)  # round-robin between users
                    if not waiters:
                        del self.clicks[user]
                    self.burst += 1
                else:
                    self.burst = 0
            if idle:
                self.heartbeat()  # outside the lock: a click may queue meanwhile
                continue
            try:
                if waiter is not None:
                    self.run_click(waiter)
                elif self.running is not None:
                    self.step()
                else:
                    self.claim()
            except Exception:  # the queue keeps going
                log.exception('the GPU queue failed a turn')
                time.sleep(0.5)

    def has_work(self) -> bool:
        if not self.link.ready:
            return False
        return bool(self.clicks) or self.running is not None or self.queued_hint

    def heartbeat(self) -> None:
        if not self.link.ready or time.monotonic() - self.last_heartbeat < HEARTBEAT_S:
            return
        self.last_heartbeat = time.monotonic()
        try:
            status = self.link.call('status', timeout=30)
        except (WorkerLost, CommandFailed):
            return
        with self.db.transaction() as cursor:
            cursor.execute('UPDATE gpu_state SET heartbeat_at = ?, memory_used_mb = ?, '
                           'memory_total_mb = ?, updated_at = ? WHERE gpu_id = ?',
                           (self.service.now(), status.get('memory_used_mb'),
                            status.get('memory_total_mb'), self.service.now(), GPU_ID))

    def set_gpu(self, state: str, job_id: Optional[str]) -> None:
        with self.db.transaction() as cursor:
            cursor.execute('UPDATE gpu_state SET state = ?, active_job_id = ?, heartbeat_at = ?, '
                           'updated_at = ? WHERE gpu_id = ?',
                           (state, job_id, self.service.now(), self.service.now(), GPU_ID))

    # propagation jobs
    def claim(self) -> None:
        """The next queued propagation, in one short transaction; its owner must still be
        allowed to run work."""
        now = self.service.now()
        token = new_id()
        generation = self.link.generation
        with self.db.transaction() as cursor:
            rows = cursor.execute(
                "SELECT j.*, u.status AS owner_status FROM gpu_jobs j JOIN users u ON "
                "u.id = j.owner_user_id WHERE j.status = 'queued' AND j.operation_type = "
                "'propagate' ORDER BY j.queue_sequence LIMIT 1").fetchall()
            if not rows:
                self.queued_hint = False
                return
            job = rows[0]
            if job['owner_status'] != 'active':
                cursor.execute("UPDATE gpu_jobs SET status = 'cancelled', finished_at = ?, "
                               "error_code = 'owner_inactive', error_summary = ? WHERE id = ? "
                               "AND status = 'queued'",
                               (now, f'its owner is {job["owner_status"]}', job['id']))
                return
            cursor.execute(
                "UPDATE gpu_jobs SET status = 'running', execution_token = ?, "
                'worker_instance_id = ?, worker_generation = ?, started_at = ?, heartbeat_at = ?, '
                "output_reference = ? WHERE id = ? AND status = 'queued'",
                (token, self.link.instance_id, generation, now, now,
                 path.join(self.config.jobs_dir, job['id'], token), job['id']))
            if self.db.conn.changes() != 1:
                return
            cursor.execute("UPDATE gpu_state SET state = 'busy', active_job_id = ?, "
                           'updated_at = ? WHERE gpu_id = ?', (job['id'], now, GPU_ID))
        job = self.job_row(job['id'])
        snapshot = self.load_snapshot(job)
        running = Running(job, token, generation, snapshot_frames(snapshot),
                          path.join(self.results_dir, f'{job["id"]}.jsonl'))
        feed = self.feeds.setdefault(job['id'], Feed())
        feed.update(status='running', position=0)
        try:
            started = self.link.call('prop_start', timeout=START_TIMEOUT_S, job_id=job['id'],
                                     token=token, snapshot=snapshot)
        except WorkerLost as exc:
            self.finish(job, 'interrupted', 'worker_lost', str(exc))
            return
        except CommandFailed as exc:
            self.finish(job, 'failed', exc.kind, str(exc))
            return
        with open(running.journal, 'w') as f:
            f.write(json.dumps({'token': token, 'generation': generation,
                                'permanent_anchors': started.get('permanent_anchors')}) + '\n')
        self.running = running

    def step(self) -> None:
        running = self.running
        job = running.job
        current = self.job_row(job['id'])
        if running.cancel or current['status'] == 'cancel_requested':
            self.release(running, 'cancelled', 'cancelled', 'cancelled')
            return
        try:
            reply = self.link.call('prop_step', timeout=STEP_TIMEOUT_S, job_id=job['id'],
                                   token=running.token)
        except WorkerLost as exc:
            self.running = None
            self.finish(job, 'interrupted', 'worker_lost', str(exc))
            return
        except CommandFailed as exc:
            # the worker released the run itself (out of memory: never continued)
            self.running = None
            if exc.kind != 'oom':
                self.release_quietly(running)
            self.finish(job, 'failed', exc.kind, str(exc))
            return
        if self.link.generation != running.generation:
            # a reply of a worker generation that is gone: never used
            self.running = None
            self.finish(job, 'interrupted', 'worker_replaced', 'the GPU worker was replaced')
            return
        feed = self.feeds.setdefault(job['id'], Feed())
        if reply.get('sha256'):
            running.done += 1
            with open(running.journal, 'a') as f:
                f.write(json.dumps({'name': reply['name'], 'sha256': reply['sha256']}) + '\n')
        mask = reply.get('mask')
        feed.add({'kind': reply['kind'], 'ti': reply['ti'], 'name': reply['name'],
                  'mask': mask if isinstance(mask, np.ndarray) else None,
                  'messages': reply.get('messages') or [], 'memory': reply.get('memory'),
                  'stop_reason': reply.get('stop_reason')})
        progress = min(1.0, (running.done + 1) / running.total)
        feed.update(progress=progress)
        if time.monotonic() - running.last_write > 1.0 or reply.get('finished'):
            running.last_write = time.monotonic()
            with self.db.transaction() as cursor:
                cursor.execute('UPDATE gpu_jobs SET progress = ?, heartbeat_at = ? WHERE id = ? '
                               'AND execution_token = ?', (progress, self.service.now(),
                                                           job['id'], running.token))
        if reply.get('finished'):
            self.release(running, 'succeeded', None, None, reply.get('stop_reason'))

    def release(self, running: Running, status: str, code: Optional[str],
                summary: Optional[str], stop_reason: Optional[str] = None) -> None:
        """The run ends: the worker frees it (and says so) before the job is final."""
        self.running = None
        try:
            self.link.call('prop_release', timeout=STEP_TIMEOUT_S, job_id=running.job['id'],
                           token=running.token)
        except WorkerLost as exc:
            self.finish(running.job, 'interrupted', 'worker_lost', str(exc))
            return
        except CommandFailed as exc:
            log.warning('releasing %s: %s', running.job['id'], exc)
        self.finish(running.job, status, code, summary, stop_reason)

    def release_quietly(self, running: Running) -> None:
        try:
            self.link.call('prop_release', timeout=STEP_TIMEOUT_S, job_id=running.job['id'],
                           token=running.token)
        except (WorkerLost, CommandFailed):
            pass

    def finish(self, job: dict, status: str, code: Optional[str], summary: Optional[str],
               stop_reason: Optional[str] = None) -> None:
        """A propagation's final status, with its manifest: the frames its token staged."""
        now = self.service.now()
        journal = path.join(self.results_dir, f'{job["id"]}.jsonl')
        frames, token = [], job['execution_token']
        if path.exists(journal):
            with open(journal) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            if lines:
                token = lines[0].get('token', token)
                frames = [line for line in lines[1:] if 'name' in line]
        manifest_path = path.join(self.results_dir, f'{job["id"]}.json')
        manifest = json.dumps({'job_id': job['id'], 'token': token, 'status': status,
                               'stop_reason': stop_reason, 'staging': job['output_reference'],
                               'revision': job['annotation_or_anchor_revision'],
                               'frames': frames}, sort_keys=True).encode('utf-8')
        with open(manifest_path, 'wb') as f:
            f.write(manifest)
        with self.db.transaction() as cursor:
            cursor.execute(
                'UPDATE gpu_jobs SET status = ?, finished_at = ?, error_code = ?, '
                'error_summary = ?, result_manifest_reference = ?, result_manifest_sha256 = ?, '
                "result_state = CASE WHEN ? > 0 THEN 'pending' ELSE 'none' END, "
                'progress = CASE WHEN ? THEN 1 ELSE progress END WHERE id = ? AND status NOT IN '
                f"({','.join('?' * len(FINAL))})",
                (status, now, code, summary, manifest_path, hashlib.sha256(manifest).hexdigest(),
                 len(frames), status == 'succeeded', job['id'], *FINAL))
            cursor.execute("UPDATE gpu_state SET state = CASE WHEN state = 'busy' THEN 'idle' "
                           'ELSE state END, active_job_id = NULL, updated_at = ? '
                           'WHERE gpu_id = ? AND active_job_id IS ?', (now, GPU_ID, job['id']))
            audit.record(cursor, now, f'job_{status}', audit.SUCCEEDED if status == 'succeeded'
                         else audit.FAILED, None, audit.SYSTEM, 'gpu_job', job['id'],
                         error_code=code, after={'frames': len(frames), 'stop': stop_reason})
        feed = self.feeds.setdefault(job['id'], Feed())
        feed.update(status=status, error=summary, stop_reason=stop_reason, frames=len(frames))
        self.queued_hint = True

    # clicks
    def run_click(self, waiter: Waiter) -> None:
        now = self.service.now()
        with self.db.transaction() as cursor:
            cursor.execute("UPDATE gpu_jobs SET status = 'running', started_at = ?, "
                           'worker_generation = ?, execution_token = ? WHERE id = ? AND '
                           "status = 'queued'", (now, self.link.generation, new_id(),
                                                 waiter.job_id))
            claimed = self.db.conn.changes() == 1
        if not claimed:
            waiter.error = Refused(CONFLICT, 'the click was cancelled')
            waiter.done.set()
            return
        try:
            waiter.result = waiter.run()
            self.finish_click(waiter.job_id, 'succeeded')
        except WorkerLost as exc:
            waiter.error = Refused(UNAVAILABLE, f'{exc}; the click was not applied: try again')
            self.finish_click(waiter.job_id, 'interrupted', 'worker_lost', str(exc))
        except CommandFailed as exc:
            waiter.error = Refused(INVALID if exc.kind == 'invalid' else UNAVAILABLE, str(exc),
                                   kind=exc.kind)
            self.finish_click(waiter.job_id, 'failed', exc.kind, str(exc))
        finally:
            waiter.done.set()

    def finish_click(self, job_id: str, status: str, code: Optional[str] = None,
                     summary: Optional[str] = None) -> None:
        with self.db.transaction() as cursor:
            cursor.execute('UPDATE gpu_jobs SET status = ?, finished_at = ?, error_code = ?, '
                           'error_summary = ?, progress = 1 WHERE id = ?',
                           (status, self.service.now(), code, summary, job_id))

    def open_on_worker(self, interaction: Interaction) -> None:
        """The interaction as the worker of this generation knows it: opened again, with its
        accepted clicks replayed, after a restart of the worker."""
        if interaction.generation == self.link.generation:
            return
        self.link.call('ritm_open', timeout=CLICK_TIMEOUT_S, interaction_id=interaction.id,
                       image_path=interaction.image_path, mask_png=interaction.mask_png,
                       num_objects=interaction.num_objects,
                       target_object=interaction.target_object, clicks=interaction.clicks)
        interaction.generation = self.link.generation

    def submit_click(self, service: Service, ctx: Context, interaction: Interaction,
                     operation_type: str, idempotency_key: str, run: Callable[[], dict]) -> dict:
        job_id = self.insert_job(ctx, operation_type, idempotency_key, interaction.workspace,
                                 f'interaction:{interaction.id}', '', 0,
                                 {'interaction': interaction.id}, queue=False)
        if job_id is None:
            raise Refused(CONFLICT, 'that click was sent already')
        waiter = Waiter(ctx.actor.user_id, job_id, run)
        with self.cv:
            self.clicks.setdefault(ctx.actor.user_id, deque()).append(waiter)
            self.cv.notify_all()
        if not waiter.done.wait(CLICK_TIMEOUT_S):
            with self.cv:
                waiters = self.clicks.get(ctx.actor.user_id)
                if waiters is not None and waiter in waiters:
                    waiters.remove(waiter)
            self.finish_click(job_id, 'cancelled', 'timeout', 'the GPU was busy too long')
            raise Refused(UNAVAILABLE, 'the GPU is busy (or its worker is restarting); try again')
        if waiter.error is not None:
            raise waiter.error
        return waiter.result

    # rows and snapshots
    def job_row(self, job_id: str) -> dict:
        return self.db.query_one('SELECT * FROM gpu_jobs WHERE id = ?', (job_id, ))

    def load_snapshot(self, job: dict) -> dict:
        with open(job['input_reference'], 'rb') as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != job['input_revision']:
            raise RuntimeError(f'the snapshot of {job["id"]} changed')
        return decode_message(data)

    def insert_job(self, ctx: Context, operation_type: str, idempotency_key: str, workspace: str,
                   input_reference: str, input_revision: str, revision: int, parameters: dict,
                   queue: bool = True, retry_of: Optional[str] = None) -> Optional[str]:
        """A new job of the caller (None when that idempotency key was used already)."""
        now = self.service.now()
        model = self.link.hello.get('model') or {}
        with self.db.transaction() as cursor:
            if cursor.execute('SELECT 1 FROM gpu_jobs WHERE owner_user_id = ? AND '
                              'idempotency_key = ?', (ctx.actor.user_id,
                                                      idempotency_key)).fetchall():
                return None
            sequence = cursor.execute('SELECT coalesce(max(queue_sequence), 0) + 1 AS n FROM '
                                      'gpu_jobs').fetchall()[0]['n']
            job_id = new_id()
            cursor.execute(
                'INSERT INTO gpu_jobs (id, owner_user_id, submitted_by_app_session_id, '
                'desktop_slot_id, operation_type, workspace_id, input_reference, input_revision, '
                'annotation_or_anchor_revision, model_identifier, model_version, parameters_json, '
                'status, queue_sequence, idempotency_key, submitted_at, retry_of_job_id) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                (job_id, ctx.actor.user_id, ctx.session['id'], ctx.origin.slot_id,
                 operation_type, workspace, input_reference, input_revision, revision,
                 'cutie+ritm', model.get('cutie_sha256', 'unknown'),
                 json.dumps(parameters, sort_keys=True), sequence, idempotency_key, now,
                 retry_of))
            if operation_type == 'propagate':
                audit.record(cursor, now, 'submit_job', audit.SUCCEEDED, ctx.actor.user_id,
                             target_type='gpu_job', target_id=job_id,
                             request_id=ctx.request_id, after={'retry_of': retry_of})
        if queue:
            self.queued_hint = True
            self.wake()
        return job_id

    def position(self, job: dict) -> Optional[int]:
        """Jobs ahead of a queued one, the running one included; None once it runs."""
        if job['status'] != 'queued':
            return None
        ahead = self.db.query_one(
            "SELECT count(*) AS n FROM gpu_jobs WHERE operation_type = 'propagate' AND "
            "(status IN ('running', 'cancel_requested') OR (status = 'queued' AND "
            'queue_sequence < ?))', (job['queue_sequence'], ))['n']
        return int(ahead)

    def on_user_suspended(self, cursor, user: dict, actor_id, now) -> None:
        """A suspended or removed user's queued jobs are cancelled, a running one asked to."""
        cursor.execute("UPDATE gpu_jobs SET status = 'cancelled', finished_at = ?, "
                       "cancelled_by = ?, error_code = 'owner_inactive', error_summary = ? "
                       "WHERE owner_user_id = ? AND status = 'queued'",
                       (now, actor_id, f'its owner was {user["status"]}', user['id']))
        cursor.execute("UPDATE gpu_jobs SET status = 'cancel_requested', cancel_requested_at = ?, "
                       "cancelled_by = ? WHERE owner_user_id = ? AND status = 'running' "
                       "AND operation_type = 'propagate'", (now, actor_id, user['id']))
        self.service.after_commit(self.wake)


# what the windows ask

def gpu_queue(service: Service) -> GpuQueue:
    if service.gpu is None:
        raise Refused(UNAVAILABLE, 'the GPU queue is not running')
    return service.gpu


def check_workspace(service: Service, workspace) -> str:
    if not isinstance(workspace, str) or not service.config.within_workspace(workspace):
        raise Refused(INVALID, 'the task must lie in the workspace')
    return path.normpath(workspace)


def check_snapshot(service: Service, snapshot) -> dict:
    """A propagation's input, checked field by field; nothing else is kept."""
    if not isinstance(snapshot, dict):
        raise Refused(INVALID, 'snapshot is an object')
    names, paths = snapshot.get('frame_names'), snapshot.get('frame_paths')
    if not isinstance(names, list) or not isinstance(paths, list) or not names or \
            len(names) != len(paths) or len(names) > MAX_FRAMES:
        raise Refused(INVALID, 'frame_names and frame_paths are lists of one entry per frame')
    if not all(isinstance(name, str) and len(name) <= 128 for name in names) or \
            not all(isinstance(item, str) and service.config.within_workspace(item)
                    for item in paths):
        raise Refused(INVALID, 'every frame lies in the workspace')
    length = len(names)
    start = validate.integer(snapshot.get('start_ti'), 'start_ti', 0, length - 1)
    if snapshot.get('direction') not in DIRECTIONS:
        raise Refused(INVALID, f'direction is one of {", ".join(DIRECTIONS)}')
    anchors = snapshot.get('anchors')
    if not isinstance(anchors, dict) or not all(
            isinstance(name, str) and kind in ('mask', 'empty') for name, kind in anchors.items()):
        raise Refused(INVALID, 'anchors maps frame names to mask or empty')
    anchor_masks = snapshot.get('anchor_masks')
    if not isinstance(anchor_masks, dict) or not all(
            anchors.get(name) == 'mask' and isinstance(data, bytes) and len(data) <=
            MAX_MASK_BYTES for name, data in anchor_masks.items()):
        raise Refused(INVALID, 'anchor_masks holds the PNG of each mask anchor')
    height = validate.integer(snapshot.get('height'), 'height', 1, 16384)
    width = validate.integer(snapshot.get('width'), 'width', 1, 16384)
    num_objects = validate.integer(snapshot.get('num_objects'), 'num_objects', 1, 255)
    seed = snapshot.get('seed_prob')
    if not isinstance(seed, np.ndarray) or seed.dtype.kind != 'f' or \
            seed.shape != (num_objects + 1, height, width) or seed.nbytes > MAX_SEED_BYTES:
        raise Refused(INVALID, 'seed_prob is (num_objects + 1) * height * width probabilities')
    bounds = snapshot.get('bounds')
    if bounds is not None:
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise Refused(INVALID, 'bounds are [first, last] frames')
        bounds = [validate.integer(bound, 'bounds', 0, length - 1) for bound in bounds]
        if not bounds[0] <= start <= bounds[1]:
            raise Refused(INVALID, 'the start frame lies in its bounds')
    resolution = snapshot.get('resolution')
    if not isinstance(resolution, dict):
        raise Refused(INVALID, 'resolution is an object')
    memory = snapshot.get('memory')
    if not isinstance(memory, dict):
        raise Refused(INVALID, 'memory is an object')
    return {'frame_names': names, 'frame_paths': [path.normpath(item) for item in paths],
            'start_ti': start, 'direction': snapshot['direction'], 'anchors': anchors,
            'anchor_masks': anchor_masks, 'height': height, 'width': width,
            'num_objects': num_objects, 'seed_prob': seed.astype(np.float16), 'bounds': bounds,
            'resolution': {'max_internal_size': validate.integer(
                resolution.get('max_internal_size'), 'max_internal_size', 64, 8192),
                           'upscale_to_internal_size': bool(
                               resolution.get('upscale_to_internal_size'))},
            'memory': {key: validate.integer(memory.get(key), key, 1, 1 << 20)
                       for key in MEMORY_KEYS}}


def key_of(idempotency_key) -> str:
    if not isinstance(idempotency_key, str) or not 8 <= len(idempotency_key) <= 64:
        raise Refused(INVALID, 'idempotency_key is text of 8 to 64 characters')
    return idempotency_key


def job_for(service: Service, ctx: Context, job_id) -> dict:
    """A job the caller may see and act on: their own; a supervisor's annotators'; any for an
    admin. Checked against its owner's role now."""
    validate.identifier(job_id, 'job_id')
    row = service.db.query_one('SELECT j.*, u.role AS owner_role, u.login_name AS owner_login '
                               'FROM gpu_jobs j JOIN users u ON u.id = j.owner_user_id '
                               'WHERE j.id = ?', (job_id, ))
    if row is None:
        raise Refused(NOT_FOUND, 'no such job')
    authz.check_job_action(ctx.actor, row['owner_user_id'], row['owner_role'])
    return row


def public_job(service: Service, ctx: Context, row: dict) -> dict:
    queue = gpu_queue(service)
    visible = authz.can_see_job_owner(ctx.actor, row['owner_user_id'], row.get('owner_role') or
                                      service.user_row(row['owner_user_id'])['role'])
    return {'id': row['id'], 'operation_type': row['operation_type'], 'status': row['status'],
            'mine': row['owner_user_id'] == ctx.actor.user_id,
            'owner': (row.get('owner_login') or service.user_row(row['owner_user_id'])[
                'login_name']) if visible else None,
            'workspace': row['workspace_id'] if visible else None,
            'position': queue.position(row), 'progress': row['progress'],
            'submitted_at': row['submitted_at'], 'started_at': row['started_at'],
            'finished_at': row['finished_at'], 'error_code': row['error_code'],
            'error_summary': row['error_summary'], 'result_state': row['result_state'],
            'retry_of': row['retry_of_job_id']}


@op()
def open_interaction(service: Service, ctx: Context, workspace, image_path, num_objects=1,
                     target_object=1, mask_png=None) -> dict:
    """A click interaction on one frame, starting from its mask (PNG; None: empty)."""
    queue = gpu_queue(service)
    workspace = check_workspace(service, workspace)
    if not isinstance(image_path, str) or not service.config.within_workspace(image_path):
        raise Refused(INVALID, 'the frame must lie in the workspace')
    if mask_png is not None and (not isinstance(mask_png, bytes) or len(mask_png) >
                                 MAX_MASK_BYTES):
        raise Refused(INVALID, 'mask_png is a PNG')
    interaction = Interaction(new_id(), ctx.session['id'], ctx.actor.user_id, workspace,
                              path.normpath(image_path), mask_png,
                              validate.integer(num_objects, 'num_objects', 1, 255),
                              validate.integer(target_object, 'target_object', 1, 255))
    queue.interactions[interaction.id] = interaction
    return {'interaction_id': interaction.id}


def own_interaction(service: Service, ctx: Context, interaction_id) -> Interaction:
    interaction = gpu_queue(service).interactions.get(interaction_id) \
        if isinstance(interaction_id, str) else None
    if interaction is None:
        raise Refused(NOT_FOUND, 'the interaction is not open (the service restarted?)')
    if interaction.session_id != ctx.session['id']:
        raise Refused(FORBIDDEN, 'not your interaction')
    return interaction


@op()
def click(service: Service, ctx: Context, interaction_id, x, y, is_negative,
          idempotency_key) -> dict:
    interaction = own_interaction(service, ctx, interaction_id)
    queue = gpu_queue(service)
    x = validate.integer(x, 'x', 0, 1 << 16)
    y = validate.integer(y, 'y', 0, 1 << 16)

    def run() -> dict:
        queue.open_on_worker(interaction)
        reply = queue.link.call('ritm_click', timeout=CLICK_TIMEOUT_S,
                                interaction_id=interaction.id, x=x, y=y,
                                is_negative=bool(is_negative))
        interaction.clicks.append((x, y, bool(is_negative)))
        return reply

    return queue.submit_click(service, ctx, interaction, 'click', key_of(idempotency_key), run)


@op()
def undo_click(service: Service, ctx: Context, interaction_id, idempotency_key) -> dict:
    interaction = own_interaction(service, ctx, interaction_id)
    queue = gpu_queue(service)

    def run() -> dict:
        queue.open_on_worker(interaction)
        reply = queue.link.call('ritm_undo', timeout=CLICK_TIMEOUT_S,
                                interaction_id=interaction.id)
        if interaction.clicks and reply.get('mask') is not None:
            interaction.clicks.pop()
        return reply

    return queue.submit_click(service, ctx, interaction, 'click_undo', key_of(idempotency_key),
                              run)


@op()
def close_interaction(service: Service, ctx: Context, interaction_id) -> None:
    interaction = own_interaction(service, ctx, interaction_id)
    queue = gpu_queue(service)
    queue.interactions.pop(interaction.id, None)
    if interaction.generation == queue.link.generation and queue.link.ready:
        try:  # its memory on the worker; a click waiting meanwhile is fine either way
            queue.link.call('ritm_close', timeout=CLICK_TIMEOUT_S, interaction_id=interaction.id)
        except (WorkerLost, CommandFailed):
            pass


@op()
def submit_propagation(service: Service, ctx: Context, idempotency_key, workspace, revision,
                       snapshot) -> dict:
    """Queues a propagation of the caller's task from an immutable snapshot of its inputs. The
    same idempotency key returns the job it made the first time."""
    queue = gpu_queue(service)
    key = key_of(idempotency_key)
    workspace = check_workspace(service, workspace)
    revision = validate.integer(revision, 'revision', 0, 1 << 62)
    existing = service.db.query_one('SELECT * FROM gpu_jobs WHERE owner_user_id = ? AND '
                                    'idempotency_key = ?', (ctx.actor.user_id, key))
    if existing is not None:
        return public_job(service, ctx, existing)
    snapshot = check_snapshot(service, snapshot)
    pending = service.db.query_one(
        "SELECT count(*) AS n FROM gpu_jobs WHERE owner_user_id = ? AND operation_type = "
        "'propagate' AND status IN ('queued', 'running', 'cancel_requested')",
        (ctx.actor.user_id, ))['n']
    limit = service.setting('pending_jobs_per_user')
    if pending >= limit:
        raise Refused(LIMIT, f'you have {pending} propagations waiting or running (at most '
                      f'{limit}); wait for one or cancel it')
    data = encode_message(snapshot)
    digest = hashlib.sha256(data).hexdigest()
    snapshot_path = path.join(queue.snapshots_dir, f'{digest}.bin')
    if not path.exists(snapshot_path):
        # named by content: two requests with the same inputs may write it at once
        temporary = f'{snapshot_path}.{new_id()}.tmp'
        with open(temporary, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, 0o400)  # immutable from now on
        os.replace(temporary, snapshot_path)
    queue.insert_job(ctx, 'propagate', key, workspace, snapshot_path, digest, revision,
                     {'frames': len(snapshot['frame_names']),
                      'start': snapshot['frame_names'][snapshot['start_ti']],
                      'direction': snapshot['direction']})
    # this request's job, or the one a request with the same key made meanwhile
    row = service.db.query_one('SELECT * FROM gpu_jobs WHERE owner_user_id = ? AND '
                               'idempotency_key = ?', (ctx.actor.user_id, key))
    queue.feeds.setdefault(row['id'], Feed()).update(status=row['status'])
    return public_job(service, ctx, row)


@op()
def job_events(service: Service, ctx: Context, job_id, after=0, wait_ms=0) -> dict:
    """A job's state, and the frames it produced since `after` (the owner's window)."""
    row = job_for(service, ctx, job_id)
    after = validate.integer(after, 'after', 0, 1 << 62)
    wait_ms = validate.integer(wait_ms, 'wait_ms', 0, 2000)
    events = []
    feed = gpu_queue(service).feeds.get(row['id'])
    if feed is not None and row['owner_user_id'] == ctx.actor.user_id:
        events = feed.take(after, wait_ms / 1000)
        row = service.db.query_one('SELECT * FROM gpu_jobs WHERE id = ?', (row['id'], ))
    return dict(public_job(service, ctx, row), events=events)


@op()
def cancel_job(service: Service, ctx: Context, job_id) -> dict:
    """A queued job ends at once; a running one once the worker has released it."""
    queue = gpu_queue(service)
    row = job_for(service, ctx, job_id)
    now = service.now()
    with service.db.transaction() as cursor:
        cursor.execute("UPDATE gpu_jobs SET status = 'cancelled', finished_at = ?, "
                       "cancelled_by = ?, cancel_requested_at = ? WHERE id = ? AND "
                       "status = 'queued'", (now, ctx.actor.user_id, now, row['id']))
        if changed_one_row(service):
            outcome = 'cancelled'
        else:
            cursor.execute("UPDATE gpu_jobs SET status = 'cancel_requested', "
                           'cancel_requested_at = ?, cancelled_by = ? WHERE id = ? AND '
                           "status = 'running'", (now, ctx.actor.user_id, row['id']))
            outcome = 'cancel_requested' if changed_one_row(service) else row['status']
        audit.record(cursor, now, 'cancel_job', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='gpu_job', target_id=row['id'], request_id=ctx.request_id,
                     after={'status': outcome})
    if outcome == 'cancelled':
        queue.feeds.setdefault(row['id'], Feed()).update(status='cancelled')
    running = queue.running
    if running is not None and running.job['id'] == row['id']:
        running.cancel = True
    queue.wake()
    return {'status': outcome}


def changed_one_row(service: Service) -> bool:
    return service.db.conn.changes() == 1


@op()
def retry_job(service: Service, ctx: Context, job_id, idempotency_key) -> dict:
    """A new job from the same snapshot as a failed, interrupted or cancelled propagation."""
    queue = gpu_queue(service)
    row = job_for(service, ctx, job_id)
    if row['operation_type'] != 'propagate' or row['status'] not in ('failed', 'interrupted',
                                                                       'cancelled'):
        raise Refused(INVALID, 'only a failed, interrupted or cancelled propagation is retried')
    if row['owner_user_id'] != ctx.actor.user_id:
        raise Refused(FORBIDDEN, 'a job is retried by its owner (a new job of theirs)')
    key = key_of(idempotency_key)
    queue.insert_job(ctx, 'propagate', key, row['workspace_id'], row['input_reference'],
                     row['input_revision'], row['annotation_or_anchor_revision'],
                     json.loads(row['parameters_json']), retry_of=row['id'])
    new = service.db.query_one('SELECT * FROM gpu_jobs WHERE owner_user_id = ? AND '
                               'idempotency_key = ?', (ctx.actor.user_id, key))
    return public_job(service, ctx, new)


@op()
def job_result(service: Service, ctx: Context, job_id) -> dict:
    """The manifest of a finished propagation: which staged masks its token wrote, with their
    SHA-256, and the task revision it started from."""
    row = job_for(service, ctx, job_id)
    if row['result_manifest_reference'] is None:
        raise Refused(INVALID, 'the job has no result')
    with open(row['result_manifest_reference'], 'rb') as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != row['result_manifest_sha256']:
        raise Refused(CONFLICT, 'the manifest does not match its record')
    return dict(json.loads(data), result_state=row['result_state'])


@op()
def record_result(service: Service, ctx: Context, job_id, execution_token, outcome) -> dict:
    """The owner's window says what it did with the result (applied to the task, held for review
    because the task changed, or discarded)."""
    row = job_for(service, ctx, job_id)
    if row['owner_user_id'] != ctx.actor.user_id:
        raise Refused(FORBIDDEN, 'the job\'s owner applies its result')
    if outcome not in RESULT_OUTCOMES:
        raise Refused(INVALID, f'outcome is one of {", ".join(RESULT_OUTCOMES)}')
    if execution_token != row['execution_token']:
        raise Refused(CONFLICT, 'that is not the token of this job\'s result')
    now = service.now()
    with service.db.transaction() as cursor:
        cursor.execute("UPDATE gpu_jobs SET result_state = ?, applied_at = CASE WHEN ? = "
                       "'applied' THEN ? ELSE applied_at END WHERE id = ? AND result_state IN "
                       "('pending', 'held_for_review')", (outcome, outcome, now, row['id']))
        audit.record(cursor, now, 'record_result', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='gpu_job', target_id=row['id'], request_id=ctx.request_id,
                     after={'outcome': outcome})
    return {'result_state': outcome}


@op()
def pending_results(service: Service, ctx: Context, workspace) -> list:
    """The caller's finished propagations of this task whose results are not applied yet."""
    workspace = check_workspace(service, workspace)
    rows = service.db.query(
        "SELECT * FROM gpu_jobs WHERE owner_user_id = ? AND workspace_id = ? AND result_state "
        "IN ('pending', 'held_for_review') AND operation_type = 'propagate' ORDER BY "
        'queue_sequence', (ctx.actor.user_id, workspace))
    return [public_job(service, ctx, row) for row in rows]


@op()
def gpu_status(service: Service, ctx: Context) -> dict:
    """The worker, the job it runs, the queue (owners shown as the caller may see them) and the
    caller's own recent jobs."""
    queue = gpu_queue(service)
    state = service.db.query_one('SELECT * FROM gpu_state WHERE gpu_id = ?', (GPU_ID, ))
    rows = service.db.query(
        "SELECT j.*, u.role AS owner_role, u.login_name AS owner_login FROM gpu_jobs j JOIN "
        "users u ON u.id = j.owner_user_id WHERE j.operation_type = 'propagate' AND j.status IN "
        "('queued', 'running', 'cancel_requested') ORDER BY j.queue_sequence")
    mine = service.db.query(
        "SELECT j.*, u.role AS owner_role, u.login_name AS owner_login FROM gpu_jobs j JOIN "
        "users u ON u.id = j.owner_user_id WHERE j.owner_user_id = ? AND j.operation_type = "
        "'propagate' ORDER BY j.queue_sequence DESC LIMIT 20", (ctx.actor.user_id, ))
    return {
        'worker': {'state': state['state'] if queue.link.ready or state['state'] != 'idle'
                   else 'offline', 'generation': state['worker_generation'],
                   'device': queue.link.hello.get('device_name'),
                   'memory_used_mb': state['memory_used_mb'],
                   'memory_total_mb': state['memory_total_mb'],
                   'heartbeat_at': state['heartbeat_at'], 'last_error': state['last_error']},
        'queue': [public_job(service, ctx, row) for row in rows],
        'mine': [public_job(service, ctx, row) for row in mine],
    }
