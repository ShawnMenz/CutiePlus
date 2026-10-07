"""
The order in which the one GPU worker serves requests.

Interactive requests (clicks, click undo, reconstruction of an interaction) come first, round-robin
between sessions so that one annotator's clicks cannot hold back another's. Background work
(propagation and the permanent-memory initialization, one step at a time) runs in between: after
`max_interactive_burst` interactive requests in a row, or once a ready step has waited
`max_background_wait` seconds, the next step goes first. One propagation job holds the GPU
context; the others wait in order, and none holds GPU memory while it waits.
"""
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional


@dataclass
class SchedulerConfig:
    max_interactive_burst: int = 4
    max_background_wait: float = 0.25  # seconds
    max_interactive_queue: int = 16  # per session
    max_jobs: int = 32  # queued and running propagation jobs

    @classmethod
    def from_config(cls, section: Optional[dict]) -> 'SchedulerConfig':
        section = dict(section or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in section.items() if key in known})


class QueueFull(Exception):
    pass


@dataclass
class InteractiveItem:
    session_id: str
    run: Callable[[], Any]
    submitted_at: float
    label: str = ''


@dataclass
class JobEntry:
    job_id: str
    session_id: str
    state: str = 'ready'  # ready | busy | blocked
    ready_since: float = 0.0
    not_before: float = 0.0  # an admission retry waits until then


@dataclass
class Decision:
    kind: str  # 'interactive' | 'job'
    item: Optional[InteractiveItem] = None
    job_id: Optional[str] = None
    admission: bool = False  # the job has no GPU context yet
    waited: float = 0.0  # seconds since the request or step was ready


class Scheduler:

    def __init__(self, config: SchedulerConfig, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.clock = clock
        self.cond = threading.Condition()
        self.interactive: 'OrderedDict[str, Deque[InteractiveItem]]' = OrderedDict()
        self.jobs: 'OrderedDict[str, JobEntry]' = OrderedDict()  # in submission order
        self.active_job: Optional[str] = None  # holds the GPU propagation context
        self.burst = 0
        self.stopped = False
        self.history: Deque[tuple] = deque(maxlen=1000)  # (kind, session or job), for tests and status

    # interactive -------------------------------------------------------------------------
    def submit(self, session_id: str, run: Callable[[], Any], label: str = '') -> InteractiveItem:
        with self.cond:
            queue = self.interactive.setdefault(session_id, deque())
            if len(queue) >= self.config.max_interactive_queue:
                raise QueueFull(f'{len(queue)} requests of this session are waiting for the GPU')
            item = InteractiveItem(session_id, run, self.clock(), label)
            queue.append(item)
            self.cond.notify_all()
            return item

    def drop_session(self, session_id: str) -> List[InteractiveItem]:
        """Removes a session's waiting interactive requests (it expired); returns them."""
        with self.cond:
            return list(self.interactive.pop(session_id, ()))

    # background --------------------------------------------------------------------------
    def add_job(self, job_id: str, session_id: str) -> None:
        with self.cond:
            if len(self.jobs) >= self.config.max_jobs:
                raise QueueFull(f'{len(self.jobs)} propagation jobs are queued')
            self.jobs[job_id] = JobEntry(job_id, session_id, 'ready', self.clock())
            self.cond.notify_all()

    def job_ready(self, job_id: str, not_before: float = 0.0) -> None:
        with self.cond:
            entry = self.jobs.get(job_id)
            if entry is not None:
                entry.state = 'ready'
                entry.ready_since = self.clock()
                entry.not_before = not_before
                self.cond.notify_all()

    def job_blocked(self, job_id: str) -> None:
        """Waiting for something other than the GPU (its last result being committed)."""
        with self.cond:
            entry = self.jobs.get(job_id)
            if entry is not None:
                entry.state = 'blocked'

    def activate(self, job_id: str) -> None:
        with self.cond:
            self.active_job = job_id

    def release(self, job_id: str) -> None:
        """The job gave up its GPU context (finished, cancelled, failed, or waits again)."""
        with self.cond:
            if self.active_job == job_id:
                self.active_job = None
            self.cond.notify_all()

    def remove_job(self, job_id: str) -> None:
        with self.cond:
            self.jobs.pop(job_id, None)
            if self.active_job == job_id:
                self.active_job = None
            self.cond.notify_all()

    def requeue(self, job_id: str, not_before: float) -> None:
        """Back to waiting (no GPU context) until not_before, e.g. when memory is short."""
        with self.cond:
            if self.active_job == job_id:
                self.active_job = None
            entry = self.jobs.get(job_id)
            if entry is not None:
                entry.state = 'ready'
                entry.ready_since = self.clock()
                entry.not_before = not_before
            self.cond.notify_all()

    def queue_position(self, job_id: str) -> Optional[int]:
        """How many jobs are ahead of a job waiting for the GPU context; None when it holds it."""
        with self.cond:
            if self.active_job == job_id:
                return None
            ahead = 1 if self.active_job is not None else 0
            for other in self.jobs:
                if other == job_id:
                    return ahead
                if other != self.active_job:
                    ahead += 1
            return None

    # decisions ---------------------------------------------------------------------------
    def _background(self, now: float) -> Optional[JobEntry]:
        if self.active_job is not None:
            entry = self.jobs.get(self.active_job)
            ready = entry is not None and entry.state == 'ready' and entry.not_before <= now
            return entry if ready else None
        for entry in self.jobs.values():
            if entry.state == 'ready' and entry.not_before <= now:
                return entry
        return None

    def _choose(self) -> Optional[Decision]:
        now = self.clock()
        background = self._background(now)
        sessions = [session for session, queue in self.interactive.items() if queue]
        if sessions and background is not None:
            waited = now - background.ready_since
            if self.burst < self.config.max_interactive_burst and \
                    waited < self.config.max_background_wait:
                background = None
        if background is None and sessions:
            session = sessions[0]
            item = self.interactive[session].popleft()
            self.interactive.move_to_end(session)  # round-robin between sessions
            self.burst += 1
            self.history.append(('interactive', session, item.label))
            return Decision('interactive', item=item, waited=now - item.submitted_at)
        if background is not None:
            self.burst = 0
            background.state = 'busy'
            admission = self.active_job is None
            if admission:
                self.active_job = background.job_id
            self.history.append(('job', background.job_id, 'admission' if admission else 'step'))
            return Decision('job', job_id=background.job_id, admission=admission,
                            waited=now - background.ready_since)
        return None

    def next(self, timeout: Optional[float] = None) -> Optional[Decision]:
        """The next thing to run on the GPU, waiting up to timeout seconds for one."""
        deadline = None if timeout is None else self.clock() + timeout
        with self.cond:
            while not self.stopped:
                decision = self._choose()
                if decision is not None:
                    return decision
                wait = None if deadline is None else deadline - self.clock()
                if wait is not None and wait <= 0:
                    return None
                # a job waiting for memory becomes eligible later without a notification
                retry = [entry.not_before for entry in self.jobs.values()
                         if entry.state == 'ready' and entry.not_before > self.clock()]
                if retry:
                    soonest = min(retry) - self.clock()
                    wait = soonest if wait is None else min(wait, soonest)
                self.cond.wait(None if wait is None else max(0.001, wait))
            return None

    def stop(self) -> None:
        with self.cond:
            self.stopped = True
            self.cond.notify_all()

    def snapshot(self) -> dict:
        with self.cond:
            return {
                'interactive': {session: len(queue) for session, queue in self.interactive.items()
                                if queue},
                'jobs': [{'job_id': entry.job_id, 'session_id': entry.session_id,
                          'state': entry.state} for entry in self.jobs.values()],
                'active_job': self.active_job,
            }
