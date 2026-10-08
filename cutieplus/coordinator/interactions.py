"""
Click interactions of the shared service. The coordinator keeps the record of each one (task,
frame, image modality, target object, the committed mask it started from, the accepted clicks in
order); the GPU worker keeps its model state, at most one on the GPU. A click is inferred in the
worker, committed here (the frame becomes an anchor), and only then accepted into the record. When
the worker's state may differ from the record (a refused commit, a restarted worker), the worker
rebuilds the interaction from the record by replaying its accepted clicks.
"""
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from cutieplus.coordinator.tasks import TaskError, TaskService

log = logging.getLogger(__name__)


class InteractionError(Exception):
    pass


@dataclass
class Interaction:
    interaction_id: str
    session_id: str
    task_id: str
    lease_epoch: int
    frame: str
    modality: str
    image_path: str
    target_object: int
    num_objects: int
    initial: Optional[dict]  # the committed mask the clicks start from (mask source)
    clicks: List[Tuple[int, int, bool]] = field(default_factory=list)
    accepted_seq: int = 0
    generation: int = 0  # the worker generation that has it open; 0: none
    dirty: bool = False  # the worker may hold state the record does not
    lock: threading.Lock = field(default_factory=threading.Lock)


class InteractionManager:

    def __init__(self, tasks: TaskService, run_on_gpu: Callable, worker):
        self.tasks = tasks
        self.run_on_gpu = run_on_gpu  # (session_id, fn, label) -> fn's result, on the dispatcher
        self.worker = worker
        self.interactions: Dict[str, Interaction] = {}
        self.pending_close: Dict[str, int] = {}  # closed here, to close in the worker generation
        self.lock = threading.Lock()

    def open(self, session_id: str, task_id: str, epoch: int, frame: str, modality: str,
             target_object: int, num_objects: int = 1) -> dict:
        self.tasks.context(task_id, session_id, epoch)
        info = self.tasks.info_by_id(task_id)
        if frame not in info.name_to_ti:
            raise TaskError('invalid', f'{task_id} has no frame {frame}')
        if modality not in info.frame_dirs:
            raise TaskError('invalid', f'{task_id} has no modality {modality}')
        self.close_session(session_id)  # one interaction per window
        record = Interaction(uuid.uuid4().hex[:16], session_id, task_id, epoch, frame, modality,
                             info.image_path(frame, modality), int(target_object), int(num_objects),
                             self.tasks.mask_source(info, frame))
        with self.lock:
            self.interactions[record.interaction_id] = record
        return {'interaction_id': record.interaction_id}

    def get(self, interaction_id: str, session_id: str) -> Interaction:
        record = self.interactions.get(interaction_id)
        if record is None or record.session_id != session_id:
            raise InteractionError(f'interaction {interaction_id} is not open')
        return record

    def close(self, interaction_id: str, session_id: Optional[str] = None) -> None:
        with self.lock:
            record = self.interactions.get(interaction_id)
            if record is None or (session_id is not None and record.session_id != session_id):
                return
            del self.interactions[interaction_id]
            if record.generation:
                self.pending_close[interaction_id] = record.generation

    def close_session(self, session_id: str) -> None:
        for record in [r for r in self.interactions.values() if r.session_id == session_id]:
            self.close(record.interaction_id)

    def records(self) -> List[Interaction]:
        return list(self.interactions.values())

    # dispatcher thread ---------------------------------------------------------------------
    def _close_pending(self) -> None:
        with self.lock:
            pending, self.pending_close = self.pending_close, {}
        for interaction_id, generation in pending.items():
            if generation == self.worker.generation:
                self.worker.call('ritm_close', interaction_id=interaction_id)

    def _ensure_open(self, record: Interaction) -> None:
        self._close_pending()
        if record.generation == self.worker.generation and not record.dirty:
            return
        if record.generation == self.worker.generation:
            self.worker.call('ritm_close', interaction_id=record.interaction_id)
        self.worker.call('ritm_open', interaction_id=record.interaction_id,
                         image_path=record.image_path, mask=record.initial,
                         num_objects=record.num_objects, target_object=record.target_object,
                         clicks=[list(click) for click in record.clicks])
        record.generation = self.worker.generation
        record.dirty = False

    # operations ----------------------------------------------------------------------------
    def _apply(self, record: Interaction, op: str, worker_args: dict, request_id: Optional[str],
               accept: Callable[[], None]) -> dict:
        with record.lock:  # one operation of an interaction at a time
            stored = self.tasks.db.query_one('SELECT response FROM requests WHERE request_id = ?',
                                             (request_id, )) if request_id else None
            if stored is not None:
                from cutieplus.storage.db import loads
                response = loads(stored['response'])
                sha = response.get('mask_object')
                return dict(response, replayed=True,
                            mask=self.tasks.objects.mask(sha) if sha else None,
                            clicks=len(record.clicks))
            self.tasks.context(record.task_id, record.session_id, record.lease_epoch)

            def work():
                self._ensure_open(record)
                return self.worker.call(op, interaction_id=record.interaction_id, **worker_args)

            submitted = time.perf_counter()
            try:
                result = self.run_on_gpu(record.session_id, work, op)
            except BaseException:
                # the worker may have applied it anyway (a time-out while it ran): rebuild the
                # interaction from the accepted clicks next time
                record.dirty = True
                raise
            gpu_seconds = time.perf_counter() - submitted
            if result.get('mask') is None:
                return {'mask': None, 'clicks': len(record.clicks)}
            commit_start = time.perf_counter()
            try:
                commit = self.tasks.commit_frame(
                    record.task_id, record.session_id, record.lease_epoch, record.frame,
                    mask=result['mask'], anchor='edit', request_id=request_id,
                    op='click' if op == 'ritm_click' else 'undo_click')
            except Exception:
                record.dirty = True  # the worker has a click the task does not
                raise
            accept()
            record.accepted_seq += 1
            return dict(commit, mask=result['mask'], clicks=len(record.clicks),
                        accepted_seq=record.accepted_seq,
                        reconstruction=result.get('reconstruction'),
                        reconstruction_seconds=result.get('reconstruction_seconds'),
                        inference_seconds=result.get('inference_seconds'),
                        queue_seconds=result.get('queue_seconds'),
                        gpu_request_seconds=gpu_seconds,
                        commit_seconds=time.perf_counter() - commit_start)

    def click(self, interaction_id: str, session_id: str, x: int, y: int, is_negative: bool,
              request_id: Optional[str] = None) -> dict:
        record = self.get(interaction_id, session_id)
        click = (int(x), int(y), bool(is_negative))
        return self._apply(record, 'ritm_click', {'x': click[0], 'y': click[1],
                                                  'is_negative': click[2]}, request_id,
                           lambda: record.clicks.append(click))

    def undo(self, interaction_id: str, session_id: str, request_id: Optional[str] = None) -> dict:
        record = self.get(interaction_id, session_id)
        if len(record.clicks) <= 1:
            return {'mask': None, 'clicks': len(record.clicks)}
        return self._apply(record, 'ritm_undo', {}, request_id, lambda: record.clicks.pop())

    def replay(self, interaction_id: str, session_id: str) -> dict:
        """Rebuilds the interaction from its accepted clicks in the worker and returns the last
        mask, without writing anything (verification and measurements)."""
        record = self.get(interaction_id, session_id)
        with record.lock:
            def work():
                self._ensure_open(record)
                return self.worker.call('ritm_replay', interaction_id=record.interaction_id)

            return self.run_on_gpu(record.session_id, work, 'ritm_replay')
