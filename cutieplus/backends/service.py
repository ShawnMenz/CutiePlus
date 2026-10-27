"""
The labeling window's backends on a host with the cooperation service. Neither loads a model: the
window's process never touches the GPU (interactive_demo.py hides it).

ServiceTaskBackend sends every inference to the shared GPU worker through the service:
- a click goes to the worker, which keeps the interaction's state; the window saves the mask as
  before (the frame becomes an anchor);
- a propagation is a job of the GPU queue, submitted with an immutable snapshot of its inputs (the
  frames' paths, the anchors and their masks, the start frame's probabilities, the memory settings)
  and the task's revision (metadata.json). The window shows the frames as the worker produces
  them; once the job ends (finished, paused, or interrupted), the window applies the masks the
  worker staged, each checked against the SHA-256 the service recorded, never over an anchor, and
  only if the task is still at the revision the job started from. Otherwise the result is held
  for review: the window asks whether to apply it anyway or discard it, and so does the next
  window that opens the task.

OfflineTaskBackend is the window when the service cannot be reached: everything but clicks and
propagation.
"""
import hashlib
import io
import logging
import shutil
import time
import uuid
from os import path
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from cutieplus.backends.base import (BackendError, ClickResult, GpuGauges, PropagatedFrame,
                                     PropagationProgress)
from cutieplus.backends.task_data import TaskDataBackend
from cutieplus.cooperation.client import ClientError, ServiceClient
from cutieplus.engine import policies
from cutieplus.engine.propagation import memory_config
from gui.interactive_utils import index_numpy_to_one_hot_torch
from gui.resource_manager import ResourceManager, write_atomically
from gui.task_state import TaskState

log = logging.getLogger(__name__)

OFFLINE = 'the CutiePlus service is not reachable: clicks and propagation are off'
FINAL = ('succeeded', 'failed', 'cancelled', 'interrupted')
POLL_MS = 150
CLICK_TIMEOUT_S = 75.0
CANCEL_WAIT_S = 60.0
GAUGE_EVERY_S = 5.0


def png_of(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


class OfflineTaskBackend(TaskDataBackend):
    """The task's files without any inference."""
    device = 'cpu'
    has_local_memory = False  # no model memory in the window: its reset buttons are hidden

    def __init__(self, cfg: DictConfig, res_man: ResourceManager, task: Optional[TaskState],
                 reason: str = OFFLINE):
        super().__init__(cfg, res_man, task)
        self.reason = reason
        self.num_objects = int(cfg['num_objects'])

    def memory_settings(self) -> Tuple[int, int, int, int]:
        """As the memory manager reads the configuration (InferenceCore)."""
        long_term = self.cfg.long_term
        return (int(long_term['min_mem_frames']) - 1, int(long_term['max_mem_frames']) - 1,
                int(long_term['max_num_tokens']), int(self.cfg['mem_every']))

    def update_config(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    # no inference
    def open_interaction(self, ti, modality, image, prev_prob, target_object) -> None:
        raise BackendError(self.reason)

    def add_click(self, x, y, is_negative) -> ClickResult:
        raise BackendError(self.reason)

    def undo_click(self) -> Optional[ClickResult]:
        return None

    def interaction_clicks(self) -> int:
        return 0

    def close_interaction(self) -> None:
        pass

    def start_propagation(self, start_ti, direction, seed_prob, modality=None) -> None:
        raise BackendError(self.reason)

    def advance_propagation(self) -> PropagationProgress:
        return PropagationProgress('finished')

    def cancel_propagation(self) -> None:
        pass

    def release_propagation(self) -> None:
        pass

    def memory_gauges(self):
        return None

    def gpu_gauges(self) -> GpuGauges:
        return GpuGauges()

    def clear_memory(self, permanent: bool) -> None:
        pass

    def commit_to_permanent_memory(self, image, prob):
        raise BackendError('there is no permanent memory in this window')

    def queue_state(self) -> Optional[dict]:
        return None

    def pending_results(self) -> List[dict]:
        return []

    def close(self) -> None:
        pass


class ServiceTaskBackend(OfflineTaskBackend):

    def __init__(self, cfg: DictConfig, res_man: ResourceManager, task: Optional[TaskState],
                 client: ServiceClient):
        if task is None:
            raise BackendError('the cooperation service works on tasks')
        super().__init__(cfg, res_man, task)
        self.client = client.copy()  # the window's own connection
        self.workspace = path.normpath(res_man.workspace)
        self.max_internal_size = int(cfg['max_internal_size'])
        self.upscale_to_internal_size = False
        self.configure_task_resolution()
        self.interaction_id: Optional[str] = None
        self.interaction_ti: Optional[int] = None
        self.clicks = 0
        self.job: Optional[dict] = None  # the propagation this window watches
        self.after = 0  # the last event of it shown
        self.last_memory = None
        self.gauges = GpuGauges()
        self.gauges_at = 0.0

    def call(self, op: str, timeout: Optional[float] = None, **args):
        try:
            return self.client.call(op, timeout=timeout, **args)
        except ClientError as exc:
            raise BackendError(str(exc)) from None

    def configure_task_resolution(self) -> None:
        names = policies.target_area_frames(self.task.anchors, self.task.prefill)
        name_to_ti = self.res_man.name_to_ti
        area = policies.median_target_area(
            names, lambda name: self.res_man.get_mask(name_to_ti[name])
            if name in name_to_ti else None)
        resolution = policies.choose_resolution(area, self.res_man.h, self.res_man.w,
                                                self.max_internal_size)
        self.max_internal_size = resolution.max_internal_size
        self.upscale_to_internal_size = bool(resolution.small_target)
        self.resolution_message = resolution.message()

    # clicks
    def open_interaction(self, ti: int, modality: str, image: np.ndarray, prev_prob: torch.Tensor,
                         target_object: int) -> None:
        self.close_interaction()
        mask = torch.argmax(prev_prob, dim=0).to('cpu').numpy().astype(np.uint8)
        reply = self.call('open_interaction', workspace=self.workspace,
                          image_path=self.res_man.image_path(ti, modality),
                          num_objects=self.num_objects, target_object=int(target_object),
                          mask_png=png_of(mask))
        self.interaction_id = reply['interaction_id']
        self.interaction_ti = ti
        self.clicks = 0

    def _click_result(self, reply: dict) -> ClickResult:
        update = self.commit_frame(self.interaction_ti, reply['mask'], 'edit')
        self.clicks = reply['clicks']
        prob = torch.from_numpy(np.asarray(reply['prob'], np.float32))
        return ClickResult(update, prob, reply['clicks'])

    def add_click(self, x: int, y: int, is_negative: bool) -> ClickResult:
        if self.interaction_id is None:
            raise BackendError('no interaction is open')
        reply = self.call('click', timeout=CLICK_TIMEOUT_S, interaction_id=self.interaction_id,
                          x=int(x), y=int(y), is_negative=bool(is_negative),
                          idempotency_key=uuid.uuid4().hex)
        return self._click_result(reply)

    def undo_click(self) -> Optional[ClickResult]:
        if self.interaction_id is None:
            return None
        reply = self.call('undo_click', timeout=CLICK_TIMEOUT_S,
                          interaction_id=self.interaction_id, idempotency_key=uuid.uuid4().hex)
        if reply.get('mask') is None:
            return None
        return self._click_result(reply)

    def interaction_clicks(self) -> int:
        return self.clicks if self.interaction_id is not None else 0

    def close_interaction(self) -> None:
        if self.interaction_id is not None:
            try:
                self.client.call('close_interaction', interaction_id=self.interaction_id)
            except ClientError:
                pass  # gone with the service, or with the session
        self.interaction_id = self.interaction_ti = None
        self.clicks = 0

    # propagation
    def snapshot(self, start_ti: int, direction: str, seed_prob: torch.Tensor,
                 modality: Optional[str]) -> dict:
        """The run's inputs, as they are now: what the job will use, whatever happens to the
        task meanwhile."""
        names = list(self.res_man.names)
        anchors = {name: entry['kind'] for name, entry in self.task.anchors.items()}
        anchor_masks = {}
        for name, kind in anchors.items():
            mask_file = path.join(self.res_man.mask_dir, f'{name}.png')
            if kind == 'mask':
                if not path.exists(mask_file):
                    raise BackendError(f'Anchor {name} has no mask file; remove the anchor or '
                                       'import its mask.')
                with open(mask_file, 'rb') as f:
                    anchor_masks[name] = f.read()
        bounds = policies.propagation_bounds(self.task.intervals(len(names))[0], start_ti)
        return {'frame_names': names,
                'frame_paths': [self.res_man.image_path(ti, modality) for ti in range(len(names))],
                'start_ti': int(start_ti), 'direction': direction, 'anchors': anchors,
                'anchor_masks': anchor_masks, 'height': int(self.res_man.h),
                'width': int(self.res_man.w), 'num_objects': self.num_objects,
                'seed_prob': seed_prob.detach().to('cpu').numpy().astype(np.float16),
                'bounds': None if bounds is None else list(bounds),
                'resolution': {'max_internal_size': self.max_internal_size,
                               'upscale_to_internal_size': self.upscale_to_internal_size},
                'memory': memory_config(self.cfg)}

    def start_propagation(self, start_ti: int, direction: str, seed_prob: torch.Tensor,
                          modality: Optional[str] = None) -> None:
        self.release_propagation()
        self.res_man.save_queue.join()  # the anchors' masks as saved
        snapshot = self.snapshot(start_ti, direction, seed_prob, modality)
        self.job = self.call('submit_propagation', idempotency_key=uuid.uuid4().hex,
                             workspace=self.workspace, revision=self.task.revision(),
                             snapshot=snapshot)
        self.after = 0

    def queue_state(self) -> Optional[dict]:
        """The watched job's status and position (None without one)."""
        if self.job is None:
            return None
        if self.job['status'] not in FINAL:
            reply = self.call('job_events', job_id=self.job['id'], after=self.after)
            self.job = {key: value for key, value in reply.items() if key != 'events'}
        return self.job

    def _frame(self, event: dict) -> PropagatedFrame:
        mask = event['mask']
        if mask is None and event['kind'] == 'frame':  # beyond the service's buffer: read it
            result_dir = path.join(self.client_staging(), f'{event["name"]}.png')
            mask = np.array(Image.open(result_dir))
        prob = None if mask is None else index_numpy_to_one_hot_torch(mask, self.num_objects + 1)
        return PropagatedFrame(event['ti'], event['kind'], mask, prob)

    def client_staging(self) -> str:
        if not hasattr(self, '_staging') or self._staging[0] != self.job['id']:
            row = self.call('job_result', job_id=self.job['id'])
            self._staging = (self.job['id'], row['staging'])
        return self._staging[1]

    def advance_propagation(self) -> PropagationProgress:
        if self.job is None:
            return PropagationProgress('finished')
        reply = self.call('job_events', timeout=30, job_id=self.job['id'], after=self.after,
                          wait_ms=POLL_MS)
        self.job = {key: value for key, value in reply.items() if key != 'events'}
        progress = PropagationProgress('running')
        for event in reply['events']:
            self.after = event['seq']
            progress.messages += event.get('messages') or []
            if event.get('memory'):
                self.last_memory = event['memory']
            if event['kind'] == 'stop':
                progress.stopped_at = event['ti']
            elif event.get('mask') is not None or event['kind'] == 'frame':
                progress.frames.append(self._frame(event))
        if reply['status'] in FINAL and not reply['events']:
            progress.state = 'failed' if reply['status'] in ('failed', 'interrupted') else \
                'finished'
            progress.stop_reason = reply.get('error_summary') or reply['status']
            progress.messages += self.finish_job()
        return progress

    def finish_job(self) -> List[str]:
        """The ended job's result, applied to the task (or held); messages for the window."""
        job, self.job = self.job, None
        if job is None or job.get('result_state') not in ('pending', 'held_for_review'):
            return [] if job is None or job['status'] == 'succeeded' else [
                f'Propagation {job["status"]}: {job.get("error_summary") or "no frames"}.']
        outcome, count = self.apply_result(job['id'])
        if outcome == 'applied':
            return [f'{count} propagated masks saved.'] + (
                [] if job['status'] == 'succeeded' else
                [f'Propagation {job["status"]}: the frames done before were kept.'])
        return ['The task changed while the propagation waited or ran: its result is held. '
                'It is offered for review when the task is opened again.']

    def apply_result(self, job_id: str, force: bool = False) -> Tuple[str, int]:
        """('applied' | 'held_for_review', masks written). force: apply even though the task
        changed since the job started (the user reviewed it); anchors are never overwritten."""
        result = self.call('job_result', job_id=job_id)
        if not force and self.task.revision() != result['revision']:
            self.call('record_result', job_id=job_id, execution_token=result['token'],
                      outcome='held_for_review')
            return 'held_for_review', 0
        self.res_man.save_queue.join()
        written = 0
        for frame in result['frames']:
            name = frame['name']
            if name in self.task.anchors or name not in self.res_man.name_to_ti:
                continue  # an anchor is a human decision: never replaced
            staged = path.join(result['staging'], f'{name}.png')
            with open(staged, 'rb') as f:
                data = f.read()
            if hashlib.sha256(data).hexdigest() != frame['sha256']:
                raise BackendError(f'the staged mask of {name} is not the one the job wrote')

            def write(temporary: str, data=data) -> None:
                with open(temporary, 'wb') as f:
                    f.write(data)

            write_atomically(path.join(self.res_man.mask_dir, f'{name}.png'), write)
            self.res_man.invalidate(self.res_man.name_to_ti[name])
            written += 1
        if written:
            self.task.touch(changes_masks=True)
        self.call('record_result', job_id=job_id, execution_token=result['token'],
                  outcome='applied')
        shutil.rmtree(path.dirname(result['staging']), ignore_errors=True)
        return 'applied', written

    def discard_result(self, job_id: str) -> None:
        result = self.call('job_result', job_id=job_id)
        self.call('record_result', job_id=job_id, execution_token=result['token'],
                  outcome='discarded')
        shutil.rmtree(path.dirname(result['staging']), ignore_errors=True)

    def pending_results(self) -> List[dict]:
        """Results of this user's earlier jobs on the task that were never applied."""
        try:
            return self.client.call('pending_results', workspace=self.workspace)
        except ClientError:
            return []

    def cancel_propagation(self) -> None:
        """Pause: the job ends once the worker has let go of it; the frames done are kept, as
        when the window ran the models itself."""
        if self.job is None:
            return
        if self.job['status'] not in FINAL:
            try:
                self.call('cancel_job', job_id=self.job['id'])
            except BackendError as exc:
                log.warning('cancelling %s: %s', self.job['id'], exc)
            deadline = time.monotonic() + CANCEL_WAIT_S
            while time.monotonic() < deadline:
                state = self.queue_state()
                if state is None or state['status'] in FINAL:
                    break
                time.sleep(0.1)
        if self.job is not None and self.job['status'] in FINAL:
            self.finish_job()

    def release_propagation(self) -> None:
        if self.job is not None and self.job['status'] in FINAL:
            self.finish_job()

    # gauges
    def memory_gauges(self):
        return self.last_memory

    def gpu_gauges(self) -> GpuGauges:
        if time.monotonic() - self.gauges_at > GAUGE_EVERY_S:
            self.gauges_at = time.monotonic()
            try:
                worker = self.client.call('gpu_status')['worker']
                if worker.get('memory_total_mb'):
                    self.gauges = GpuGauges(worker['memory_used_mb'] / 1024,
                                            worker['memory_total_mb'] / 1024)
            except ClientError:
                self.gauges = GpuGauges()
        return self.gauges

    def close(self) -> None:
        """A queued or running propagation goes on; its result waits for the next window."""
        self.close_interaction()
        self.client.close()
