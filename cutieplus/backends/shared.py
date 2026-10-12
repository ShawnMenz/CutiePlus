"""
The labeling window in shared mode: every inference runs in the service's GPU worker and every
change of the task's official data is committed by the coordinator; this process only reads the
task's files (the coordinator keeps them current) and never initializes CUDA.

The window keeps a mirror of the task's anchors (TaskState) updated from the coordinator's
answers. A lease lost to a restart of the coordinator is taken again when the task has not changed
since (same content revision); otherwise the window is told to open the task again.
"""
import logging
import time
from os import path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from cutieplus.backends.base import (BackendError, BinaryExport, ClickResult, FrameUpdate,
                                     GpuGauges, Interval, LeaseLost, PropagatedFrame,
                                     PropagationProgress)
from cutieplus.client import ClientError, ServiceClient, ServiceUnavailable, new_request_id
from cutieplus.engine.propagation import memory_config
from gui.edit_history import FrameState
from gui.interactive_utils import index_numpy_to_one_hot_torch
from gui.resource_manager import ResourceManager
from gui.task_state import (ANCHORS_FILE, PREFILL_FILE, TaskLocked, TaskState, _read_json,
                            saved_description)

log = logging.getLogger(__name__)

GPU_GAUGE_SECONDS = 5.0


class SharedTaskBackend:
    kind = 'shared'
    device = 'cpu'

    @classmethod
    def factory(cls, client: ServiceClient) -> Callable:
        """MainController's backend_factory: takes the task's lease (TaskLocked when another
        window or program has the task)."""

        def make(cfg, res_man: ResourceManager, task: Optional[TaskState]) -> 'SharedTaskBackend':
            if task is None:
                raise BackendError('the shared service works on tasks (a folder with task.json)')
            workspace = path.abspath(res_man.workspace)
            try:
                reply = client.call('open_task', workspace=workspace)
            except ClientError as exc:
                if exc.code == 'locked':
                    raise TaskLocked(workspace, exc.details.get('holder') or {}) from None
                raise BackendError(str(exc)) from None
            return cls(client, cfg, res_man, task, reply)

        return make

    def __init__(self, client: ServiceClient, cfg, res_man: ResourceManager, task: TaskState,
                 open_reply: dict):
        self.client = client
        self.cfg = cfg
        self.res_man = res_man
        self.task = task
        self.workspace = path.abspath(res_man.workspace)
        self.task_id = open_reply['task_id']
        self.epoch = open_reply['lease_epoch']
        self.content_revision = open_reply['content_revision']
        self.resolution_message = open_reply['resolution']['message']
        self.num_objects = int(cfg['num_objects'])
        self.memory = memory_config(cfg)
        self.interaction_id: Optional[str] = None
        self.interaction_ti: Optional[int] = None
        self.clicks = 0
        self.job_id: Optional[str] = None
        self.job_next = 0
        self.job_messages = 0
        self.job_done = True
        self.gauges: Tuple[float, GpuGauges] = (0.0, GpuGauges())
        res_man.read_only = True  # the coordinator writes the task's files
        self._reload_task()

    def _reload_task(self) -> None:
        """The mirror from the files the coordinator has just brought up to date."""
        self.task.anchors = _read_json(path.join(self.workspace, ANCHORS_FILE), {}) or {}
        self.task.prefill = _read_json(path.join(self.workspace, PREFILL_FILE), {}) or {}
        self.task.description = saved_description(self.workspace) or self.task.original_description
        self.res_man.get_mask.clear()

    # requests ---------------------------------------------------------------------------------
    def _call(self, op: str, retries: int = 1, **args):
        """A task request; a lost lease is taken again once when nothing changed meanwhile; a
        restarted GPU worker gets the same request again (it carries the same request id)."""
        for attempt in range(retries + 1):
            try:
                return self.client.call(op, task_id=self.task_id, lease_epoch=self.epoch, **args)
            except ServiceUnavailable as exc:
                raise BackendError(str(exc)) from None
            except ClientError as exc:
                if exc.code == 'lease_lost' and attempt < retries:
                    self._reopen()
                    continue
                if exc.code == 'worker_restarted' and attempt < retries:
                    continue
                if exc.code == 'lease_lost':
                    raise LeaseLost(str(exc)) from None
                raise BackendError(str(exc)) from None

    def _reopen(self) -> None:
        try:
            reply = self.client.call('open_task', workspace=self.workspace)
        except ClientError as exc:
            raise LeaseLost(f'the task could not be taken again: {exc}') from None
        if reply['content_revision'] != self.content_revision:
            raise LeaseLost('the task changed while this window did not hold it; open it again')
        self.epoch = reply['lease_epoch']

    def _apply(self, ti: int, reply: dict) -> Optional[dict]:
        """Mirrors a committed change of a frame."""
        name = self.res_man.names[ti]
        anchor = reply.get('anchor')
        if anchor is None:
            self.task.anchors.pop(name, None)
        else:
            self.task.anchors[name] = anchor
        self.res_man.invalidate(ti)
        if reply.get('content_revision') is not None:
            self.content_revision = reply['content_revision']
        if reply.get('views_written') is False:
            log.warning('%s: saved, but its files were not updated yet', name)
        return anchor

    def _prob(self, mask: np.ndarray) -> torch.Tensor:
        return index_numpy_to_one_hot_torch(mask, self.num_objects + 1)

    # clicks -----------------------------------------------------------------------------------
    def open_interaction(self, ti: int, modality: str, image: np.ndarray, prev_prob: torch.Tensor,
                         target_object: int) -> None:
        # the coordinator starts from the frame's committed mask, which is what the window shows
        self.close_interaction()
        reply = self._call('open_interaction', frame=self.res_man.names[ti], modality=modality,
                           target_object=int(target_object))
        self.interaction_id = reply['interaction_id']
        self.interaction_ti = ti
        self.clicks = 0

    def _click_result(self, reply: dict) -> ClickResult:
        mask = reply['mask']
        anchor = self._apply(self.interaction_ti, reply)
        self.clicks = reply['clicks']
        update = FrameUpdate(self.interaction_ti, mask, anchor, reply.get('content_revision'))
        return ClickResult(update, self._prob(mask), self.clicks, reply.get('reconstruction'))

    def add_click(self, x: int, y: int, is_negative: bool) -> ClickResult:
        reply = self._call('click', interaction_id=self.interaction_id, x=int(x), y=int(y),
                           is_negative=bool(is_negative), request_id=new_request_id())
        return self._click_result(reply)

    def undo_click(self) -> Optional[ClickResult]:
        if self.interaction_id is None:
            return None
        reply = self._call('undo_click', interaction_id=self.interaction_id,
                           request_id=new_request_id())
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
                pass
        self.interaction_id = self.interaction_ti = None
        self.clicks = 0

    # propagation ------------------------------------------------------------------------------
    def start_propagation(self, start_ti: int, direction: str, seed_prob: torch.Tensor,
                          modality: Optional[str] = None) -> None:
        # the start frame's committed mask is the seed (the window shows nothing uncommitted)
        reply = self._call('start_propagation', start_frame=self.res_man.names[start_ti],
                           direction=direction, memory=self.memory, modality=modality,
                           request_id=new_request_id())
        self.job_id = reply['job_id']
        self.job_next = self.job_messages = 0
        self.job_done = False

    def advance_propagation(self) -> PropagationProgress:
        if self.job_id is None:
            return PropagationProgress('finished')
        try:
            reply = self.client.call('poll_propagation', job_id=self.job_id, since=self.job_next,
                                     messages_since=self.job_messages)
        except ClientError as exc:
            raise BackendError(str(exc)) from None
        self.job_next = reply['next']
        self.job_messages = reply['next_message']
        for ti in reply['frames']:
            self.res_man.invalidate(ti)
        state = reply['state']
        progress = PropagationProgress(
            {'queued': 'queued', 'waiting_memory': 'queued', 'running': 'running',
             'committing': 'running'}.get(state, state),
            messages=list(reply['messages']), stop_reason=reply['stop_reason'],
            stopped_at=reply['stopped_at'], queue_position=reply['queue_position'],
            masks_written=reply['next'])
        if reply['latest_ti'] is not None:
            progress.frames.append(PropagatedFrame(reply['latest_ti'], 'frame', reply['latest_mask']))
        if reply.get('content_revision') is not None:
            self.content_revision = reply['content_revision']
        if progress.done:
            self.job_done = True
            if state == 'failed' and reply.get('error'):
                progress.stop_reason = reply['error']
        return progress

    def cancel_propagation(self) -> None:
        if self.job_id is not None and not self.job_done:
            try:
                self.client.call('cancel_propagation', job_id=self.job_id)
                # no frame is committed after the cancellation; wait for the job to stop
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    reply = self.client.call('poll_propagation', job_id=self.job_id,
                                             since=self.job_next, messages_since=self.job_messages)
                    for ti in reply['frames']:
                        self.res_man.invalidate(ti)
                    self.job_next = reply['next']
                    if reply.get('content_revision') is not None:
                        self.content_revision = reply['content_revision']
                    if reply['state'] in ('finished', 'cancelled', 'failed'):
                        break
                    time.sleep(0.05)
            except ClientError as exc:
                log.warning('cancelling the propagation: %s', exc)
        self.job_done = True

    def release_propagation(self) -> None:
        pass

    # official frame data ---------------------------------------------------------------------
    def commit_frame(self, ti: int, mask: np.ndarray, anchor: Optional[str]) -> FrameUpdate:
        reply = self._call('commit_frame', frame=self.res_man.names[ti],
                           mask=np.ascontiguousarray(mask, dtype=np.uint8), anchor=anchor,
                           request_id=new_request_id())
        return FrameUpdate(ti, mask, self._apply(ti, reply), reply.get('content_revision'))

    def remove_anchor(self, ti: int) -> FrameUpdate:
        reply = self._call('remove_anchor', frame=self.res_man.names[ti],
                           request_id=new_request_id())
        self._apply(ti, reply)
        return FrameUpdate(ti, self.res_man.get_mask(ti), None, reply.get('content_revision'))

    def restore_frame(self, ti: int, state: FrameState) -> FrameUpdate:
        reply = self._call('restore_frame', frame=self.res_man.names[ti], mask_png=state.mask,
                           anchor_entry=state.anchor, request_id=new_request_id())
        anchor = self._apply(ti, reply)
        return FrameUpdate(ti, self.res_man.get_mask(ti), anchor, reply.get('content_revision'))

    def frame_state(self, ti: int) -> FrameState:
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        mask = None
        if path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                mask = f.read()
        anchor = self.task.anchor(name)
        return FrameState(mask, dict(anchor) if anchor is not None else None)

    def wait_for_saves(self) -> None:
        pass  # every change is confirmed before its request returns

    # task data --------------------------------------------------------------------------------
    def set_intervals(self, intervals: List[Interval],
                      descriptions: Optional[Dict[Interval, str]] = None) -> None:
        # [start, end], with the description of an added interval as a third element
        entries = [list(i) + ([descriptions[i]] if descriptions and descriptions.get(i) else [])
                   for i in intervals]
        reply = self._call('set_intervals', intervals=entries, request_id=new_request_id())
        self.content_revision = reply['content_revision']

    def set_description(self, text: str) -> None:
        reply = self._call('set_description', text=text, request_id=new_request_id())
        self.content_revision = reply['content_revision']
        self.task.description = text

    def set_attention(self, flagged: bool) -> None:
        reply = self._call('set_attention', flagged=bool(flagged), request_id=new_request_id())
        self.task.attention = reply['attention']

    def export_binary(self, target_objects: List[int],
                      progress: Callable[[float], None]) -> Optional[BinaryExport]:
        reply = self._call('submit_for_review', target_objects=list(target_objects),
                           request_id=new_request_id())
        progress(1.0)
        return BinaryExport(reply['folder'], [tuple(i) for i in reply['intervals']])

    def reset_all_masks(self, export_dirs: List[str]) -> Tuple[int, int]:
        reply = self._call('reset_all_masks', export_dirs=list(export_dirs),
                           request_id=new_request_id())
        self.content_revision = reply['content_revision']
        self.task.anchors.clear()
        self.task.prefill.clear()
        self.res_man.get_mask.clear()
        return reply['removed'], reply['removed_exports']

    # the external export command runs in the window; the coordinator records and verifies it
    def begin_export(self) -> str:
        return self._call('begin_export')['export_id']

    def finish_export(self, export_id: str, ok: bool) -> dict:
        try:
            return self.client.call('finish_export', export_id=export_id, exit_ok=bool(ok))
        except ClientError as exc:
            return {'status': 'failed', 'detail': str(exc)}

    # memory settings and gauges ---------------------------------------------------------------
    def memory_settings(self) -> Tuple[int, int, int, int]:
        # what the window's own MemoryManager reported in local mode (configured - 1 frames)
        long_term = self.cfg.long_term
        return (int(long_term['min_mem_frames']) - 1, int(long_term['max_mem_frames']) - 1,
                int(long_term['max_num_tokens']), int(self.cfg['mem_every']))

    def update_config(self, cfg) -> None:
        self.memory = memory_config(cfg)

    def memory_gauges(self) -> Optional[Dict[str, Tuple[int, int]]]:
        return None

    def gpu_gauges(self) -> GpuGauges:
        last, gauges = self.gauges
        if time.monotonic() - last < GPU_GAUGE_SECONDS:
            return gauges
        try:
            status = self.client.call('gpu_status')
            memory = status['memory']
            gauges = GpuGauges((memory['total'] - memory['free_device']) / 2**30,
                               memory['total'] / 2**30, memory['allocated'] / 2**30)
        except ClientError:
            gauges = GpuGauges()
        self.gauges = (time.monotonic(), gauges)
        return gauges

    def clear_memory(self, permanent: bool) -> None:
        pass  # each propagation builds its memory from the anchors in the GPU worker

    def commit_to_permanent_memory(self, image, prob):
        raise BackendError('the permanent memory is built from the anchors of a task')

    def close(self) -> None:
        self.close_interaction()
        self.cancel_propagation()
        try:
            self.client.call('close_task', task_id=self.task_id, lease_epoch=self.epoch)
        except ClientError as exc:
            log.warning('closing %s: %s', self.task_id, exc)
