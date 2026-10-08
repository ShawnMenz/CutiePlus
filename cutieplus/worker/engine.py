"""
The GPU worker's engine: the only owner of CUDA, of the two models and of the mutable inference
state of the shared service. It keeps at most one RITM interaction and at most one Cutie
propagation context on the GPU; the coordinator decides what runs when (one command at a time).

Commands and replies are plain dicts with numpy arrays (cutieplus.worker.main sends them through
a pipe); no CUDA tensor ever leaves this process.
"""
import copy
import gc
import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from cutieplus.engine import models
from cutieplus.engine.images import FramePrefetcher, read_mask_source, read_rgb
from cutieplus.engine.propagation import (PropagationPlan, PropagationRuntime, apply_memory_config,
                                          new_inference_core)
from cutieplus.engine.ritm import InteractionError, RitmEngine
from cutieplus.gpu import hardware as hw
from cutieplus.gpu.budget import GPUResourceManager, MemoryModel, MemoryRequest, ResourcePolicy
from cutieplus.gpu.profiler import Profiler, memory_tokens
from gui.interactive_utils import index_numpy_to_one_hot_torch

log = logging.getLogger(__name__)

PREFETCH_FRAMES = 8


class CommandError(Exception):
    """A refused command; kind: 'invalid' | 'rejected' | 'oom' | 'pressure'."""

    def __init__(self, kind: str, message: str, **details):
        super().__init__(message)
        self.kind = kind
        self.details = details


def one_hot(mask: np.ndarray, num_objects: int) -> torch.Tensor:
    return index_numpy_to_one_hot_torch(mask, num_objects + 1)


class _Job:
    """The active propagation context."""

    def __init__(self, job_id: str, attempt_id: str, runtime: PropagationRuntime,
                 prefetcher: FramePrefetcher, plan_info: dict):
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.runtime = runtime
        self.prefetcher = prefetcher
        self.plan_info = plan_info


class TorchWorkerEngine:

    def __init__(self, cfg: DictConfig, service: dict):
        """cfg: the GUI configuration (models, memory settings); service: the service section
        (device, allow_cpu, gpu policy, profile frame size)."""
        self.cfg = cfg
        self.service = service
        torch.set_grad_enabled(False)
        if service.get('torch_threads'):
            torch.set_num_threads(int(service['torch_threads']))
        self.hardware = hw.detect(service.get('device', 'cuda'), bool(service.get('allow_cpu')))
        self.device = self.hardware.device
        if self.hardware.backend == 'cuda':
            torch.cuda.set_device(self.device)
            self.probe = hw.TorchMemoryProbe(self.device)
        else:
            self.probe = hw.CpuMemoryProbe()
        self.amp, self.quality_changes = hw.effective_precision(
            bool(cfg['amp']), self.hardware, bool(service.get('allow_precision_change')))
        for change in self.quality_changes:
            log.warning('Quality-relevant change: %s', change)

        start = time.perf_counter()
        before = self.probe.snapshot().allocated
        self.cutie, self.click_controller = models.load_networks(
            cfg, self.device, empty_cache_after_click=False)
        self.probe.synchronize()
        self.model_bytes = self.probe.snapshot().allocated - before
        self.load_seconds = time.perf_counter() - start
        self.identity = models.model_identity(cfg)

        self.policy = ResourcePolicy.from_config(service.get('gpu'))
        if self.hardware.backend == 'cuda' and service.get('profile', True):
            frame_size = tuple(service.get('profile_frame_size', (1080, 1920)))
            self.memory_model = Profiler(self.cutie, self.click_controller, cfg, self.device,
                                         self.amp, self.probe, frame_size).run(
                                             self.hardware, self.model_bytes)
        else:
            self.memory_model = MemoryModel(False, self.hardware.name, self.hardware.uuid,
                                            self.model_bytes)
        self.resources = GPUResourceManager(self.probe, self.policy, self.memory_model,
                                            unlimited=self.hardware.backend != 'cuda')
        self.ritm = RitmEngine(self.click_controller, self.device, self.amp,
                               snapshot_budget_bytes=int(service.get('ritm_snapshot_mib', 2048)) << 20,
                               max_interactions=int(service.get('max_interactions', 64)))
        self.job: Optional[_Job] = None
        self.interaction_frames: Dict[str, tuple] = {}

    # status
    def compute_dtype(self) -> str:
        """The dtype a convolution computes in under the engine's autocast: the precision the
        models really run in (float16 with amp on CUDA)."""
        from cutieplus.engine.device import autocast
        with autocast(self.device, self.amp):
            weight = torch.ones((1, 1, 1, 1), device=self.device)
            sample = torch.ones((1, 1, 2, 2), device=self.device)
            return str(torch.nn.functional.conv2d(sample, weight).dtype)

    def hello(self) -> dict:
        return {
            'compute_dtype': self.compute_dtype(),
            'hardware': self.hardware.as_dict(),
            'memory_model': self.memory_model.as_dict(),
            'model_identity': self.identity.as_dict(),
            'model_bytes': self.model_bytes,
            'load_seconds': self.load_seconds,
            'policy': vars(self.policy),
            'quality_changes': self.quality_changes,
            'amp': self.amp,
        }

    def status(self) -> dict:
        snapshot = self.probe.snapshot()
        return {
            'memory': snapshot.as_dict(),
            'budget': self.resources.budget().as_dict(),
            'peak_allocated': self.probe.peak_allocated(),
            'peak_reserved': self.probe.peak_reserved(),
            'active_interaction': self.ritm.active.interaction_id if self.ritm.active else None,
            'interactions': len(self.ritm.interactions),
            'ritm_snapshot_bytes': self.ritm.snapshot_bytes(),
            'job': None if self.job is None else {
                'job_id': self.job.job_id, 'attempt_id': self.job.attempt_id,
                'memory_tokens': memory_tokens(self.job.runtime.processor),
                'last_ti': self.job.runtime.last_ti},
        }

    # clicks
    def ritm_open(self, interaction_id: str, image_path: str, mask: Optional[dict],
                  num_objects: int, target_object: int, clicks=()) -> dict:
        """mask: the frame's committed mask before the clicks, {'path', 'binary'}; None: none."""
        image = read_rgb(image_path)
        mask = read_mask_source(mask, image.shape[:2])
        if mask.shape != image.shape[:2]:
            raise CommandError('invalid', f'mask {mask.shape} does not fit the image {image.shape}')
        prev = one_hot(mask, num_objects)
        try:
            self.ritm.open(interaction_id, image, prev, target_object,
                           [tuple(click) for click in clicks])
        except InteractionError as exc:
            raise CommandError('invalid', str(exc)) from None
        self.interaction_frames[interaction_id] = image.shape[:2]
        return {}

    def _click_reply(self, outcome) -> dict:
        return {'mask': outcome.mask, 'clicks': outcome.clicks,
                'reconstruction': outcome.reconstruction,
                'reconstruction_seconds': outcome.reconstruction_seconds,
                'inference_seconds': outcome.inference_seconds}

    def _guard_click(self, interaction_id: str) -> None:
        height, width = self.interaction_frames.get(interaction_id, (0, 0))
        if self.hardware.backend == 'cuda':
            reason = self.resources.admit_click(height, width)
            if reason is not None:
                self.relieve_memory()
                reason = self.resources.admit_click(height, width)
            if reason is not None:
                raise CommandError('rejected', f'not enough GPU memory for a click: {reason}')

    def _ritm_call(self, interaction_id: str, call):
        self._guard_click(interaction_id)
        try:
            return call()
        except InteractionError as exc:
            raise CommandError('invalid', str(exc)) from None
        except torch.OutOfMemoryError as exc:
            # the predictor may hold half a click: drop its GPU state; the accepted clicks are
            # replayed next time
            self.ritm.drop_live(interaction_id)
            self._free()
            raise CommandError('oom', f'out of GPU memory during a click: {exc}') from None

    def ritm_click(self, interaction_id: str, x: int, y: int, is_negative: bool) -> dict:
        return self._click_reply(self._ritm_call(
            interaction_id, lambda: self.ritm.click(interaction_id, x, y, is_negative)))

    def ritm_undo(self, interaction_id: str) -> dict:
        outcome = self._ritm_call(interaction_id, lambda: self.ritm.undo(interaction_id))
        return {'mask': None} if outcome is None else self._click_reply(outcome)

    def ritm_replay(self, interaction_id: str) -> dict:
        return self._click_reply(self._ritm_call(interaction_id,
                                                 lambda: self.ritm.replay(interaction_id)))

    def ritm_close(self, interaction_id: str) -> dict:
        self.ritm.close(interaction_id)
        self.interaction_frames.pop(interaction_id, None)
        return {}

    # propagation
    def prop_start(self, job: dict, plan_only: bool = False) -> dict:
        """Admission and, when admitted, the job's propagation context. Nothing is allocated for
        a job that is queued or rejected, nor with plan_only (the decision alone)."""
        if self.job is not None:
            raise CommandError('invalid', f'propagation context {self.job.job_id} is still active')
        resolution = job['resolution']
        memory = job['memory']
        mask_anchor_tis = sorted(int(ti) for ti in job['anchor_masks'])
        # a replay after a failure keeps the job's recorded anchors: it may wait, not change
        forced = job.get('permanent_tis')
        plan = self.resources.plan_propagation(
            self._request(job, mask_anchor_tis), job.get('click_frame'),
            None if forced is None else [int(ti) for ti in forced])
        if plan.decision != 'run' or plan_only:
            return {'decision': plan.decision, 'plan': plan.as_dict()}

        names = tuple(job['frame_names'])
        frame_paths = job['frame_paths']
        anchor_sources = {int(ti): source for ti, source in job['anchor_masks'].items()}
        num_objects = int(job.get('num_objects', 1))
        mask_anchor_count = len(mask_anchor_tis)
        bounds = job.get('bounds')  # absent in jobs recorded before intervals bounded runs
        propagation_plan = PropagationPlan(names, int(job['start_ti']), job['direction'],
                                           dict(job['anchors']), tuple(plan.permanent_tis), True,
                                           mask_anchor_count, num_objects,
                                           None if bounds is None else tuple(bounds))

        def read_anchor_mask(ti: int) -> np.ndarray:
            if ti not in anchor_sources:
                raise CommandError('invalid', f'anchor {names[ti]} has no mask')
            return read_mask_source(anchor_sources[ti])

        height, width = int(job['height']), int(job['width'])
        seed_mask = read_mask_source(job.get('seed_mask'), (height, width))
        try:
            job_cfg = apply_memory_config(copy.deepcopy(self.cfg), memory)
            # constructed with the default settings, then configured, as the window's processor
            processor = new_inference_core(self.cutie, self.cfg, int(resolution['max_internal_size']),
                                           bool(resolution['upscale_to_internal_size']))
            prefetcher = FramePrefetcher(lambda ti: read_rgb(frame_paths[ti]),
                                         propagation_plan.input_order(), depth=PREFETCH_FRAMES)
            runtime = PropagationRuntime(processor, propagation_plan, prefetcher.get,
                                         read_anchor_mask,
                                         one_hot(seed_mask, num_objects).to(self.device),
                                         self.device, self.amp, job_cfg)
        except torch.OutOfMemoryError as exc:
            self._free()
            raise CommandError('oom', f'out of GPU memory creating the context: {exc}') from None
        self.job = _Job(job['job_id'], job['attempt_id'], runtime, prefetcher, plan.as_dict())
        return {'decision': 'run', 'plan': plan.as_dict()}

    def _request(self, job: dict, anchor_tis) -> MemoryRequest:
        memory = job['memory']
        resolution = job['resolution']
        use_long_term = bool(self.cfg.use_long_term)
        return MemoryRequest(int(job['height']), int(job['width']),
                             int(resolution['max_internal_size']),
                             bool(resolution['upscale_to_internal_size']),
                             int(memory['max_mem_frames']) - 1, use_long_term,
                             int(memory['max_num_tokens']), anchor_tis, int(job['start_ti']))

    def prop_step(self, job_id: str, attempt_id: str) -> dict:
        job = self.job
        if job is None or job.job_id != job_id or job.attempt_id != attempt_id:
            raise CommandError('invalid', f'no propagation context for {job_id}/{attempt_id}')
        runtime = job.runtime
        if self.hardware.backend == 'cuda' and runtime.phase == 'frames':
            query = job.plan_info['query_tokens']
            tokens = memory_tokens(runtime.processor)
            if not self.resources.step_fits(query, tokens):
                actions = self.relieve_memory()
                if not self.resources.step_fits(query, tokens):
                    raise CommandError('pressure', 'GPU memory dropped below the next step\'s need',
                                       actions=actions)
        start = time.perf_counter()
        try:
            result = runtime.step()
            mask = result.mask if result.needs_commit else None  # computed on the device
            if self.hardware.backend == 'cuda':
                torch.cuda.synchronize(self.device)
        except torch.OutOfMemoryError as exc:
            # the temporal state may be half updated: never continue from it
            self.prop_release(job_id)
            self._free()
            raise CommandError('oom', f'out of GPU memory in a propagation step: {exc}') from None
        reply = {'kind': result.kind, 'ti': result.ti, 'mask': mask, 'finished': result.finished,
                 'stop_reason': result.stop_reason, 'messages': result.messages,
                 'inference_seconds': time.perf_counter() - start}
        if result.finished:
            self.prop_release(job_id)
        return reply

    def prop_release(self, job_id: str) -> dict:
        if self.job is not None and self.job.job_id == job_id:
            self.job.prefetcher.close()
            self.job = None
            # the GPU is shared with other programs: give the context's cached blocks back (the
            # tensors are gone already; the next job allocates again in milliseconds)
            self._free()
        return {}

    # memory
    def _free(self) -> None:
        gc.collect()
        self.probe.release_cache()

    def relieve_memory(self) -> list:
        """The recovery order of the specification, as far as the worker can go on its own:
        release cached blocks, move the inactive interaction state off the GPU, keep fewer frames
        read ahead. Returns what was done."""
        actions = []
        self._free()
        actions.append('released cached GPU blocks')
        if self.ritm.active is not None:
            self.ritm.deactivate()
            self._free()
            actions.append('moved the active click interaction to CPU memory')
        if self.job is not None and self.job.prefetcher.depth > 1:
            self.job.prefetcher.set_depth(1)
            actions.append('read fewer frames ahead')
        return actions

    def close(self) -> None:
        if self.job is not None:
            self.prop_release(self.job.job_id)
        self.ritm.close_all()
