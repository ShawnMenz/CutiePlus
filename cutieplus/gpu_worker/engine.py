"""
What the GPU worker does for the service, one command at a time: RITM click interactions (each
with its own state; the network is shared) and Cutie propagation runs, one at a time, each step
sent by the service so that clicks run between the steps of a long run. Every input comes from the
job's immutable snapshot: the frames are read from the input, the anchor masks and the start
frame's probabilities are the ones captured at submission. A propagated mask is written to the
job's staging folder (palette PNG, as the window writes masks) and its SHA-256 returned; the window
applies it once the job is done. Nothing is published from here.

TorchEngine runs the models; tests use FakeEngine (fake.py), which answers the same commands
without them.
"""
import copy
import gc
import hashlib
import io
import logging
import os
import time
from os import path
from typing import Dict, Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PREFETCH_FRAMES = 8
OPS = ('hello', 'status', 'reset', 'ritm_open', 'ritm_click', 'ritm_undo', 'ritm_close',
       'prop_start', 'prop_step', 'prop_release')


class CommandError(Exception):
    """A refused or failed command; kind: 'invalid' | 'oom' | 'failed'."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def inside(file_path: str, root: str, what: str) -> str:
    """The real path of file_path, which must lie in root."""
    real = path.realpath(file_path)
    root = path.realpath(root)
    if not (real == root or real.startswith(root + os.sep)):
        raise CommandError('invalid', f'{what} {file_path} lies outside {root}')
    return real


def mask_from_png(data: Optional[bytes], shape) -> np.ndarray:
    if data is None:
        return np.zeros(shape, np.uint8)
    mask = np.array(Image.open(io.BytesIO(data)))
    if mask.shape != tuple(shape):
        raise CommandError('invalid', f'a mask of {mask.shape} for frames of {tuple(shape)}')
    return mask.astype(np.uint8)


def stage_mask(folder: str, name: str, mask: np.ndarray, palette) -> str:
    """Writes the mask as <folder>/<name>.png (whole or not at all); returns its SHA-256."""
    os.makedirs(folder, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8))
    image.putpalette(palette)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    data = buffer.getvalue()
    final = path.join(folder, f'{name}.png')
    temporary = final + '.tmp'
    with open(temporary, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, final)
    return hashlib.sha256(data).hexdigest()


class Job:
    """The active propagation run."""

    def __init__(self, job_id: str, token: str, staging: str, names, runtime, prefetcher=None):
        self.job_id = job_id
        self.token = token
        self.staging = staging
        self.names = names
        self.runtime = runtime
        self.prefetcher = prefetcher


class Engine:
    """What both engines share: the checks of a command's paths, and the active job."""

    def __init__(self, workspace_root: str, jobs_dir: str):
        self.workspace_root = workspace_root
        self.jobs_dir = jobs_dir
        self.job: Optional[Job] = None
        self.interaction_shapes: Dict[str, tuple] = {}

    def active_job(self, job_id: str, token: str) -> Job:
        if self.job is None or self.job.job_id != job_id or self.job.token != token:
            raise CommandError('invalid', f'no run {job_id} with that token here')
        return self.job

    def check_snapshot(self, job_id: str, token: str, snapshot: dict) -> tuple:
        """(frame paths, staging folder) of a run, inside the workspace and the jobs folder."""
        if self.job is not None:
            raise CommandError('invalid', f'run {self.job.job_id} is still active')
        paths = [inside(frame, self.workspace_root, 'frame') for frame in snapshot['frame_paths']]
        if len(paths) != len(snapshot['frame_names']):
            raise CommandError('invalid', 'one path per frame')
        staging = inside(path.join(self.jobs_dir, job_id, token), self.jobs_dir, 'staging')
        return paths, staging

    def status_extra(self) -> dict:
        return {'job': None if self.job is None else self.job.job_id,
                'interactions': len(self.interaction_shapes)}


class TorchEngine(Engine):

    def __init__(self, cfg, workspace_root: str, jobs_dir: str, device: str = 'cuda'):
        super().__init__(workspace_root, jobs_dir)
        import torch
        from cutie.utils.palette import davis_palette
        from cutieplus.engine import models
        from cutieplus.engine.ritm import RitmEngine
        self.torch = torch
        torch.set_grad_enabled(False)
        self.cfg = cfg
        self.device = device
        self.amp = bool(cfg['amp'])
        self.palette = davis_palette
        start = time.perf_counter()
        self.cutie, self.click_controller = models.load_networks(cfg, device,
                                                                 empty_cache_after_click=False)
        self.load_seconds = time.perf_counter() - start
        self.identity = models.model_identity(cfg)
        self.ritm = RitmEngine(self.click_controller, device, self.amp)

    def _one_hot(self, mask: np.ndarray, num_objects: int):
        from gui.interactive_utils import index_numpy_to_one_hot_torch
        return index_numpy_to_one_hot_torch(mask, num_objects + 1)

    def _free(self) -> None:
        gc.collect()
        if 'cuda' in self.device:
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()

    # status
    def hello(self) -> dict:
        name = self.torch.cuda.get_device_name() if 'cuda' in self.device else self.device
        return {'engine': 'torch', 'device': self.device, 'device_name': name, 'amp': self.amp,
                'model': self.identity.as_dict(), 'load_seconds': self.load_seconds}

    def status(self) -> dict:
        status = self.status_extra()
        if 'cuda' in self.device:
            free, total = self.torch.cuda.mem_get_info()
            status.update(memory_used_mb=(total - free) >> 20, memory_total_mb=total >> 20)
        return status

    def reset(self) -> dict:
        """A new link to the service: nothing of the previous one is kept."""
        if self.job is not None:
            self.prop_release(self.job.job_id, self.job.token)
        self.ritm.close_all()
        self.interaction_shapes.clear()
        self._free()
        return {}

    # clicks
    def ritm_open(self, interaction_id: str, image_path: str, mask_png: Optional[bytes],
                  num_objects: int, target_object: int, clicks=()) -> dict:
        from cutieplus.engine.images import read_rgb
        from cutieplus.engine.ritm import InteractionError
        image = read_rgb(inside(image_path, self.workspace_root, 'frame'))
        mask = mask_from_png(mask_png, image.shape[:2])
        try:
            self.ritm.open(interaction_id, image, self._one_hot(mask, num_objects),
                           int(target_object), [tuple(click) for click in clicks])
        except InteractionError as exc:
            raise CommandError('invalid', str(exc)) from None
        self.interaction_shapes[interaction_id] = image.shape[:2]
        return {}

    def _click_reply(self, outcome) -> dict:
        return {'mask': outcome.mask, 'clicks': outcome.clicks,
                'prob': outcome.prob.float().cpu().numpy().astype(np.float16)}

    def _ritm(self, interaction_id: str, call):
        from cutieplus.engine.ritm import InteractionError
        try:
            return call()
        except InteractionError as exc:
            raise CommandError('invalid', str(exc)) from None
        except self.torch.OutOfMemoryError as exc:
            # the predictor may hold half a click: its GPU state goes; the accepted clicks are
            # replayed at the next one
            self.ritm.drop_live(interaction_id)
            self._free()
            raise CommandError('oom', f'out of GPU memory during a click: {exc}') from None

    def ritm_click(self, interaction_id: str, x: int, y: int, is_negative: bool) -> dict:
        return self._click_reply(self._ritm(
            interaction_id, lambda: self.ritm.click(interaction_id, int(x), int(y),
                                                    bool(is_negative))))

    def ritm_undo(self, interaction_id: str) -> dict:
        outcome = self._ritm(interaction_id, lambda: self.ritm.undo(interaction_id))
        return {'mask': None, 'clicks': 0} if outcome is None else self._click_reply(outcome)

    def ritm_close(self, interaction_id: str) -> dict:
        self.ritm.close(interaction_id)
        self.interaction_shapes.pop(interaction_id, None)
        return {}

    # propagation
    def prop_start(self, job_id: str, token: str, snapshot: dict) -> dict:
        from cutieplus.engine.images import FramePrefetcher, read_rgb
        from cutieplus.engine.propagation import (PropagationRuntime, apply_memory_config,
                                                  new_inference_core, task_plan)
        paths, staging = self.check_snapshot(job_id, token, snapshot)
        names = tuple(snapshot['frame_names'])
        height, width = int(snapshot['height']), int(snapshot['width'])
        num_objects = int(snapshot['num_objects'])
        anchor_masks = {name: mask_from_png(data, (height, width))
                        for name, data in snapshot['anchor_masks'].items()}
        bounds = snapshot.get('bounds')
        try:
            job_cfg = apply_memory_config(copy.deepcopy(self.cfg), snapshot['memory'])
            resolution = snapshot['resolution']
            processor = new_inference_core(self.cutie, job_cfg,
                                           int(resolution['max_internal_size']),
                                           bool(resolution['upscale_to_internal_size']))
            plan = task_plan(processor, names, height, width, int(snapshot['start_ti']),
                             snapshot['direction'], dict(snapshot['anchors']),
                             None if bounds is None else tuple(bounds), num_objects)
            prefetcher = FramePrefetcher(lambda ti: read_rgb(paths[ti]), plan.input_order(),
                                         depth=PREFETCH_FRAMES)

            def anchor_mask(ti: int) -> np.ndarray:
                if names[ti] not in anchor_masks:
                    raise CommandError('invalid', f'anchor {names[ti]} has no mask')
                return anchor_masks[names[ti]]

            seed = self.torch.from_numpy(np.asarray(snapshot['seed_prob'], np.float32))
            runtime = PropagationRuntime(processor, plan, prefetcher.get, anchor_mask,
                                         seed.to(self.device), self.device, self.amp, job_cfg)
        except self.torch.OutOfMemoryError as exc:
            self._free()
            raise CommandError('oom', f'out of GPU memory preparing the run: {exc}') from None
        self.job = Job(job_id, token, staging, names, runtime, prefetcher)
        return {'permanent_anchors': [names[ti] for ti in plan.permanent_tis]}

    def prop_step(self, job_id: str, token: str) -> dict:
        from cutieplus.engine.propagation import memory_sizes
        job = self.active_job(job_id, token)
        try:
            result = job.runtime.step()
            if 'cuda' in self.device:
                self.torch.cuda.synchronize()
        except self.torch.OutOfMemoryError as exc:
            # the temporal state may be half updated: never continued
            self.prop_release(job_id, token)
            raise CommandError('oom', f'out of GPU memory in a propagation step: {exc}') from None
        reply = {'kind': result.kind, 'ti': result.ti, 'finished': result.finished,
                 'stop_reason': result.stop_reason, 'messages': list(result.messages),
                 'memory': memory_sizes(job.runtime.processor), 'mask': result.mask,
                 'name': None if result.ti is None else job.names[result.ti]}
        if result.needs_commit:
            reply['sha256'] = stage_mask(job.staging, job.names[result.ti], result.mask,
                                         self.palette)
        return reply

    def prop_release(self, job_id: str, token: str) -> dict:
        """Ends the run: its tensors go, the device is synchronized and its cache returned
        before the reply, so the next run never overlaps it."""
        if self.job is not None and self.job.job_id == job_id and self.job.token == token:
            if self.job.prefetcher is not None:
                self.job.prefetcher.close()
            self.job = None
            self._free()
        return {'released': True}
