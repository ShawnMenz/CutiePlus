"""
The labeling window's backend: its process holds Cutie and RITM and writes the task files itself
(masks through the ResourceManager's save threads, anchors and progress through TaskState). The
inference runs through cutieplus/engine.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

try:
    from torch import mps
except ImportError:  # pragma: no cover
    mps = None

from cutie.inference.inference_core import InferenceCore
from cutieplus.backends.base import (BackendError, ClickResult, GpuGauges, PropagatedFrame,
                                     PropagationProgress)
from cutieplus.backends.task_data import TaskDataBackend
from cutieplus.engine import policies
from cutieplus.engine.device import autocast
from cutieplus.engine.images import FramePrefetcher
from cutieplus.engine.propagation import (PropagationPlan, PropagationRuntime, memory_sizes,
                                          task_plan)
from cutieplus.engine.ritm import RitmEngine
from gui.resource_manager import ResourceManager
from gui.task_state import TaskState

# frames read ahead of a propagation run
PREFETCH_FRAMES = 8


class LocalTaskBackend(TaskDataBackend):

    def __init__(self, cfg: DictConfig, networks, res_man: ResourceManager,
                 task: Optional[TaskState]):
        super().__init__(cfg, res_man, task)
        self.device: str = cfg['device']
        self.amp: bool = cfg['amp']
        self.num_objects: int = cfg['num_objects']
        self.cutie, click_controller = networks
        # the window's InferenceCore shares the window's config object, as before
        self.processor: Optional[InferenceCore] = InferenceCore(self.cutie, cfg)
        self.ritm = RitmEngine(click_controller, self.device, self.amp, max_interactions=1)
        self.interaction_id: Optional[str] = None
        self.interaction_ti: Optional[int] = None
        self.interactions_opened = 0
        self.runtime: Optional[PropagationRuntime] = None
        self.prefetcher: Optional[FramePrefetcher] = None
        self.masks_written = 0
        if task is not None:
            self.configure_task_resolution()

    # processing size
    def configure_task_resolution(self) -> None:
        names = policies.target_area_frames(self.task.anchors, self.task.prefill)
        name_to_ti = self.res_man.name_to_ti
        area = policies.median_target_area(
            names, lambda name: self.res_man.get_mask(name_to_ti[name])
            if name in name_to_ti else None)
        resolution = policies.choose_resolution(area, self.res_man.h, self.res_man.w,
                                                self.processor.max_internal_size)
        self.processor.max_internal_size = resolution.max_internal_size
        if resolution.small_target:
            self.processor.upscale_to_internal_size = True
        self.resolution_message = resolution.message()

    def memory_settings(self) -> Tuple[int, int, int, int]:
        """(min and max working memory frames, max long-term tokens, memory frame every) as the
        window's controls start."""
        memory = self.processor.memory
        return (memory.min_mem_frames, memory.max_mem_frames, memory.max_long_tokens,
                self.processor.mem_every)

    def update_config(self, cfg: DictConfig) -> None:
        self.processor.update_config(cfg)

    # clicks
    def open_interaction(self, ti: int, modality: str, image: np.ndarray, prev_prob: torch.Tensor,
                         target_object: int) -> None:
        self.close_interaction()
        self.interactions_opened += 1
        self.interaction_id = f'local-{self.interactions_opened}'
        self.interaction_ti = ti
        self.ritm.open(self.interaction_id, image, prev_prob, target_object)

    def add_click(self, x: int, y: int, is_negative: bool) -> ClickResult:
        outcome = self.ritm.click(self.interaction_id, x, y, is_negative)
        update = self.commit_frame(self.interaction_ti, outcome.mask, 'edit')
        return ClickResult(update, outcome.prob, outcome.clicks, outcome.reconstruction)

    def undo_click(self) -> Optional[ClickResult]:
        if self.interaction_id is None:
            return None
        outcome = self.ritm.undo(self.interaction_id)
        if outcome is None:
            return None
        update = self.commit_frame(self.interaction_ti, outcome.mask, 'edit')
        return ClickResult(update, outcome.prob, outcome.clicks, outcome.reconstruction)

    def interaction_clicks(self) -> int:
        if self.interaction_id is None:
            return 0
        return len(self.ritm.clicks(self.interaction_id))

    def close_interaction(self) -> None:
        if self.interaction_id is not None:
            self.ritm.close(self.interaction_id)
        self.interaction_id = self.interaction_ti = None

    # propagation
    def plan(self, start_ti: int, direction: str) -> PropagationPlan:
        names = tuple(self.res_man.names)
        if self.task is None:
            return PropagationPlan(names, start_ti, direction, {}, (), False, 0, self.num_objects)
        anchors = {name: anchor['kind'] for name, anchor in self.task.anchors.items()}
        # inside an interval, the run stays in it
        bounds = policies.propagation_bounds(self.task.intervals(len(names))[0], start_ti)
        return task_plan(self.processor, names, self.res_man.h, self.res_man.w, start_ti,
                         direction, anchors, bounds, self.num_objects)

    def anchor_mask(self, ti: int) -> np.ndarray:
        mask = self.res_man.get_mask(ti)
        if mask is None:
            raise BackendError(f'Anchor {self.res_man.names[ti]} has no mask file; '
                               'remove the anchor or import its mask.')
        return mask

    def start_propagation(self, start_ti: int, direction: str, seed_prob: torch.Tensor,
                          modality: Optional[str] = None) -> None:
        self.release_propagation()
        # a mask still being saved would be read half-finished
        self.res_man.save_queue.join()
        plan = self.plan(start_ti, direction)
        workers = int(self.cfg.get('num_read_workers', 0))
        # the frames of the modality shown: the anchors' masks fit both, which are aligned
        self.prefetcher = FramePrefetcher(
            lambda ti: self.res_man._get_image_unbuffered(ti, modality), plan.input_order(),
            depth=PREFETCH_FRAMES, threads=max(1, min(workers, 4)))
        self.runtime = PropagationRuntime(self.processor, plan, self.prefetcher.get,
                                          self.anchor_mask, seed_prob, self.device, self.amp,
                                          self.cfg if self.task is not None else None)
        self.masks_written = 0

    def advance_propagation(self) -> PropagationProgress:
        if self.runtime is None:
            return PropagationProgress('finished', masks_written=self.masks_written)
        result = self.runtime.step()
        progress = PropagationProgress('running', messages=list(result.messages))
        if result.kind == 'stop':
            progress.stopped_at = result.ti
        elif result.kind != 'anchor_memory':
            if result.needs_commit:
                self.res_man.save_mask(result.ti, result.mask)
                self.masks_written += 1
            progress.frames.append(PropagatedFrame(result.ti, result.kind, result.mask,
                                                   result.prob, result.image))
        if result.finished:
            progress.state = 'finished'
            progress.stop_reason = result.stop_reason
            self.release_propagation()
        progress.masks_written = self.masks_written
        return progress

    def cancel_propagation(self) -> None:
        self.release_propagation()

    def release_propagation(self) -> None:
        if self.prefetcher is not None:
            self.prefetcher.close()
            self.prefetcher = None
        if self.runtime is not None:
            self.runtime = None
            if self.task is not None and self.masks_written:
                self.task.touch(changes_masks=True)

    # memory buttons and gauges
    def clear_memory(self, permanent: bool) -> None:
        if permanent:
            self.processor.clear_memory()
        else:
            self.processor.clear_non_permanent_memory()
        self._empty_cache()
        self.processor.update_config(self.cfg)

    def commit_to_permanent_memory(self, image: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        with autocast(self.device, self.amp):
            return self.processor.step(image, prob[1:], idx_mask=False, force_permanent=True)

    def memory_gauges(self) -> Optional[Dict[str, Tuple[int, int]]]:
        return memory_sizes(self.processor)

    def gpu_gauges(self) -> GpuGauges:
        if 'cuda' in self.device:
            free, total = torch.cuda.mem_get_info()
            return GpuGauges((total - free) / 2**30, total / 2**30,
                             torch.cuda.max_memory_allocated() / 2**30)
        if 'mps' in self.device and mps is not None:
            return GpuGauges(mps.current_allocated_memory() / 2**30)
        return GpuGauges()

    def _empty_cache(self) -> None:
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device and mps is not None:
            mps.empty_cache()

    def close(self) -> None:
        self.close_interaction()
        self.ritm.close_all()
        self.release_propagation()
        # the window's signal connections keep its controller alive; free the GPU state now
        self.processor = None
        self._empty_cache()
