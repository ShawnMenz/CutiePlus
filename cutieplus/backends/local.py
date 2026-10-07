"""
The original mode: the labeling window's process holds Cutie and RITM and writes the task files
itself (masks through the ResourceManager's save threads, anchors and progress through
TaskState). The inference runs through the same engine as the shared service's GPU worker.
"""
import os
import shutil
from os import path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

try:
    from torch import mps
except ImportError:  # pragma: no cover
    mps = None

from cutie.inference.inference_core import InferenceCore
from cutieplus.backends.base import (BackendError, BinaryExport, ClickResult, FrameUpdate,
                                     GpuGauges, Interval, PropagatedFrame, PropagationProgress)
from cutieplus.engine import policies
from cutieplus.engine.device import autocast
from cutieplus.engine.images import FramePrefetcher
from cutieplus.engine.propagation import PropagationPlan, PropagationRuntime
from cutieplus.engine.ritm import RitmEngine
from gui.edit_history import FrameState
from gui.exporter import convert_mask_to_binary
from gui.resource_manager import ResourceManager, write_atomically
from gui.task_state import (TaskState, submit_for_review, write_description_text,
                            write_interval_text)

# frames read ahead of a propagation run
PREFETCH_FRAMES = 8


class LocalTaskBackend:
    kind = 'local'

    def __init__(self, cfg: DictConfig, networks, res_man: ResourceManager,
                 task: Optional[TaskState]):
        self.cfg = cfg
        self.device: str = cfg['device']
        self.amp: bool = cfg['amp']
        self.num_objects: int = cfg['num_objects']
        self.cutie, click_controller = networks
        self.res_man = res_man
        self.task = task
        # the window's InferenceCore shares the window's config object, as before
        self.processor: Optional[InferenceCore] = InferenceCore(self.cutie, cfg)
        self.ritm = RitmEngine(click_controller, self.device, self.amp, max_interactions=1)
        self.interaction_id: Optional[str] = None
        self.interaction_ti: Optional[int] = None
        self.interactions_opened = 0
        self.runtime: Optional[PropagationRuntime] = None
        self.prefetcher: Optional[FramePrefetcher] = None
        self.masks_written = 0
        self.resolution_message: Optional[str] = None
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
    def permanent_anchor_limit(self) -> int:
        # the original budget, unchanged in local mode
        tokens = policies.memory_tokens_per_frame(self.res_man.h, self.res_man.w,
                                                  self.processor.max_internal_size,
                                                  self.processor.upscale_to_internal_size)
        memory = self.processor.memory
        other = policies.other_memory_tokens(tokens, memory.max_mem_frames, memory.use_long_term,
                                             getattr(memory, 'max_long_tokens', 0))
        return policies.permanent_anchor_limit(tokens, other)

    def plan(self, start_ti: int, direction: str) -> PropagationPlan:
        names = tuple(self.res_man.names)
        if self.task is None:
            return PropagationPlan(names, start_ti, direction, {}, (), False, 0, self.num_objects)
        name_to_ti = self.res_man.name_to_ti
        anchor_tis = sorted(name_to_ti[name] for name, anchor in self.task.anchors.items()
                            if anchor['kind'] == 'mask' and name in name_to_ti)
        permanent = policies.select_permanent_anchors(anchor_tis, start_ti,
                                                      self.permanent_anchor_limit())
        anchors = {name: anchor['kind'] for name, anchor in self.task.anchors.items()}
        return PropagationPlan(names, start_ti, direction, anchors, tuple(permanent), True,
                               len(anchor_tis), self.num_objects)

    def anchor_mask(self, ti: int) -> np.ndarray:
        mask = self.res_man.get_mask(ti)
        if mask is None:
            raise BackendError(f'Anchor {self.res_man.names[ti]} has no mask file; '
                               'remove the anchor or import its mask.')
        return mask

    def start_propagation(self, start_ti: int, direction: str, seed_prob: torch.Tensor) -> None:
        self.release_propagation()
        # a mask still being saved would be read half-finished
        self.res_man.save_queue.join()
        plan = self.plan(start_ti, direction)
        workers = int(self.cfg.get('num_read_workers', 0))
        self.prefetcher = FramePrefetcher(self.res_man._get_image_unbuffered, plan.input_order(),
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
                self.task.touch()

    # official frame data
    def commit_frame(self, ti: int, mask: np.ndarray, anchor: Optional[str]) -> FrameUpdate:
        """Saves the frame's mask; anchor 'edit' records a human edit, 'accept' accepts the shown
        mask as an anchor, None leaves the anchors alone."""
        self.res_man.save_mask(ti, mask)
        name = self.res_man.names[ti]
        if self.task is not None and anchor is not None:
            kind = 'mask' if (mask > 0).any() else 'empty'
            self.task.set_anchor(name, kind, edited=(anchor == 'edit'))
        return FrameUpdate(ti, mask, self._anchor(name))

    def remove_anchor(self, ti: int) -> FrameUpdate:
        name = self.res_man.names[ti]
        self.task.remove_anchor(name)
        self.res_man.delete_mask(ti)
        return FrameUpdate(ti, self.res_man.get_mask(ti), None)

    def restore_frame(self, ti: int, state: FrameState) -> FrameUpdate:
        self.res_man.save_queue.join()
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        if state.mask is None:
            if path.exists(mask_path):
                os.remove(mask_path)
        else:
            def write(temporary: str) -> None:
                with open(temporary, 'wb') as f:
                    f.write(state.mask)

            write_atomically(mask_path, write)
        self.res_man.invalidate(ti)
        if self.task is not None:
            self.task.restore_anchor(name, state.anchor)
        return FrameUpdate(ti, self.res_man.get_mask(ti), self._anchor(name))

    def frame_state(self, ti: int) -> FrameState:
        self.res_man.save_queue.join()
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        mask = None
        if path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                mask = f.read()
        anchor = self._anchor(name)
        return FrameState(mask, dict(anchor) if anchor is not None else None)

    def wait_for_saves(self) -> None:
        self.res_man.save_queue.join()

    def _anchor(self, name: str) -> Optional[dict]:
        return self.task.anchor(name) if self.task is not None else None

    # task data
    def set_intervals(self, intervals: List[Interval]) -> None:
        self.task.save_intervals(intervals)
        self.task.touch()

    def set_description(self, text: str) -> None:
        self.task.save_description(text)
        self.task.touch()

    def export_binary(self, target_objects: List[int],
                      progress: Callable[[float], None]) -> Optional[BinaryExport]:
        """Binary PNGs of the masks in binary_masks/; a task also writes its intervals and
        description there and goes under review. None when there are no masks."""
        self.res_man.save_queue.join()
        intervals = None
        if self.task is not None:
            intervals, _ = self.task.intervals(self.res_man.T)
            self.task.save_intervals(intervals)
        mask_folder = self.res_man.mask_dir
        save_folder = path.join(self.res_man.workspace, 'binary_masks')
        if not path.exists(mask_folder):
            return None
        os.makedirs(save_folder, exist_ok=True)
        convert_mask_to_binary(mask_folder, save_folder, target_objects,
                               progress_callback=progress)
        if self.task is not None:
            write_interval_text(save_folder, intervals)
            write_description_text(save_folder, self.task.description)
            submit_for_review(self.res_man.workspace, self.task.labeler)
        return BinaryExport(save_folder, intervals)

    def reset_all_masks(self, export_dirs: List[str]) -> Tuple[int, int]:
        """(mask PNGs removed, export trees removed); raises OSError part way."""
        removed = self.res_man.clear_sequence_masks()
        removed_exports = 0
        if self.task is not None:
            self.task.clear_mask_references()
            for directory in export_dirs:
                if path.isdir(directory):
                    shutil.rmtree(directory)
                    removed_exports += 1
        return removed, removed_exports

    # the external export command (task_set.json) runs in the window; local mode records nothing
    def begin_export(self) -> Optional[str]:
        return None

    def finish_export(self, export_id: Optional[str], ok: bool) -> dict:
        return {'status': 'complete' if ok else 'failed', 'detail': None}

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
        try:
            memory = self.processor.memory
            permanent = memory.work_mem.perm_size(0)
            return {
                'permanent': (permanent, permanent),
                'working': (memory.work_mem.non_perm_size(0), memory.max_work_tokens),
                'long_term': (memory.long_mem.non_perm_size(0), memory.max_long_tokens),
            }
        except AttributeError:
            return None

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
