"""
Step-based Cutie propagation.

The labeling window used to propagate in one loop. PropagationRuntime does the same work one GPU
step at a time, so a scheduler can serve a click between two steps: each mask anchor put into the
permanent memory is one step, the start frame is one step, and every following frame is one step.
Between steps the whole Cutie temporal state stays inside the runtime's InferenceCore, untouched.

The sequence of InferenceCore calls is the one of the original MainController.on_propagate and
rebuild_permanent_memory, so the masks are the same.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from cutie.inference.inference_core import InferenceCore
from cutieplus.engine import policies
from cutieplus.engine.device import autocast
from gui.interactive_utils import index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask

Direction = str  # 'forward' | 'backward'


@dataclass(frozen=True)
class PropagationPlan:
    """What one propagation run does; immutable once the run starts."""
    frame_names: Tuple[str, ...]  # every frame of the sequence, in order (ti = index)
    start_ti: int
    direction: Direction
    anchors: Mapping[str, str]  # frame name -> 'mask' | 'empty', every anchor of the task
    permanent_tis: Tuple[int, ...] = ()  # mask anchors put into the permanent memory, time order
    rebuild_permanent_memory: bool = True  # task mode; a plain workspace keeps its memory
    mask_anchor_count: int = 0  # mask anchors of the task, for the message
    num_objects: int = 1
    # the frames the run may reach (0-indexed, inclusive): the interval of the start frame
    bounds: Optional[Tuple[int, int]] = None

    @property
    def length(self) -> int:
        return len(self.frame_names)

    @property
    def step_sign(self) -> int:
        return 1 if self.direction == 'forward' else -1

    def frames_after_start(self) -> List[int]:
        """The frames the run may reach, in order (the reader of the original loop)."""
        return policies.frames_after_start(self.start_ti, self.direction, self.length, self.bounds)

    def interval_text(self) -> Optional[str]:
        """'12-20' (1-indexed, as the interval boxes) when the run is kept inside an interval
        smaller than the sequence."""
        if self.bounds is None or (self.bounds[0] <= 0 and self.bounds[1] >= self.length - 1):
            return None
        return f'{self.bounds[0] + 1}-{self.bounds[1] + 1}'

    def interval_end_message(self) -> List[str]:
        """The run has read its last frame: say so when that is the edge of its interval."""
        interval = self.interval_text()
        if interval is None:
            return []
        edge = 'end' if self.direction == 'forward' else 'start'
        return [f'Reached the {edge} of the interval {interval}.']

    def input_order(self) -> List[int]:
        """Every frame whose image the run reads, in the order it reads them."""
        return list(self.permanent_tis) + [self.start_ti] + self.frames_after_start()


@dataclass
class StepResult:
    kind: str  # 'anchor_memory' | 'seed' | 'anchor' | 'frame' | 'stop'
    ti: Optional[int]
    prob: Optional[torch.Tensor] = None  # (num_objects + 1) * H * W on the device
    mask: Optional[np.ndarray] = None  # 'frame': the new mask; 'seed'/'anchor': the mask shown
    image: Optional[np.ndarray] = None  # the model-modality frame that was processed
    finished: bool = False
    stop_reason: Optional[str] = None  # 'end' | 'empty_anchor'
    messages: List[str] = field(default_factory=list)

    @property
    def needs_commit(self) -> bool:
        """A propagated mask to save; anchors and the start frame are never written."""
        return self.kind == 'frame'


class PropagationRuntime:
    """
    processor: the InferenceCore that holds the run's temporal state. Task mode clears and
    rebuilds its memory first; a fresh one per run is equivalent.
    read_image(ti): the frame in the anchor (model) modality, H * W * 3 uint8.
    read_anchor_mask(ti): the index mask of a mask anchor.
    seed_prob: (num_objects + 1) * H * W probabilities of the start frame.
    cfg: the configuration applied when the memory is rebuilt (memory sizes, mem_every, top_k).
    """

    def __init__(self, processor: InferenceCore, plan: PropagationPlan,
                 read_image: Callable[[int], np.ndarray],
                 read_anchor_mask: Callable[[int], np.ndarray], seed_prob: torch.Tensor,
                 device: str, amp: bool, cfg: Optional[DictConfig] = None):
        self.processor = processor
        self.plan = plan
        self.read_image = read_image
        self.read_anchor_mask = read_anchor_mask
        self.seed_prob = seed_prob
        self.device = device
        self.amp = amp
        self.cfg = cfg
        self.phase = 'anchors' if plan.rebuild_permanent_memory else 'seed'
        self.anchor_cursor = 0
        self.frames = plan.frames_after_start()
        self.frame_cursor = 0
        self.last_ti: Optional[int] = None  # the last frame processed
        self.steps = 0
        self._pending_messages: List[str] = []

    @property
    def finished(self) -> bool:
        return self.phase == 'finished'

    def _autocast(self):
        return autocast(self.device, self.amp)

    def _model_image(self, ti: int) -> Tuple[np.ndarray, torch.Tensor]:
        from torchvision.transforms.functional import to_tensor
        image = self.read_image(ti)
        return image, to_tensor(image).to(self.device, non_blocking=True)

    def _mask_prob(self, mask: np.ndarray) -> torch.Tensor:
        return index_numpy_to_one_hot_torch(mask, self.plan.num_objects + 1).to(self.device)

    def step(self) -> StepResult:
        if self.finished:
            raise RuntimeError('the propagation has finished')
        self.steps += 1
        with self._autocast():
            if self.phase == 'anchors':
                result = self._anchor_memory_step()
                if result is not None:
                    return result
            if self.phase == 'seed':
                return self._seed_step()
            return self._frame_step()

    def _anchor_memory_step(self) -> Optional[StepResult]:
        """One permanent anchor; None once they are all in memory (the caller goes on to the
        start frame in the same step)."""
        if self.anchor_cursor == 0:
            self.processor.clear_memory()
            if self.cfg is not None:
                self.processor.update_config(self.cfg)
        if self.anchor_cursor < len(self.plan.permanent_tis):
            ti = self.plan.permanent_tis[self.anchor_cursor]
            self.anchor_cursor += 1
            _, image = self._model_image(ti)
            prob = self._mask_prob(self.read_anchor_mask(ti))
            self.processor.step(image, prob[1:], idx_mask=False, force_permanent=True)
            return StepResult('anchor_memory', ti)
        self.processor.clear_sensory_memory()
        self.phase = 'seed'
        self._pending_messages = [
            f'Permanent memory rebuilt from {len(self.plan.permanent_tis)} of '
            f'{self.plan.mask_anchor_count} mask anchors.']
        return None

    def _seed_step(self) -> StepResult:
        messages, self._pending_messages = self._pending_messages, []
        ti = self.plan.start_ti
        messages.append(f'Propagation started at t={ti}.')
        if self.plan.interval_text() is not None:
            messages.append(f'Propagation stays inside the interval {self.plan.interval_text()}.')
        image_np, image = self._model_image(ti)
        self.processor.clear_sensory_memory()
        prob = self.processor.step(image, self.seed_prob[1:], idx_mask=False)
        self.phase = 'frames'
        self.last_ti = ti
        finished = not self.frames
        if finished:
            self.phase = 'finished'
            messages += self.plan.interval_end_message()
        return StepResult('seed', ti, prob=prob, mask=torch_prob_to_numpy_mask(prob),
                          image=image_np, finished=finished, stop_reason='end' if finished else None,
                          messages=messages)

    def _frame_step(self) -> StepResult:
        ti = self.frames[self.frame_cursor]
        self.frame_cursor += 1
        name = self.plan.frame_names[ti]
        kind = self.plan.anchors.get(name)
        if kind == 'empty':
            self.phase = 'finished'
            return StepResult('stop', ti, finished=True, stop_reason='empty_anchor', messages=[
                f'Stopped at t={ti}: the target is marked absent here (empty anchor).'])
        image_np, image = self._model_image(ti)
        if kind == 'mask':
            # a human mask: fed to the model, its file is left untouched
            mask = self.read_anchor_mask(ti).copy()
            prob = self.processor.step(image, self._mask_prob(mask)[1:], idx_mask=False)
            result = StepResult('anchor', ti, prob=prob, mask=mask, image=image_np)
        else:
            prob = self.processor.step(image)
            result = StepResult('frame', ti, prob=prob, mask=torch_prob_to_numpy_mask(prob),
                                image=image_np)
        self.last_ti = ti
        if self.frame_cursor >= len(self.frames):
            self.phase = 'finished'
            result.finished = True
            result.stop_reason = 'end'
            result.messages += self.plan.interval_end_message()
        return result


def new_inference_core(network, cfg: DictConfig, max_internal_size: int,
                       upscale_to_internal_size: bool) -> InferenceCore:
    """A fresh InferenceCore at the given processing size."""
    processor = InferenceCore(network, cfg)
    processor.max_internal_size = max_internal_size
    processor.upscale_to_internal_size = upscale_to_internal_size
    return processor


def memory_config(cfg: DictConfig) -> dict:
    """The memory settings of a run, as the window's controls set them."""
    return {
        'mem_every': int(cfg['mem_every']),
        'top_k': int(cfg['top_k']),
        'min_mem_frames': int(cfg.long_term['min_mem_frames']),
        'max_mem_frames': int(cfg.long_term['max_mem_frames']),
        'max_num_tokens': int(cfg.long_term['max_num_tokens']),
    }


def apply_memory_config(cfg: DictConfig, memory: Mapping[str, int]) -> DictConfig:
    with open_dict(cfg):
        cfg['mem_every'] = memory['mem_every']
        cfg['top_k'] = memory['top_k']
        cfg.long_term['min_mem_frames'] = memory['min_mem_frames']
        cfg.long_term['max_mem_frames'] = memory['max_mem_frames']
        cfg.long_term['max_num_tokens'] = memory['max_num_tokens']
    return cfg
