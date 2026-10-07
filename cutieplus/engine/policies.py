"""
Inference policies of task mode, as pure functions: the processing size chosen for small targets,
memory tokens per frame, and which mask anchors go into the permanent memory. The labeling window
(local mode) and the shared service use the same functions, so both decide alike.
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# a target whose area at the default internal size is below SMALL_TARGET_AREA pixels is
# processed at SMALL_TARGET_INTERNAL_SIZE instead (on RGBT sequences: a 78 px car went from
# mean IoU 0.48 to 0.64 against SAM3; targets of 2.5k-12k px were unchanged, at ~2.5x the time)
SMALL_TARGET_AREA = 1500
SMALL_TARGET_INTERNAL_SIZE = 960
TARGET_AREA_SAMPLES = 25
# memory tokens x query tokens one propagation step may compare. The original release fixed it
# at 4e8, described as ~8 GB of GPU memory. It is now a quality and speed cap only: the memory
# side is measured on the GPU at run time (cutieplus.gpu), which can lower the limit further.
LEGACY_MEMORY_READ_BUDGET = 4e8


@dataclass(frozen=True)
class Resolution:
    """The processing size of a sequence: max_internal_size and whether smaller frames are
    enlarged to it."""
    max_internal_size: int
    upscale_to_internal_size: bool
    median_area: Optional[int]
    small_target: bool

    def message(self) -> Optional[str]:
        if self.median_area is None:
            return None
        if self.small_target:
            return (f'Small target (median {self.median_area} px): processing at '
                    f'{self.max_internal_size} px.')
        return f'Target median {self.median_area} px: processing at {self.max_internal_size} px.'


def target_area_frames(anchors: Dict[str, dict], prefill: Iterable[str]) -> List[str]:
    """The frames whose masks measure the target: the mask anchors, else a sample of the
    prefill frames."""
    names = [name for name, anchor in anchors.items() if anchor['kind'] == 'mask']
    if not names:
        ordered = sorted(prefill)
        names = ordered[::max(1, len(ordered) // TARGET_AREA_SAMPLES)][:TARGET_AREA_SAMPLES]
    return names


def median_target_area(names: Sequence[str],
                       read_mask: Callable[[str], Optional[np.ndarray]]) -> Optional[int]:
    areas = []
    for name in names:
        mask = read_mask(name)
        if mask is not None and mask.any():
            areas.append(int((mask > 0).sum()))
    return int(np.median(areas)) if areas else None


def choose_resolution(area: Optional[int], height: int, width: int,
                      default_size: int) -> Resolution:
    """Stride-16 features lose small targets at the default size; those are processed larger."""
    if area is None or default_size <= 0:
        return Resolution(default_size, False, None, False)
    scale = min(1.0, default_size / min(height, width))
    if area * scale**2 < SMALL_TARGET_AREA:
        return Resolution(SMALL_TARGET_INTERNAL_SIZE, True, area, True)
    return Resolution(default_size, False, area, False)


def internal_frame_size(height: int, width: int, max_internal_size: int,
                        upscale: bool) -> tuple:
    """(height, width) at which InferenceCore processes a frame of this size."""
    min_side = min(height, width)
    if max_internal_size > 0 and (min_side > max_internal_size or
                                  (upscale and min_side < max_internal_size)):
        return int(height / min_side * max_internal_size), int(width / min_side * max_internal_size)
    return height, width


def memory_tokens_per_frame(height: int, width: int, max_internal_size: int,
                            upscale: bool) -> int:
    """Key/value tokens of one memory frame (features at stride 16, padded)."""
    h, w = internal_frame_size(height, width, max_internal_size, upscale)
    return math.ceil(h / 16) * math.ceil(w / 16)


def other_memory_tokens(tokens_per_frame: int, max_mem_frames: int, use_long_term: bool,
                        max_long_tokens: int) -> int:
    """Memory tokens besides the permanent anchors: the working memory (MemoryManager's
    max_mem_frames, which already excludes the first frame, plus the frame being added) and the
    long-term memory."""
    tokens = (max_mem_frames + 1) * tokens_per_frame
    if use_long_term:
        tokens += max_long_tokens
    return tokens


def permanent_anchor_limit(tokens_per_frame: int, other_tokens: int,
                           read_budget: float = LEGACY_MEMORY_READ_BUDGET) -> int:
    """How many anchor frames fit in the permanent memory when one step may compare read_budget
    memory x query tokens (at least one)."""
    return max(1, int((read_budget / tokens_per_frame - other_tokens) // tokens_per_frame))


def select_permanent_anchors(anchor_tis: Iterable[int], start_ti: int, limit: int) -> List[int]:
    """The documented selection policy: the mask anchors nearest to the start frame (ties: the
    earlier frame), in time order. The others still guide the run as ordinary memory frames when
    propagation reaches them; no anchor is ever dropped from the task."""
    ordered = sorted(anchor_tis)
    nearest = sorted(ordered, key=lambda ti: abs(ti - start_ti))
    return sorted(nearest[:max(0, limit)])
