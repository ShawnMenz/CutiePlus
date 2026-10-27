"""
What the labeling window asks of its backend, LocalTaskBackend (cutieplus/backends/local.py): the
window's process holds the models and writes the task files itself. The window draws; the backend
runs the models and changes the task's official data.

The interface carries numpy arrays, tensors and plain values, never Qt objects.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

from gui.edit_history import FrameState

Interval = Tuple[int, int]


class BackendError(Exception):
    """An operation was refused or failed; the message is meant for the window's log."""


@dataclass
class FrameUpdate:
    """A frame after a change of its official data."""
    ti: int
    mask: Optional[np.ndarray]  # the frame's mask file after the change; None: no file
    anchor: Optional[dict]  # the frame's anchor entry after the change; None: not an anchor


@dataclass
class ClickResult:
    update: FrameUpdate
    prob: torch.Tensor  # (num_objects + 1) * H * W soft result, on the window's device
    clicks: int  # accepted clicks of the interaction
    reconstruction: Optional[str] = None  # 'snapshot' | 'replay' when the state was rebuilt


@dataclass
class PropagatedFrame:
    ti: int
    kind: str  # 'seed' | 'anchor' | 'frame'
    mask: np.ndarray
    prob: Optional[torch.Tensor] = None  # the soft result, for the display
    image: Optional[np.ndarray] = None  # the anchor-modality frame, when the backend read it


@dataclass
class PropagationProgress:
    state: str  # 'running' | 'finished' | 'cancelled' | 'failed'
    frames: List[PropagatedFrame] = field(default_factory=list)  # since the last call, in order
    messages: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None  # 'end' | 'empty_anchor' | 'cancelled' | an error
    stopped_at: Optional[int] = None  # the empty anchor that stopped the run
    masks_written: int = 0

    @property
    def done(self) -> bool:
        return self.state in ('finished', 'cancelled', 'failed')


@dataclass
class BinaryExport:
    folder: str  # a plain workspace's binary_masks/


@dataclass
class GpuGauges:
    """The window's two GPU gauges, in GiB; None when unknown."""
    used: Optional[float] = None
    total: Optional[float] = None
    torch_used: Optional[float] = None


class TaskBackend(Protocol):
    device: str  # where the models and the window's display tensors live
    resolution_message: Optional[str]

    # clicks (RITM); one interaction at a time per window
    def open_interaction(self, ti: int, modality: str, image: np.ndarray, prev_prob: torch.Tensor,
                         target_object: int) -> None: ...
    def add_click(self, x: int, y: int, is_negative: bool) -> ClickResult: ...
    def undo_click(self) -> Optional[ClickResult]: ...
    def interaction_clicks(self) -> int: ...
    def close_interaction(self) -> None: ...

    # propagation (Cutie)
    # modality: the frames the run reads (the one shown); None: the task's anchor modality
    def start_propagation(self, start_ti: int, direction: str, seed_prob: torch.Tensor,
                          modality: Optional[str] = None) -> None: ...
    def advance_propagation(self) -> PropagationProgress: ...
    def cancel_propagation(self) -> None: ...
    def release_propagation(self) -> None: ...

    # official frame data
    def commit_frame(self, ti: int, mask: np.ndarray, anchor: Optional[str]) -> FrameUpdate: ...
    def remove_anchor(self, ti: int) -> FrameUpdate: ...
    def restore_frame(self, ti: int, state: FrameState) -> FrameUpdate: ...
    def frame_state(self, ti: int) -> FrameState: ...
    def wait_for_saves(self) -> None: ...

    # task data
    # descriptions: the texts of the added intervals (the first one's is the sequence's)
    def set_intervals(self, intervals: List[Interval],
                      descriptions: Optional[Dict[Interval, str]] = None,
                      categories: Optional[Dict[Interval, str]] = None) -> None: ...
    def set_description(self, text: str) -> None: ...
    def set_category(self, name: str) -> None: ...  # one of the subset's categories
    def set_attention(self, flagged: bool) -> None: ...  # task.attention follows
    # a plain workspace's binary PNGs; a task's Save submits it for review instead
    def export_binary(self, target_objects: List[int],
                      progress: Callable[[float], None]) -> Optional[BinaryExport]: ...
    def submit(self) -> List[Interval]: ...
    def reset_all_masks(self) -> int: ...
    # accepts the task under review; quit_at: when Ctrl+Q asked for it
    def mark_done(self, quit_at: Optional[str] = None) -> None: ...

    # memory settings and gauges
    def update_config(self, cfg) -> None: ...
    def memory_gauges(self) -> Optional[Dict[str, Tuple[int, int]]]: ...
    def gpu_gauges(self) -> GpuGauges: ...
    def clear_memory(self, permanent: bool) -> None: ...
    def commit_to_permanent_memory(self, image: torch.Tensor, prob: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...
