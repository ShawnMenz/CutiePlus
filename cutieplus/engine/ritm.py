"""
RITM click interactions with isolated state.

Each interaction (one frame, one image modality, one target object, its ordered clicks) has its
own predictor state; the network is shared. Only the active interaction holds GPU tensors.
Switching to another interaction keeps the previous one as a CPU snapshot, taken through RITM's
own state API (the one its click undo uses), and reconstructs the next one from its snapshot or,
when there is none (evicted, or a restarted worker), by replaying its accepted clicks in order.

The click path is the original one: ClickInteraction and ClickController (gui/interaction.py,
gui/click_controller.py) run exactly as in the labeling window, so the masks are the same.
"""
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from gui.click_controller import ClickController
from gui.interaction import ClickInteraction
from gui.interactive_utils import torch_prob_to_numpy_mask

Click = Tuple[int, int, bool]  # (x, y, is_negative) in image pixels


class InteractionError(Exception):
    pass


@dataclass
class ClickOutcome:
    prob: torch.Tensor  # (num_objects + 1) * H * W, on the engine's device
    mask: np.ndarray  # uint8 H * W index mask
    clicks: int  # accepted clicks after this operation
    reconstruction: Optional[str]  # None (already active), 'snapshot' or 'replay'
    reconstruction_seconds: float
    inference_seconds: float


def _move(value: Any, device, memo: Dict[int, Any]) -> Any:
    """value with every tensor on device; shared tensors stay shared."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in memo:
            memo[key] = value.to(device)
        return memo[key]
    if isinstance(value, dict):
        return {k: _move(v, device, memo) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, device, memo) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device, memo) for v in value)
    return value


def _nbytes(value: Any, seen: set) -> int:
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return value.nbytes
    if isinstance(value, dict):
        return sum(_nbytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v, seen) for v in value)
    return 0


@dataclass
class _Interaction:
    interaction_id: str
    image: np.ndarray  # H * W * 3 uint8, the shown modality
    prev_prob: torch.Tensor  # (num_objects + 1) * H * W on the CPU, the mask before the clicks
    target_object: int
    clicks: List[Click] = field(default_factory=list)
    live: Optional[ClickInteraction] = None  # only for the active interaction
    snapshot: Optional[dict] = None  # CPU state of an inactive interaction
    snapshot_bytes: int = 0

    @property
    def size(self) -> Tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]


class RitmEngine:
    """Interactions keyed by an id the caller chooses. Not thread-safe: one caller at a time."""

    def __init__(self,
                 click_controller: ClickController,
                 device: str,
                 amp: bool,
                 *,
                 snapshot_budget_bytes: int = 2 << 30,
                 max_interactions: int = 64):
        self.base = click_controller
        self.device = device
        self.amp = amp
        self.snapshot_budget_bytes = snapshot_budget_bytes
        self.max_interactions = max_interactions
        self.interactions: 'OrderedDict[str, _Interaction]' = OrderedDict()  # least recent first
        self.active: Optional[_Interaction] = None

    def _autocast(self):
        return torch.autocast(self.device, enabled=(self.amp and self.device == 'cuda'))

    # interactions
    def open(self, interaction_id: str, image: np.ndarray, prev_prob: torch.Tensor,
             target_object: int, clicks: Optional[List[Click]] = None) -> None:
        """Registers an interaction on the CPU; the GPU is used from its first click. clicks:
        accepted clicks of an interaction reopened elsewhere (a restarted worker); they are
        replayed when it becomes active."""
        if interaction_id in self.interactions:
            raise InteractionError(f'interaction {interaction_id} is already open')
        if len(self.interactions) >= self.max_interactions:
            raise InteractionError(f'{self.max_interactions} interactions are open; close one first')
        self.interactions[interaction_id] = _Interaction(
            interaction_id, np.ascontiguousarray(image), prev_prob.detach().to('cpu').clone(),
            int(target_object), list(clicks or []))

    def close(self, interaction_id: str) -> None:
        interaction = self.interactions.pop(interaction_id, None)
        if interaction is not None and interaction is self.active:
            interaction.live = None
            self.active = None

    def close_all(self) -> None:
        self.interactions.clear()
        self.active = None

    def clicks(self, interaction_id: str) -> List[Click]:
        return list(self._get(interaction_id).clicks)

    def click(self, interaction_id: str, x: int, y: int, is_negative: bool) -> ClickOutcome:
        interaction = self._get(interaction_id)
        kind, rebuild_seconds = self._activate(interaction)
        start = time.perf_counter()
        with self._autocast():
            interaction.live.push_point(x, y, is_neg=is_negative)
            prob = interaction.live.predict()
        interaction.clicks.append((int(x), int(y), bool(is_negative)))
        return self._outcome(interaction, prob, kind, rebuild_seconds, start)

    def undo(self, interaction_id: str) -> Optional[ClickOutcome]:
        """Takes back the last click; None when the interaction has one click or none (the
        window then undoes the whole edit instead)."""
        interaction = self._get(interaction_id)
        if len(interaction.clicks) <= 1:
            return None
        kind, rebuild_seconds = self._activate(interaction)
        start = time.perf_counter()
        controller = interaction.live.controller.controller
        controller.undo_click()
        interaction.live.pop_click()
        interaction.live.obj_mask = controller.probs_history[-1][1]
        with self._autocast():
            prob = interaction.live.predict()
        interaction.clicks.pop()
        return self._outcome(interaction, prob, kind, rebuild_seconds, start)

    def replay(self, interaction_id: str) -> ClickOutcome:
        """Rebuilds the interaction from its clicks alone (dropping any snapshot) and returns the
        result of the last click; used to verify reconstruction."""
        interaction = self._get(interaction_id)
        if not interaction.clicks:
            raise InteractionError('an interaction without clicks has nothing to replay')
        if interaction is self.active:
            interaction.live = None
            self.active = None
        interaction.snapshot, interaction.snapshot_bytes = None, 0
        kind, rebuild_seconds = self._activate(interaction)
        with self._autocast():
            prob = interaction.live.predict()
        return self._outcome(interaction, prob, kind, rebuild_seconds, time.perf_counter())

    # GPU residency
    def deactivate(self) -> None:
        """Moves the active interaction to the CPU (no GPU tensors are left behind)."""
        interaction = self.active
        if interaction is None:
            return
        self.active = None
        live, interaction.live = interaction.live, None
        controller = live.controller.controller
        memo: Dict[int, Any] = {}
        snapshot = _move({
            'clicker': controller.clicker.get_state(),
            'predictor': controller.predictor.get_states(),
            'states': controller.states,
            'probs_history': controller.probs_history,
            'obj_mask': live.obj_mask,
            'first_click': live.first_click,
            'clicks': list(live.clicks),
        }, 'cpu', memo)
        interaction.snapshot = snapshot
        interaction.snapshot_bytes = _nbytes(snapshot, set())
        self._enforce_snapshot_budget()

    def snapshot_bytes(self) -> int:
        return sum(item.snapshot_bytes for item in self.interactions.values())

    def drop_snapshots(self) -> int:
        """Frees every inactive snapshot (memory pressure); those interactions replay their clicks
        when used again. Returns the bytes freed."""
        freed = 0
        for interaction in self.interactions.values():
            if interaction is not self.active and interaction.snapshot is not None:
                freed += interaction.snapshot_bytes
                interaction.snapshot, interaction.snapshot_bytes = None, 0
        return freed

    def _enforce_snapshot_budget(self) -> None:
        # least recently used first
        for interaction in self.interactions.values():
            if self.snapshot_bytes() <= self.snapshot_budget_bytes:
                return
            if interaction is not self.active and interaction.snapshot is not None:
                interaction.snapshot, interaction.snapshot_bytes = None, 0

    def _get(self, interaction_id: str) -> _Interaction:
        try:
            interaction = self.interactions[interaction_id]
        except KeyError:
            raise InteractionError(f'interaction {interaction_id} is not open') from None
        self.interactions.move_to_end(interaction_id)
        return interaction

    def _new_live(self, interaction: _Interaction) -> ClickInteraction:
        # as the labeling window builds one: the shown image and the probabilities before clicking
        image = to_tensor(interaction.image).to(self.device, non_blocking=True)
        prev_prob = interaction.prev_prob.to(self.device, non_blocking=True)
        return ClickInteraction(image, prev_prob, interaction.size, self.base.session(),
                                interaction.target_object)

    def _activate(self, interaction: _Interaction) -> Tuple[Optional[str], float]:
        if interaction is self.active:
            return None, 0.0
        self.deactivate()
        start = time.perf_counter()
        live = self._new_live(interaction)
        if interaction.snapshot is not None:
            self._restore(live, interaction.snapshot)
            kind = 'snapshot'
        elif interaction.clicks:
            with self._autocast():
                for x, y, is_negative in interaction.clicks:
                    live.push_point(x, y, is_neg=is_negative)
            kind = 'replay'
        else:
            kind = None  # nothing to rebuild before the first click
        interaction.snapshot, interaction.snapshot_bytes = None, 0
        interaction.live = live
        self.active = interaction
        if kind is not None and self.device == 'cuda':
            torch.cuda.synchronize()
        return kind, time.perf_counter() - start if kind is not None else 0.0

    def _restore(self, live: ClickInteraction, snapshot: dict) -> None:
        # what ClickController.interact does before an interaction's first click, then the saved
        # predictor and click state, as RITM's undo restores them
        snapshot = _move(snapshot, self.device, {})
        controller = live.controller.controller
        controller.set_image(live.image.unsqueeze(0))
        controller.reset_predictor()
        live.controller.anchored = True
        controller.clicker.set_state(snapshot['clicker'])
        controller.predictor.set_states(snapshot['predictor'])
        controller.states = snapshot['states']
        controller.probs_history = snapshot['probs_history']
        live.obj_mask = snapshot['obj_mask']
        live.first_click = snapshot['first_click']
        live.clicks = list(snapshot['clicks'])

    def _outcome(self, interaction: _Interaction, prob: torch.Tensor, kind: Optional[str],
                 rebuild_seconds: float, start: float) -> ClickOutcome:
        mask = torch_prob_to_numpy_mask(prob)  # synchronizes with the device
        return ClickOutcome(prob, mask, len(interaction.clicks), kind, rebuild_seconds,
                            time.perf_counter() - start)
