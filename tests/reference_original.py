"""
The original click and propagation code of MainController (commit 039c919), without the window:
the reference the new engine is compared with. Only the GUI calls were removed; the model calls
and their order are verbatim.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import autocast
from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from gui.click_controller import ClickController
from gui.interaction import ClickInteraction
from gui.interactive_utils import index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask


def original_clicks(click_ctrl: ClickController, image: np.ndarray, prev_prob: torch.Tensor,
                    tar_obj: int, clicks: List[Tuple[int, int, bool]], device: str, amp: bool,
                    undo_after: Optional[int] = None) -> List[torch.Tensor]:
    """The probabilities after each click (MainController.click_fn); undo_after=k takes back
    the last click after the k-th one (MainController.on_undo) and appends that result too."""
    results = []
    with autocast(device, enabled=(amp and device == 'cuda')):
        image_torch = to_tensor(image).to(device, non_blocking=True)
        click_ctrl.unanchor()
        interaction = ClickInteraction(image_torch, prev_prob, image.shape[:2], click_ctrl, tar_obj)
        for index, (x, y, is_neg) in enumerate(clicks, 1):
            interaction.push_point(x, y, is_neg=is_neg)
            results.append(interaction.predict().to(device, non_blocking=True))
            if undo_after == index:
                click_ctrl.controller.undo_click()
                interaction.pop_click()
                interaction.obj_mask = click_ctrl.controller.probs_history[-1][1]
                results.append(interaction.predict().to(device))
    return results


def original_propagation(processor: InferenceCore, cfg, frames: List[np.ndarray],
                         names: List[str], anchors: Dict[str, dict],
                         masks: Dict[int, np.ndarray], start_ti: int, direction: str,
                         seed_prob: torch.Tensor, device: str, amp: bool, anchor_limit: int,
                         num_objects: int = 1) -> Dict[int, np.ndarray]:
    """MainController.rebuild_permanent_memory + on_propagate in task mode; returns the masks the
    run saves (frame index -> mask)."""
    saved = {}
    with autocast(device, enabled=(amp and device == 'cuda')):

        def model_image(ti):
            return to_tensor(frames[ti]).to(device, non_blocking=True)

        # rebuild_permanent_memory
        processor.clear_memory()
        processor.update_config(cfg)
        name_to_ti = {name: ti for ti, name in enumerate(names)}
        all_anchor_tis = sorted(name_to_ti[name] for name, anchor in anchors.items()
                                if anchor['kind'] == 'mask' and name in name_to_ti)
        nearest = sorted(all_anchor_tis, key=lambda ti: abs(ti - start_ti))
        anchor_tis = sorted(nearest[:anchor_limit])
        for ti in anchor_tis:
            prob = index_numpy_to_one_hot_torch(masks[ti], num_objects + 1).to(device)
            processor.step(model_image(ti), prob[1:], idx_mask=False, force_permanent=True)
        processor.clear_sensory_memory()

        # on_propagate
        processor.clear_sensory_memory()
        processor.step(model_image(start_ti), seed_prob[1:], idx_mask=False)
        step = 1 if direction == 'forward' else -1
        ti = start_ti + step
        while 0 <= ti < len(frames):
            anchor = anchors.get(names[ti])
            if anchor is not None and anchor['kind'] == 'empty':
                break
            if anchor is not None:
                anchor_prob = index_numpy_to_one_hot_torch(masks[ti], num_objects + 1).to(device)
                processor.step(model_image(ti), anchor_prob[1:], idx_mask=False)
            else:
                prob = processor.step(model_image(ti))
                saved[ti] = torch_prob_to_numpy_mask(prob)
            if ti == 0 or ti == len(frames) - 1:
                break
            ti += step
    return saved
