from typing import Optional

import torch
from .ritm.controller import InteractiveController
from .ritm.inference import utils


class ClickController:
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda',
                 max_size: int = 800,
                 *,
                 net: Optional[torch.nn.Module] = None,
                 empty_cache_after_click: bool = True):
        """Loads the RITM network from checkpoint_path, or shares an already loaded net.
        empty_cache_after_click: release the CUDA cache after every click (the original
        behaviour); the shared GPU worker turns it off, as it would discard other work's cache."""
        if net is None:
            net = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)
        self.net = net
        self.max_size = max_size
        self.empty_cache_after_click = empty_cache_after_click

        # Predictor params
        zoomin_params = {
            'skip_clicks': 1,
            'target_size': 480,
            'expansion_ratio': 1.4,
        }

        # NoBRS with no click limit: every click reaches the network. With f-BRS-B, clicks
        # after net_clicks_limit went to the L-BFGS refinement instead, which degraded the
        # mask (mean IoU 0.914 -> 0.839 at the 9th click on real labels).
        predictor_params = {
            'brs_mode': 'NoBRS',
            'prob_thresh': 0.5,
            'zoom_in_params': zoomin_params,
            'predictor_params': {
                'net_clicks_limit': None,
                'max_size': max_size,
            },
            'with_flip': True,
        }

        self.controller = InteractiveController(net, device, predictor_params,
                                                empty_cache_after_click=empty_cache_after_click)
        self.anchored = False
        self.device = device

    def session(self) -> 'ClickController':
        """A controller of its own (predictor, clicks, undo states) for one interaction, sharing
        this controller's network and settings."""
        return ClickController(device=self.device, max_size=self.max_size, net=self.net,
                               empty_cache_after_click=self.empty_cache_after_click)

    def unanchor(self):
        self.anchored = False

    def interact(self, image: torch.Tensor, x: int, y: int, is_positive: bool,
                 prev_mask: torch.Tensor):
        if not self.anchored:
            image = image.to(self.device, non_blocking=True)
            self.controller.set_image(image)
            self.controller.reset_predictor()
            self.anchored = True

        self.controller.add_click(x, y, is_positive, prev_mask=prev_mask)
        # return self.controller.result_mask
        return self.controller.probs_history[-1][1]
        # return (self.controller.probs_history[-1][1] > 0.5).float()

    def undo(self):
        self.controller.undo_click()
        if len(self.controller.probs_history) == 0:
            return None
        else:
            return (self.controller.probs_history[-1][1] > 0.5).float()
