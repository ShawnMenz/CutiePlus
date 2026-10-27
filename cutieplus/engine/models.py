"""
Loading the two models: Cutie (propagation) and RITM (clicks). The weights are only read; the
labeling process loads them once and reuses them for every task it opens.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from os import path
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig

from cutieplus.engine.settings import model_config


def weight_paths(cfg: DictConfig) -> Tuple[str, str]:
    """(Cutie weights, RITM weights). Relative paths in the config refer to the downloaded weights,
    so the program works regardless of the working directory."""
    from cutie.utils.download_models import download_models_if_needed
    weight_dir = download_models_if_needed()
    weights, ritm_weights = cfg.weights, cfg.ritm_weights
    if not path.isabs(weights):
        weights = path.join(weight_dir, path.basename(weights))
    if not path.isabs(ritm_weights):
        ritm_weights = path.join(weight_dir, path.basename(ritm_weights))
    return weights, ritm_weights


def load_cutie(cfg: DictConfig, weights: str, device: str):
    from cutie.model.cutie import CUTIE
    cutie = CUTIE(cfg).eval().to(device)
    model_weights = torch.load(weights, map_location=device, weights_only=True)
    cutie.load_weights(model_weights)
    return cutie


def load_ritm(ritm_weights: str, device: str, empty_cache_after_click: bool = True):
    from gui.click_controller import ClickController
    return ClickController(ritm_weights, device=device,
                           empty_cache_after_click=empty_cache_after_click)


def load_networks(cfg: DictConfig, device: str, empty_cache_after_click: bool = True):
    """(CUTIE, ClickController) on device."""
    weights, ritm_weights = weight_paths(cfg)
    return (load_cutie(cfg, weights, device),
            load_ritm(ritm_weights, device, empty_cache_after_click))


def file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    """Which weights and which model settings produced a result."""
    cutie_weights: str
    cutie_sha256: str
    ritm_weights: str
    ritm_sha256: str
    model_config_sha256: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def model_identity(cfg: DictConfig) -> ModelIdentity:
    weights, ritm_weights = weight_paths(cfg)
    config_digest = hashlib.sha256(
        json.dumps(model_config(cfg), sort_keys=True).encode()).hexdigest()
    return ModelIdentity(path.abspath(weights), file_sha256(weights), path.abspath(ritm_weights),
                         file_sha256(ritm_weights), config_digest)
