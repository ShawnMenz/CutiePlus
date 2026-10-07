"""Configuration helpers without torch (the coordinator uses them and never imports torch)."""
from omegaconf import DictConfig, OmegaConf

# inference settings that change a model's output; they belong to a job's model configuration
MODEL_CONFIG_KEYS = ('max_internal_size', 'amp', 'flip_aug', 'top_k', 'mem_every', 'stagger_updates',
                     'chunk_size', 'use_long_term', 'max_mem_frames', 'long_term')


def model_config(cfg: DictConfig) -> dict:
    """The output-relevant part of the configuration, plus the network architecture."""
    selected = {key: cfg[key] for key in MODEL_CONFIG_KEYS if key in cfg}
    selected['model'] = cfg.model
    return OmegaConf.to_container(OmegaConf.create(selected), resolve=True)
