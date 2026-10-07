"""Device helpers: the precision of inference must not depend on how a device is spelled."""
import torch


def device_type(device) -> str:
    """'cuda' for 'cuda', 'cuda:0', torch.device('cuda', 1); 'cpu', 'mps' likewise."""
    return torch.device(device).type


def autocast(device, amp: bool):
    """Mixed precision as the labeling window has always used it: float16 autocast on CUDA when
    amp is set, nothing on other devices. 'cuda:0' and 'cuda' behave the same."""
    kind = device_type(device)
    return torch.autocast(kind, enabled=(amp and kind == 'cuda'))
