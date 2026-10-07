"""
The GPU the worker runs on, as PyTorch reports it, and probes of its memory. Exactly one device:
the configured CUDA device (default cuda:0, i.e. the first device CUDA_VISIBLE_DEVICES exposes).
Nothing here depends on the GPU model; limits come from what the device reports and from
measurements (cutieplus.gpu.profiler, cutieplus.gpu.budget).
"""
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Protocol, Tuple

import torch


class NoGpuError(RuntimeError):
    """No usable CUDA device; the service does not fall back to the CPU on its own."""


@dataclass(frozen=True)
class HardwareInfo:
    backend: str  # 'cuda' | 'cpu' (CPU only for development and tests, when allowed)
    device: str  # e.g. 'cuda:0'
    index: Optional[int]
    name: str
    uuid: Optional[str]
    compute_capability: Optional[Tuple[int, int]]
    multiprocessors: Optional[int]
    total_bytes: Optional[int]
    free_bytes: Optional[int]  # when detected
    torch_allocated_bytes: int
    torch_reserved_bytes: int
    precisions: Dict[str, bool]
    torch_version: str
    cuda_version: Optional[str]
    visible_devices: Optional[str]  # CUDA_VISIBLE_DEVICES as set
    device_count: int

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.backend != 'cuda':
            return f'CPU ({self.name})'
        major, minor = self.compute_capability
        return (f'{self.name} ({self.device}, compute capability {major}.{minor}, '
                f'{self.total_bytes / 2**30:.1f} GiB)')


def resolve_device(device: str) -> str:
    """'cuda' is the first visible device, 'cuda:N' that one; anything else is the CPU."""
    if device == 'cuda':
        return 'cuda:0'
    return device


def supported_precisions(capability: Tuple[int, int], index: int) -> Dict[str, bool]:
    return {
        'float32': True,
        # half-precision arithmetic from compute capability 5.3; autocast uses it for convolutions
        'float16': capability >= (5, 3),
        'bfloat16': bool(torch.cuda.is_bf16_supported()) if index is not None else False,
        'tf32': capability >= (8, 0),
    }


def detect(device: str, allow_cpu: bool = False) -> HardwareInfo:
    """The device the worker will use. Raises NoGpuError when a CUDA device is asked for and
    there is none (CPU only with allow_cpu, for tests)."""
    device = resolve_device(device)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise NoGpuError(
                f'No CUDA device is available (CUDA_VISIBLE_DEVICES={visible!r}, PyTorch '
                f'{torch.__version__}, CUDA {torch.version.cuda}). The GPU worker needs one; '
                'check the driver with nvidia-smi.')
        index = int(device.split(':')[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise NoGpuError(f'{device} does not exist: {count} visible CUDA device(s).')
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        capability = (properties.major, properties.minor)
        uuid = getattr(properties, 'uuid', None)
        return HardwareInfo('cuda', device, index, properties.name,
                            str(uuid) if uuid is not None else None, capability,
                            properties.multi_processor_count, total, free,
                            torch.cuda.memory_allocated(index), torch.cuda.memory_reserved(index),
                            supported_precisions(capability, index), torch.__version__,
                            torch.version.cuda, visible, count)
    if not allow_cpu:
        raise NoGpuError(f'The GPU worker was configured for {device!r}; set allow_cpu only for '
                         'development and tests.')
    return HardwareInfo('cpu', 'cpu', None, 'cpu', None, None, os.cpu_count(), None, None, 0, 0,
                        {'float32': True, 'float16': False, 'bfloat16': False, 'tf32': False},
                        torch.__version__, torch.version.cuda, visible,
                        torch.cuda.device_count() if torch.cuda.is_available() else 0)


def effective_precision(amp: bool, hardware: HardwareInfo, allow_change: bool) -> Tuple[bool, list]:
    """(mixed precision on, recorded changes). Mixed precision (float16 autocast, the configured
    default) needs float16 support; without it the worker refuses to start unless the
    configuration allows running in float32, which is recorded as a quality-relevant change."""
    if not amp or hardware.backend != 'cuda' or hardware.precisions.get('float16'):
        return amp, []
    if not allow_change:
        raise NoGpuError(f'{hardware.name} has no float16 support and amp is on; set amp: False, '
                         'or allow_precision_change to run in float32.')
    return False, [{'setting': 'amp', 'configured': True, 'effective': False,
                    'reason': f'{hardware.name} does not support float16'}]


@dataclass
class MemorySnapshot:
    """Memory of the device and of this process's PyTorch allocator, in bytes."""
    total: int
    free_device: int  # free on the device, for every process (cudaMemGetInfo)
    allocated: int  # live tensors of this process
    reserved: int  # held by this process's caching allocator (allocated + cached free blocks)

    @property
    def cache_slack(self) -> int:
        """Reserved by the allocator but not in use: reusable without asking the device."""
        return max(0, self.reserved - self.allocated)

    @property
    def available(self) -> int:
        """What new tensors of this process can use: the device's free memory plus the
        allocator's own cached blocks. Reserved memory is not also counted as used: free_device
        already excludes it."""
        return self.free_device + self.cache_slack

    @property
    def outside_allocator(self) -> int:
        """Used on the device outside this process's allocator: other processes, and this
        process's CUDA context and libraries."""
        return max(0, self.total - self.free_device - self.reserved)

    def as_dict(self) -> dict:
        return dict(asdict(self), cache_slack=self.cache_slack, available=self.available,
                    outside_allocator=self.outside_allocator)


class MemoryProbe(Protocol):
    def snapshot(self) -> MemorySnapshot: ...
    def reset_peak(self) -> None: ...
    def peak_allocated(self) -> int: ...
    def peak_reserved(self) -> int: ...
    def synchronize(self) -> None: ...
    def release_cache(self) -> None: ...


class TorchMemoryProbe:
    """The real device."""

    def __init__(self, device: str):
        self.device = torch.device(resolve_device(device))

    def snapshot(self) -> MemorySnapshot:
        free, total = torch.cuda.mem_get_info(self.device)
        return MemorySnapshot(total, free, torch.cuda.memory_allocated(self.device),
                              torch.cuda.memory_reserved(self.device))

    def reset_peak(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def peak_allocated(self) -> int:
        return torch.cuda.max_memory_allocated(self.device)

    def peak_reserved(self) -> int:
        return torch.cuda.max_memory_reserved(self.device)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def release_cache(self) -> None:
        # returns cached free blocks to the device; live tensors are not affected
        torch.cuda.empty_cache()


@dataclass
class SimulatedMemoryProbe:
    """A GPU of any size for tests: other processes' usage and this process's allocator are set
    by the test."""
    total: int
    used_by_others: int = 0
    allocated: int = 0
    reserved: int = 0
    context: int = 300 << 20  # CUDA context and libraries of this process
    peak: int = 0
    history: list = field(default_factory=list)

    def snapshot(self) -> MemorySnapshot:
        free = self.total - self.used_by_others - self.context - self.reserved
        return MemorySnapshot(self.total, max(0, free), self.allocated, self.reserved)

    def reset_peak(self) -> None:
        self.peak = self.allocated

    def peak_allocated(self) -> int:
        return self.peak

    def peak_reserved(self) -> int:
        return max(self.peak, self.reserved)

    def synchronize(self) -> None:
        pass

    def release_cache(self) -> None:
        self.history.append('release_cache')
        self.reserved = self.allocated


class CpuMemoryProbe:
    """No device memory to manage (CPU development backend)."""

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(0, 0, 0, 0)

    def reset_peak(self) -> None:
        pass

    def peak_allocated(self) -> int:
        return 0

    def peak_reserved(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def release_cache(self) -> None:
        pass
