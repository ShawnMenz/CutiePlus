"""
GPU memory budgets from measurements on the device in use.

MemoryModel holds linear models of the memory the work needs, fitted by the profiler
(cutieplus.gpu.profiler) on the current device; GPUResourceManager compares them with what the
device has free right now (other processes included) and decides whether a click or a
propagation job may start, and with how many permanent anchors.

Quality-sensitive settings (processing size, precision, memory sizes) are never changed here. The
one setting the budget may lower is the number of mask anchors in the permanent memory, with the
documented selection policy (nearest to the start frame); the plan records it.
"""
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from cutieplus.engine import policies
from cutieplus.gpu.hardware import MemoryProbe, MemorySnapshot

MiB = 1 << 20


@dataclass
class MemoryModel:
    """Bytes, as fitted on one device. Q: query tokens of a frame (features at stride 16); N:
    memory tokens read by a step; pixels: frame height x width."""
    measured: bool  # False: conservative estimates from tensor sizes (no profile yet)
    device_name: str = ''
    device_uuid: Optional[str] = None
    model_bytes: int = 0  # both models' weights and buffers
    # RITM click, above what is resident before it: fixed + per_pixel * pixels
    ritm_fixed: float = 512 * MiB
    ritm_per_pixel: float = 200.0
    # Cutie run, resident between steps: fixed + per_query * Q + per_memory_token * N
    resident_fixed: float = 64 * MiB
    resident_per_query: float = 64 * 1024.0
    resident_per_memory_token: float = 4096.0
    # Cutie step, transient above the resident state: fixed + per_query * Q + per_product * Q * N
    step_fixed: float = 256 * MiB
    step_per_query: float = 256 * 1024.0
    step_per_product: float = 24.0
    samples: List[dict] = field(default_factory=list)  # the measurements behind the fit

    def ritm_peak(self, height: int, width: int) -> int:
        return int(self.ritm_fixed + self.ritm_per_pixel * height * width)

    def propagation_resident(self, query_tokens: int, memory_tokens: int) -> int:
        return int(self.resident_fixed + self.resident_per_query * query_tokens +
                   self.resident_per_memory_token * memory_tokens)

    def propagation_step(self, query_tokens: int, memory_tokens: int) -> int:
        return int(self.step_fixed + self.step_per_query * query_tokens +
                   self.step_per_product * query_tokens * memory_tokens)

    def propagation_peak(self, query_tokens: int, memory_tokens: int) -> int:
        return (self.propagation_resident(query_tokens, memory_tokens) +
                self.propagation_step(query_tokens, memory_tokens))

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryModel':
        return cls(**data)


@dataclass
class ResourcePolicy:
    """Configurable limits (gui_config.yaml, section service.gpu)."""
    headroom_fraction: float = 0.05  # of the device's memory, never planned for
    headroom_min_bytes: int = 512 * MiB
    max_worker_bytes: Optional[int] = None  # optional cap on what the worker's allocator holds
    estimate_margin: float = 1.25  # applied to every estimate
    reserve_for_clicks: bool = True  # keep room for one click while a propagation holds memory
    # memory x query tokens per step: a quality and speed cap (the original release's 4e8); the
    # memory budget may lower the number of anchors further
    max_affinity_elements: float = policies.LEGACY_MEMORY_READ_BUDGET
    allow_fewer_permanent_anchors: bool = True  # documented selection policy under memory limits
    queue_wait_seconds: float = 600.0  # a job that cannot get memory for this long is rejected
    max_oom_retries: int = 1  # automatic retries after an out-of-memory error, if conditions changed

    @classmethod
    def from_config(cls, section: Optional[dict]) -> 'ResourcePolicy':
        section = dict(section or {})
        if section.get('max_worker_gib') is not None:
            section['max_worker_bytes'] = int(float(section.pop('max_worker_gib')) * (1 << 30))
        section.pop('max_worker_gib', None)
        if section.get('headroom_min_mib') is not None:
            section['headroom_min_bytes'] = int(float(section.pop('headroom_min_mib')) * MiB)
        section.pop('headroom_min_mib', None)
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in section.items() if key in known})


@dataclass
class Budget:
    snapshot: MemorySnapshot
    headroom: int
    available: int  # for new allocations of the worker, after headroom and the cap
    ceiling: int  # the most the worker could ever get (no other process, empty caches)

    def as_dict(self) -> dict:
        return {'snapshot': self.snapshot.as_dict(), 'headroom': self.headroom,
                'available': self.available, 'ceiling': self.ceiling}


@dataclass
class MemoryRequest:
    """What a propagation job asks for."""
    height: int
    width: int
    max_internal_size: int
    upscale_to_internal_size: bool
    max_mem_frames: int  # as MemoryManager counts them (the configured value - 1)
    use_long_term: bool
    max_long_tokens: int
    anchor_tis: Sequence[int]  # mask anchors with a mask
    start_ti: int


@dataclass
class ResourcePlan:
    """Admission of a propagation job and its effective resource configuration."""
    decision: str  # 'run' | 'queue' | 'reject'
    reason: str
    permanent_tis: List[int]
    estimate_bytes: int
    query_tokens: int
    memory_tokens: int
    quality_anchor_limit: int  # anchors allowed by max_affinity_elements
    memory_anchor_limit: Optional[int]  # anchors allowed by the memory budget
    limited_by: Optional[str]  # None | 'quality_cap' | 'memory'
    budget: dict
    model_measured: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def message(self, mask_anchors: int) -> Optional[str]:
        if self.limited_by == 'memory':
            return (f'GPU memory allows {len(self.permanent_tis)} of {mask_anchors} mask anchors in '
                    'the permanent memory (those nearest to the start frame); the others guide '
                    'the run when it reaches them.')
        return None


class GPUResourceManager:

    def __init__(self, probe: MemoryProbe, policy: ResourcePolicy, model: MemoryModel):
        self.probe = probe
        self.policy = policy
        self.model = model

    def headroom(self, total: int) -> int:
        return int(max(self.policy.headroom_fraction * total, self.policy.headroom_min_bytes))

    def budget(self) -> Budget:
        snapshot = self.probe.snapshot()
        headroom = self.headroom(snapshot.total)
        available = snapshot.available
        ceiling = snapshot.total - snapshot.allocated  # nothing else on the device
        if self.policy.max_worker_bytes is not None:
            room_under_cap = self.policy.max_worker_bytes - snapshot.allocated
            available = min(available, room_under_cap + headroom)
            ceiling = min(ceiling, room_under_cap + headroom)
        return Budget(snapshot, headroom, available - headroom, ceiling - headroom)

    def click_requirement(self, height: int, width: int) -> int:
        return int(self.policy.estimate_margin * self.model.ritm_peak(height, width))

    def admit_click(self, height: int, width: int) -> Optional[str]:
        """None when a click fits now, else why not."""
        need = self.click_requirement(height, width)
        budget = self.budget()
        if need <= budget.available:
            return None
        return (f'a click needs ~{need / 2**30:.2f} GiB of GPU memory, '
                f'{max(0, budget.available) / 2**30:.2f} GiB is free')

    def plan_propagation(self, request: MemoryRequest, click_frame: Optional[tuple] = None,
                         forced_permanent: Optional[Sequence[int]] = None) -> ResourcePlan:
        """click_frame: (height, width) of the frames clicks are made on, to keep room for one
        while the job holds its memory. forced_permanent: the anchors of an earlier attempt of the
        same job (a replay keeps its configuration; it may only wait or be rejected)."""
        q = policies.memory_tokens_per_frame(request.height, request.width,
                                             request.max_internal_size,
                                             request.upscale_to_internal_size)
        other = policies.other_memory_tokens(q, request.max_mem_frames, request.use_long_term,
                                             request.max_long_tokens)
        quality_limit = policies.permanent_anchor_limit(q, other, self.policy.max_affinity_elements)
        want = min(quality_limit, len(request.anchor_tis))
        floor = min(1, want)  # like the original: at least one anchor when there is one
        budget = self.budget()
        reserve = 0
        if self.policy.reserve_for_clicks and click_frame is not None:
            reserve = self.click_requirement(*click_frame)
        room = budget.available - reserve
        ceiling = budget.ceiling - reserve

        def need(anchors: int) -> int:
            tokens = anchors * q + other
            return int(self.policy.estimate_margin * self.model.propagation_peak(q, tokens))

        memory_limit = None
        anchors = want
        if forced_permanent is not None:
            anchors = len(forced_permanent)
        elif need(want) > room and self.policy.allow_fewer_permanent_anchors:
            # the largest count that fits (need grows with the count); without permission the
            # job keeps every anchor and waits for memory instead
            low, high = floor, want
            while low < high:
                middle = (low + high + 1) // 2
                if need(middle) <= room:
                    low = middle
                else:
                    high = middle - 1
            anchors = low
            memory_limit = anchors
        estimate = need(anchors)
        if forced_permanent is not None:
            selected = sorted(forced_permanent)
        else:
            selected = policies.select_permanent_anchors(request.anchor_tis, request.start_ti,
                                                         anchors)
        limited_by = None
        if anchors < want:
            limited_by = 'memory'
        elif want < len(request.anchor_tis):
            limited_by = 'quality_cap'
        common = dict(permanent_tis=selected, estimate_bytes=estimate, query_tokens=q,
                      memory_tokens=anchors * q + other, quality_anchor_limit=quality_limit,
                      memory_anchor_limit=memory_limit, limited_by=limited_by,
                      budget=dict(budget.as_dict(), click_reserve=reserve),
                      model_measured=self.model.measured)
        if estimate <= room:
            return ResourcePlan('run', 'fits', **common)
        if estimate > ceiling:
            return ResourcePlan('reject', (
                f'needs ~{estimate / 2**30:.2f} GiB of GPU memory, more than this GPU can give '
                f'the worker ({max(0, ceiling) / 2**30:.2f} GiB)'), **common)
        return ResourcePlan('queue', (
            f'needs ~{estimate / 2**30:.2f} GiB of GPU memory, {max(0, room) / 2**30:.2f} GiB '
            'is free now'), **common)

    def step_fits(self, query_tokens: int, memory_tokens: int) -> bool:
        """Whether the next step of a running job still fits (memory used by other processes may
        have grown since admission); the resident part is already allocated."""
        need = self.policy.estimate_margin * self.model.propagation_step(query_tokens, memory_tokens)
        return need <= self.budget().available + self.headroom(self.probe.snapshot().total) / 2


def fit_line(points: Sequence[tuple]) -> tuple:
    """Least-squares (intercept, slope) through (x, y) points."""
    n = len(points)
    if n == 1:
        return points[0][1], 0.0
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    variance = sum((x - mean_x)**2 for x, _ in points)
    if variance == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return mean_y - slope * mean_x, slope
