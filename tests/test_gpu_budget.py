"""GPU detection and memory budgets on simulated GPUs of different sizes (no GPU needed)."""
import pytest

from cutieplus.engine import policies
from cutieplus.gpu import hardware as hw
from cutieplus.gpu.budget import (GPUResourceManager, MemoryModel, MemoryRequest, ResourcePolicy,
                                  fit_line)
from cutieplus.gpu.hardware import MemorySnapshot, SimulatedMemoryProbe

GiB = 1 << 30
MiB = 1 << 20

# a model in the range measured on a real GPU (see the benchmark report)
MODEL = MemoryModel(True, 'simulated', None, int(0.6 * GiB), ritm_fixed=300 * MiB,
                    ritm_per_pixel=150.0, resident_fixed=40 * MiB, resident_per_query=20 * 1024.0,
                    resident_per_memory_token=2000.0, step_fixed=100 * MiB,
                    step_per_query=60 * 1024.0, step_per_product=22.0)


def manager(total_gib, others_gib=0.0, policy=None, allocated_gib=0.6):
    probe = SimulatedMemoryProbe(total=int(total_gib * GiB), used_by_others=int(others_gib * GiB),
                                 allocated=int(allocated_gib * GiB), reserved=int(allocated_gib * GiB))
    return GPUResourceManager(probe, policy or ResourcePolicy(), MODEL), probe


def request(anchors=60, size=480, upscale=False, start=30):
    return MemoryRequest(1080, 1920, size, upscale, 9, True, 10000, list(range(0, anchors * 5, 5)),
                         start)


def test_snapshot_does_not_count_reserved_memory_twice():
    snapshot = MemorySnapshot(total=24 * GiB, free_device=10 * GiB, allocated=3 * GiB,
                              reserved=5 * GiB)
    assert snapshot.cache_slack == 2 * GiB
    assert snapshot.available == 12 * GiB  # free plus the allocator's own cached blocks
    assert snapshot.outside_allocator == 9 * GiB  # other processes and contexts


def test_ample_memory_keeps_the_original_anchor_limit():
    resources, _ = manager(48)
    plan = resources.plan_propagation(request(anchors=200))
    tokens = policies.memory_tokens_per_frame(1080, 1920, 480, False)
    legacy = policies.permanent_anchor_limit(
        tokens, policies.other_memory_tokens(tokens, 9, True, 10000))
    assert plan.decision == 'run'
    assert len(plan.permanent_tis) == legacy == 136
    assert plan.limited_by == 'quality_cap'


def test_moderate_memory_uses_fewer_anchors_nearest_to_the_start():
    resources, _ = manager(6)
    plan = resources.plan_propagation(request(anchors=60, start=150))
    assert plan.decision == 'run'
    assert plan.limited_by == 'memory'
    kept = plan.permanent_tis
    assert 1 <= len(kept) < 60
    # the nearest to frame 150, in time order
    assert kept == sorted(sorted(range(0, 300, 5), key=lambda ti: abs(ti - 150))[:len(kept)])
    assert plan.estimate_bytes <= plan.budget['available']
    assert 'nearest to the start frame' in plan.message(60)


def test_low_memory_queues_and_too_little_rejects():
    resources, probe = manager(24, others_gib=22.5)  # another process holds most of the GPU
    plan = resources.plan_propagation(request(size=960, upscale=True))
    assert plan.decision == 'queue'
    probe.used_by_others = 0  # it exits: the same job fits
    assert resources.plan_propagation(request(size=960, upscale=True)).decision == 'run'
    tiny, _ = manager(1.2)  # smaller than the job could ever get
    assert tiny.plan_propagation(request(size=960, upscale=True)).decision == 'reject'


def test_pressure_from_another_process_is_seen_at_the_next_check():
    resources, probe = manager(12)
    first = resources.plan_propagation(request(anchors=120))
    probe.used_by_others = int(8 * GiB)
    second = resources.plan_propagation(request(anchors=120))
    assert len(second.permanent_tis) < len(first.permanent_tis)
    assert second.budget['snapshot']['free_device'] < first.budget['snapshot']['free_device']


def test_worker_cap_and_headroom():
    policy = ResourcePolicy(max_worker_bytes=2 * GiB, headroom_min_bytes=256 * MiB)
    resources, _ = manager(24, policy=policy, allocated_gib=0.6)
    budget = resources.budget()
    assert budget.available == pytest.approx(1.4 * GiB, rel=1e-6)  # the cap, not the free memory
    assert resources.headroom(24 * GiB) == int(0.05 * 24 * GiB)
    assert resources.headroom(2 * GiB) == 256 * MiB


def test_click_reserve_while_propagating():
    resources, _ = manager(6)
    without = resources.plan_propagation(request(anchors=60))
    with_reserve = resources.plan_propagation(request(anchors=60), click_frame=(1080, 1920))
    assert len(with_reserve.permanent_tis) <= len(without.permanent_tis)
    assert with_reserve.budget['click_reserve'] == resources.click_requirement(1080, 1920)


def test_forced_anchors_wait_instead_of_shrinking():
    resources, probe = manager(6)
    plan = resources.plan_propagation(request(anchors=60))
    probe.used_by_others = int(4 * GiB)
    replay = resources.plan_propagation(request(anchors=60), forced_permanent=plan.permanent_tis)
    assert replay.permanent_tis == plan.permanent_tis
    assert replay.decision == 'queue'


def test_without_fewer_anchors_allowed_the_job_waits():
    resources, _ = manager(6, policy=ResourcePolicy(allow_fewer_permanent_anchors=False))
    plan = resources.plan_propagation(request(anchors=120))
    assert plan.decision in ('queue', 'reject')
    assert plan.limited_by != 'memory'


def test_click_admission():
    resources, probe = manager(4)
    assert resources.admit_click(1080, 1920) is None
    probe.used_by_others = int(3.3 * GiB)
    assert 'click needs' in resources.admit_click(1080, 1920)


def test_no_cuda_device_is_a_clear_error():
    # the CPU test run hides the GPU (tests/conftest.py)
    with pytest.raises(hw.NoGpuError, match='No CUDA device'):
        hw.detect('cuda')
    with pytest.raises(hw.NoGpuError, match='allow_cpu'):
        hw.detect('cpu')
    assert hw.detect('cpu', allow_cpu=True).backend == 'cpu'


def fake_hardware(float16: bool) -> hw.HardwareInfo:
    return hw.HardwareInfo('cuda', 'cuda:0', 0, 'Old GPU', None, (5, 2), 8, 4 * GiB, 4 * GiB, 0, 0,
                           {'float32': True, 'float16': float16, 'bfloat16': False, 'tf32': False},
                           '2.x', '13.0', None, 1)


def test_unsupported_precision():
    assert hw.effective_precision(True, fake_hardware(True), False) == (True, [])
    with pytest.raises(hw.NoGpuError, match='float16'):
        hw.effective_precision(True, fake_hardware(False), False)
    amp, changes = hw.effective_precision(True, fake_hardware(False), True)
    assert amp is False and changes[0]['setting'] == 'amp'
    assert hw.effective_precision(False, fake_hardware(False), False) == (False, [])


def test_policy_from_config():
    policy = ResourcePolicy.from_config({'max_worker_gib': 10, 'headroom_min_mib': 1024,
                                         'estimate_margin': 1.5, 'unknown': 3})
    assert policy.max_worker_bytes == 10 * GiB
    assert policy.headroom_min_bytes == 1024 * MiB
    assert policy.estimate_margin == 1.5


def test_fit_line():
    assert fit_line([(0, 1), (2, 5)]) == (1, 2)
    assert fit_line([(3, 7)]) == (7, 0.0)
