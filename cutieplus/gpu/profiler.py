"""
Measures, on the device in use, the GPU memory of the work the worker does, and fits the linear
models of MemoryModel to it. Warm-up runs on synthetic frames of a representative size (memory
does not depend on image content):

- RITM clicks at two frame sizes (peak above the resident level);
- Cutie at the default processing size with few and with more memory tokens, and at the
  small-target size: the state resident between steps, the peak of a permanent-anchor step and
  of a propagation step;
- a click while a Cutie run holds its memory.

Nothing measured on one GPU is assumed for another: the worker profiles at every start.
"""
import gc
import logging
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from cutieplus.engine import policies
from cutieplus.engine.propagation import PropagationPlan, PropagationRuntime, new_inference_core
from cutieplus.engine.ritm import RitmEngine
from cutieplus.gpu.budget import MemoryModel, fit_line
from cutieplus.gpu.hardware import HardwareInfo, MemoryProbe

log = logging.getLogger(__name__)


def memory_tokens(processor) -> int:
    """Memory tokens a step of this InferenceCore reads (working + long-term)."""
    memory = processor.memory
    total = 0
    for bucket_id in memory.work_mem.buckets:
        total += memory.work_mem.size(bucket_id)
        if memory.use_long_term and memory.long_mem.engaged(bucket_id):
            total += memory.long_mem.size(bucket_id)
    return total


class Profiler:

    def __init__(self, cutie, click_controller, cfg, device: str, amp: bool, probe: MemoryProbe,
                 frame_size: Tuple[int, int] = (1080, 1920), seed: int = 0):
        self.cutie = cutie
        self.click_controller = click_controller
        self.cfg = cfg
        self.device = device
        self.amp = amp
        self.probe = probe
        self.height, self.width = frame_size
        self.rng = np.random.default_rng(seed)
        self.samples: List[dict] = []

    def _frame(self, height: int, width: int) -> np.ndarray:
        return self.rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    def _settle(self) -> int:
        gc.collect()
        self.probe.synchronize()
        self.probe.release_cache()
        return self.probe.snapshot().allocated

    def ritm_peak(self, height: int, width: int) -> int:
        engine = RitmEngine(self.click_controller, self.device, self.amp)
        image = self._frame(height, width)
        prev = torch.zeros((2, height, width))
        prev[0] = 1
        before = self._settle()
        self.probe.reset_peak()
        engine.open('profile', image, prev, 1)
        engine.click('profile', width // 2, height // 2, False)
        # the second click works on a zoomed-in crop
        engine.click('profile', width // 2 + width // 20, height // 2, True)
        self.probe.synchronize()
        peak = self.probe.peak_allocated() - before
        engine.close_all()
        del engine
        self._settle()
        return peak

    def cutie_run(self, internal_size: int, anchors: int, click_during: bool = False) -> dict:
        """Resident state and peaks of a run with `anchors` permanent anchors."""
        height, width = self.height, self.width
        frames = [self._frame(height, width) for _ in range(anchors + 3)]
        mask = np.zeros((height, width), np.uint8)
        mask[height // 3:2 * height // 3, width // 3:2 * width // 3] = 1
        names = tuple(f'{ti:06d}' for ti in range(len(frames)))
        plan = PropagationPlan(names, anchors, 'forward', {names[ti]: 'mask' for ti in range(anchors)},
                               tuple(range(anchors)), True, anchors)
        before = self._settle()
        processor = new_inference_core(self.cutie, self.cfg, internal_size, True)
        seed = torch.zeros((2, height, width), device=self.device)
        seed[1] = torch.from_numpy(mask).to(self.device)
        seed[0] = 1 - seed[1]
        runtime = PropagationRuntime(processor, plan, lambda ti: frames[ti], lambda ti: mask, seed,
                                     self.device, self.amp, self.cfg)
        self.probe.reset_peak()
        for _ in range(anchors):
            runtime.step()
        self.probe.synchronize()
        anchor_peak = self.probe.peak_allocated() - before
        runtime.step()  # the start frame
        self.probe.synchronize()
        resident = self.probe.snapshot().allocated - before
        tokens = memory_tokens(processor)
        resident_before_step = self.probe.snapshot().allocated
        self.probe.reset_peak()
        runtime.step()
        self.probe.synchronize()
        step_peak = self.probe.peak_allocated() - resident_before_step
        query = policies.memory_tokens_per_frame(height, width, internal_size, True)
        result = {'internal_size': internal_size, 'anchors': anchors, 'query_tokens': query,
                  'memory_tokens': tokens, 'resident': resident, 'step_peak': step_peak,
                  'anchor_init_peak': anchor_peak}
        if click_during:
            # a click while the run holds its memory: the peak above the run's resident state
            result['click_with_run_peak'] = self.ritm_peak(height, width)
        del runtime, processor
        self._settle()
        return result

    def run(self, hardware: HardwareInfo, model_bytes: int,
            report: Optional[Callable[[str], None]] = None) -> MemoryModel:
        start = time.perf_counter()
        report = report or log.info
        default_size = int(self.cfg.max_internal_size)
        small_size = policies.SMALL_TARGET_INTERNAL_SIZE

        ritm_points = []
        for height, width in ((self.height, self.width), (self.height // 2, self.width // 2)):
            peak = self.ritm_peak(height, width)
            ritm_points.append((height * width, peak))
            self.samples.append({'kind': 'ritm_click', 'height': height, 'width': width,
                                 'peak': peak})
        ritm_fixed, ritm_per_pixel = fit_line(ritm_points)

        a = self.cutie_run(default_size, 1)
        b = self.cutie_run(default_size, 4, click_during=True)
        c = self.cutie_run(small_size, 1)
        for sample in (a, b, c):
            self.samples.append(dict(sample, kind='cutie_run'))

        # resident = r0 + rq * Q + rn * N
        rn = max(0.0, (b['resident'] - a['resident']) / max(1, b['memory_tokens'] - a['memory_tokens']))
        rq = max(0.0, (c['resident'] - a['resident'] - rn * (c['memory_tokens'] - a['memory_tokens']))
                 / max(1, c['query_tokens'] - a['query_tokens']))
        r0 = a['resident'] - rq * a['query_tokens'] - rn * a['memory_tokens']
        # step = s0 + sq * Q + sp * Q * N
        products = [s['query_tokens'] * s['memory_tokens'] for s in (a, b, c)]
        sp = max(0.0, (b['step_peak'] - a['step_peak']) / max(1, products[1] - products[0]))
        sq = max(0.0, (c['step_peak'] - a['step_peak'] - sp * (products[2] - products[0]))
                 / max(1, c['query_tokens'] - a['query_tokens']))
        s0 = a['step_peak'] - sq * a['query_tokens'] - sp * products[0]
        # a permanent-anchor step is a step too: cover the larger of the two peaks
        model = MemoryModel(True, hardware.name, hardware.uuid, model_bytes, max(0.0, ritm_fixed),
                            ritm_per_pixel, max(0.0, r0), rq, rn, max(0.0, s0), sq, sp,
                            self.samples)
        # never below a measurement: raise the intercepts by the largest shortfall
        for sample in (a, b, c):
            q, n = sample['query_tokens'], sample['memory_tokens']
            model.resident_fixed += max(0, sample['resident'] - model.propagation_resident(q, n))
            step_need = max(sample['step_peak'], sample['anchor_init_peak'] - sample['resident'])
            model.step_fixed += max(0, step_need - model.propagation_step(q, n))
        for pixels, peak in ritm_points:
            model.ritm_fixed += max(0, peak - (model.ritm_fixed + model.ritm_per_pixel * pixels))
        report(f'GPU memory profile on {hardware.name} in {time.perf_counter() - start:.1f} s: '
               f'click ~{model.ritm_peak(self.height, self.width) / 2**20:.0f} MiB at '
               f'{self.width}x{self.height}; propagation at {default_size} px '
               f'~{model.propagation_peak(a["query_tokens"], a["memory_tokens"]) / 2**20:.0f} MiB '
               f'({a["memory_tokens"]} memory tokens)')
        return model
