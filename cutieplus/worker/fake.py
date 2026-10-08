"""
A CPU stand-in for the GPU worker's engine, for tests of the coordinator, the scheduler and the
recovery: the same commands and replies, deterministic masks computed with numpy, a simulated GPU
(the real budget logic on a SimulatedMemoryProbe), configurable step times, and test commands
that inject an out-of-memory error, memory pressure from another process, or a worker crash.
"""
import os
import time
from typing import Dict, Optional

import numpy as np

from cutieplus.engine.images import read_mask_source, read_rgb
from cutieplus.engine.policies import frames_after_start
from cutieplus.gpu.budget import GPUResourceManager, MemoryModel, MemoryRequest, ResourcePolicy
from cutieplus.gpu.hardware import SimulatedMemoryProbe
from cutieplus.worker.engine import CommandError

GiB = 1 << 30


def click_mask(initial: np.ndarray, clicks, target: int) -> np.ndarray:
    """A disk added (positive) or removed (negative) at every click, in order."""
    mask = initial.copy()
    height, width = mask.shape
    radius = max(3, min(height, width) // 6)
    yy, xx = np.mgrid[0:height, 0:width]
    for x, y, is_negative in clicks:
        disk = (yy - y)**2 + (xx - x)**2 <= radius**2
        if is_negative:
            mask[disk & (mask == target)] = 0
        else:
            mask[disk] = target
    return mask


class FakeWorkerEngine:

    def __init__(self, cfg, service: dict):
        fake = dict(service.get('fake') or {})
        self.step_seconds = float(fake.get('step_seconds', 0.0))
        self.click_seconds = float(fake.get('click_seconds', 0.0))
        self.model_bytes = int(float(fake.get('model_gib', 0.6)) * GiB)
        self.probe = SimulatedMemoryProbe(total=int(float(fake.get('total_gib', 8)) * GiB),
                                          used_by_others=int(float(fake.get('others_gib', 0)) * GiB),
                                          allocated=self.model_bytes, reserved=self.model_bytes)
        self.memory_model = MemoryModel(True, 'fake GPU', 'fake-uuid', self.model_bytes,
                                        ritm_fixed=256 << 20, ritm_per_pixel=100.0,
                                        resident_fixed=32 << 20, resident_per_query=16 * 1024.0,
                                        resident_per_memory_token=2048.0, step_fixed=64 << 20,
                                        step_per_query=32 * 1024.0, step_per_product=20.0)
        self.policy = ResourcePolicy.from_config(service.get('gpu'))
        self.resources = GPUResourceManager(self.probe, self.policy, self.memory_model)
        self.interactions: Dict[str, dict] = {}
        self.job: Optional[dict] = None
        self.inject: Dict[str, object] = {}
        self.allow_test_commands = True

    def hello(self) -> dict:
        return {
            'hardware': {'backend': 'fake', 'device': 'fake', 'name': 'fake GPU',
                         'uuid': 'fake-uuid', 'compute_capability': [0, 0],
                         'total_bytes': self.probe.total, 'free_bytes': self.probe.snapshot().free_device,
                         'precisions': {'float32': True, 'float16': True}},
            'memory_model': self.memory_model.as_dict(),
            'model_identity': {'cutie_sha256': 'fake', 'ritm_sha256': 'fake'},
            'model_bytes': self.model_bytes,
            'load_seconds': 0.0,
            'policy': vars(self.policy),
            'quality_changes': [],
            'amp': True,
            'pid': os.getpid(),
        }

    def status(self) -> dict:
        return {'memory': self.probe.snapshot().as_dict(), 'budget': self.resources.budget().as_dict(),
                'peak_allocated': self.probe.peak_allocated(), 'peak_reserved': self.probe.peak_reserved(),
                'interactions': len(self.interactions),
                'job': None if self.job is None else {'job_id': self.job['job_id'],
                                                      'attempt_id': self.job['attempt_id'],
                                                      'last_ti': self.job['last_ti']}}

    # clicks
    def ritm_open(self, interaction_id, image_path, mask, num_objects, target_object,
                  clicks=()) -> dict:
        if interaction_id in self.interactions:
            raise CommandError('invalid', f'interaction {interaction_id} is already open')
        image = read_rgb(image_path)
        mask = read_mask_source(mask, image.shape[:2])
        self.interactions[interaction_id] = {'initial': mask, 'clicks': [tuple(c) for c in clicks],
                                             'target': int(target_object), 'live': False}
        return {}

    def _reply(self, interaction: dict, reconstruction) -> dict:
        return {'mask': click_mask(interaction['initial'], interaction['clicks'], interaction['target']),
                'clicks': len(interaction['clicks']), 'reconstruction': reconstruction,
                'reconstruction_seconds': 0.0, 'inference_seconds': self.click_seconds}

    def _activate(self, interaction_id: str):
        if interaction_id not in self.interactions:
            raise CommandError('invalid', f'interaction {interaction_id} is not open')
        if self.inject.pop('click_oom', None):
            raise CommandError('oom', 'injected out-of-memory error during a click')
        for other_id, other in self.interactions.items():
            if other_id != interaction_id:
                other['live'] = False
        interaction = self.interactions[interaction_id]
        reconstruction = None if interaction['live'] or not interaction['clicks'] else 'replay'
        interaction['live'] = True
        time.sleep(self.click_seconds)
        return interaction, reconstruction

    def ritm_click(self, interaction_id, x, y, is_negative) -> dict:
        interaction, reconstruction = self._activate(interaction_id)
        interaction['clicks'].append((int(x), int(y), bool(is_negative)))
        return self._reply(interaction, reconstruction)

    def ritm_undo(self, interaction_id) -> dict:
        interaction, reconstruction = self._activate(interaction_id)
        if len(interaction['clicks']) <= 1:
            return {'mask': None}
        interaction['clicks'].pop()
        return self._reply(interaction, reconstruction)

    def ritm_replay(self, interaction_id) -> dict:
        interaction, _ = self._activate(interaction_id)
        return self._reply(interaction, 'replay')

    def ritm_close(self, interaction_id) -> dict:
        self.interactions.pop(interaction_id, None)
        return {}

    # propagation
    def prop_start(self, job: dict, plan_only: bool = False) -> dict:
        if self.job is not None:
            raise CommandError('invalid', f'propagation context {self.job["job_id"]} is still active')
        anchor_tis = sorted(int(ti) for ti in job['anchor_masks'])
        memory = job['memory']
        resolution = job['resolution']
        request = MemoryRequest(int(job['height']), int(job['width']),
                                int(resolution['max_internal_size']),
                                bool(resolution['upscale_to_internal_size']),
                                int(memory['max_mem_frames']) - 1, True, int(memory['max_num_tokens']),
                                anchor_tis, int(job['start_ti']))
        forced = job.get('permanent_tis')
        plan = self.resources.plan_propagation(request, job.get('click_frame'),
                                               None if forced is None else [int(t) for t in forced])
        if plan.decision != 'run' or plan_only:
            return {'decision': plan.decision, 'plan': plan.as_dict()}
        resident = self.memory_model.propagation_resident(plan.query_tokens, plan.memory_tokens)
        self.probe.allocated += resident
        self.probe.reserved = max(self.probe.reserved, self.probe.allocated)
        seed = read_mask_source(job.get('seed_mask'), (int(job['height']), int(job['width'])))
        bounds = job.get('bounds')
        frames = frames_after_start(int(job['start_ti']), job['direction'], len(job['frame_names']),
                                    None if bounds is None else tuple(bounds))
        self.job = {'job_id': job['job_id'], 'attempt_id': job['attempt_id'], 'job': job,
                    'phase': 'anchors', 'permanent': list(plan.permanent_tis), 'cursor': 0,
                    'frames': frames, 'frame_cursor': 0, 'prev': seed, 'resident': resident,
                    'last_ti': None}
        return {'decision': 'run', 'plan': plan.as_dict()}

    def prop_step(self, job_id, attempt_id) -> dict:
        state = self.job
        if state is None or state['job_id'] != job_id or state['attempt_id'] != attempt_id:
            raise CommandError('invalid', f'no propagation context for {job_id}/{attempt_id}')
        time.sleep(self.step_seconds)
        job = state['job']
        if state['phase'] == 'anchors':
            if state['cursor'] < len(state['permanent']):
                ti = state['permanent'][state['cursor']]
                state['cursor'] += 1
                return self._step_reply('anchor_memory', ti)
            state['phase'] = 'seed'
        if state['phase'] == 'seed':
            state['phase'] = 'frames'
            state['last_ti'] = int(job['start_ti'])
            finished = not state['frames']
            if finished:
                self.prop_release(job_id)
            return self._step_reply('seed', int(job['start_ti']), finished=finished,
                                    stop_reason='end' if finished else None)
        ti = state['frames'][state['frame_cursor']]
        state['frame_cursor'] += 1
        crash_ti = self.inject.get('crash_on_ti')
        if crash_ti is not None and crash_ti == ti:
            os._exit(3)  # a worker that dies mid-propagation
        if self.inject.get('pressure_on_ti') == ti and self.inject.get('pressure_times', 0) > 0:
            # memory short before this step: it is not taken, the context stays
            self.inject['pressure_times'] -= 1
            state['frame_cursor'] -= 1
            raise CommandError('pressure', f'injected memory pressure at frame {ti}')
        if self.inject.get('oom_on_ti') == ti:
            self.inject.pop('oom_on_ti')
            self.prop_release(job_id)
            raise CommandError('oom', f'injected out-of-memory error at frame {ti}')
        name = job['frame_names'][ti]
        kind = job['anchors'].get(name)
        last = state['frame_cursor'] >= len(state['frames'])
        if kind == 'empty':
            self.prop_release(job_id)
            return self._step_reply('stop', ti, finished=True, stop_reason='empty_anchor')
        state['last_ti'] = ti
        if kind == 'mask':
            state['prev'] = read_mask_source(job['anchor_masks'][str(ti)])
            reply = self._step_reply('anchor', ti, finished=last, stop_reason='end' if last else None)
        else:
            state['prev'] = np.roll(state['prev'], 1, axis=1)
            reply = self._step_reply('frame', ti, mask=state['prev'].copy(), finished=last,
                                     stop_reason='end' if last else None)
        if last:
            self.prop_release(job_id)
        return reply

    def _step_reply(self, kind, ti, mask=None, finished=False, stop_reason=None) -> dict:
        return {'kind': kind, 'ti': ti, 'mask': mask, 'finished': finished,
                'stop_reason': stop_reason, 'messages': [], 'inference_seconds': self.step_seconds}

    def prop_release(self, job_id) -> dict:
        if self.job is not None and self.job['job_id'] == job_id:
            self.probe.allocated -= self.job['resident']
            self.job = None
        return {}

    def relieve_memory(self) -> list:
        self.probe.release_cache()
        return ['released cached GPU blocks']

    # test commands
    def test_inject(self, **failures) -> dict:
        self.inject.update(failures)
        return {}

    def test_set_memory(self, others_gib: float) -> dict:
        self.probe.used_by_others = int(others_gib * GiB)
        return {}

    def test_crash(self) -> dict:
        os._exit(3)

    def close(self) -> None:
        self.job = None
        self.interactions.clear()
