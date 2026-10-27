"""
A GPU worker engine without models, for the tests: it answers the same commands as TorchEngine
on the CPU. A click paints (or erases) a disk; a propagation carries the start frame's mask to
every frame it reaches, stops at an empty anchor and takes a mask anchor's mask as its own, and
stages real files. It can be slowed down, made to run out of memory or to crash at a step, and it
logs when each command ran (start and end, in seconds), so a test can check that no two GPU
commands ever overlapped, across workers too.
"""
import json
import os
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
from PIL import Image

from cutieplus.engine import policies
from cutieplus.gpu_worker.engine import (CommandError, Engine, Job, inside, mask_from_png,
                                         stage_mask)

PALETTE = [0, 0, 0, 128, 0, 0] + [0] * (768 - 6)
RADIUS = 6


class _Run:

    def __init__(self, order, anchors, anchor_masks, seed):
        self.order = order  # [('seed', ti), ('frame', ti), ...]
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.mask = seed
        self.position = 0


class FakeEngine(Engine):

    def __init__(self, workspace_root: str, jobs_dir: str, step_seconds: float = 0.0,
                 log_path: Optional[str] = None, oom_at_step: Optional[int] = None,
                 crash_at_step: Optional[int] = None):
        super().__init__(workspace_root, jobs_dir)
        self.step_seconds = step_seconds
        self.log_path = log_path
        self.oom_at_step = oom_at_step
        self.crash_at_step = crash_at_step
        self.steps = 0
        self.interactions = {}  # id -> [base mask, clicks]

    @classmethod
    def from_environment(cls, workspace_root: str, jobs_dir: str) -> 'FakeEngine':
        number = lambda name: int(os.environ[name]) if os.environ.get(name) else None
        return cls(workspace_root, jobs_dir, float(os.environ.get('CUTIEPLUS_FAKE_STEP_S', 0)),
                   os.environ.get('CUTIEPLUS_FAKE_LOG'), number('CUTIEPLUS_FAKE_OOM_AT_STEP'),
                   number('CUTIEPLUS_FAKE_CRASH_AT_STEP'))

    @contextmanager
    def running(self, op: str, job: Optional[str] = None):
        start = time.monotonic()
        yield
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'pid': os.getpid(), 'op': op, 'job': job, 'start': start,
                                    'end': time.monotonic()}) + '\n')

    def hello(self) -> dict:
        return {'engine': 'fake', 'device': 'cpu', 'device_name': 'fake', 'amp': False,
                'model': {'cutie_sha256': 'fake', 'ritm_sha256': 'fake'}}

    def status(self) -> dict:
        return dict(self.status_extra(), memory_used_mb=0, memory_total_mb=0)

    def reset(self) -> dict:
        self.job = None
        self.interactions.clear()
        self.interaction_shapes.clear()
        return {}

    # clicks
    def ritm_open(self, interaction_id, image_path, mask_png, num_objects, target_object,
                  clicks=()) -> dict:
        if interaction_id in self.interactions:
            raise CommandError('invalid', f'interaction {interaction_id} is already open')
        width, height = Image.open(inside(image_path, self.workspace_root, 'frame')).size
        self.interactions[interaction_id] = [mask_from_png(mask_png, (height, width)),
                                             [tuple(click) for click in clicks]]
        self.interaction_shapes[interaction_id] = (height, width)
        return {}

    def _mask(self, interaction_id: str) -> np.ndarray:
        base, clicks = self.interactions[interaction_id]
        mask = base.copy()
        rows, columns = np.ogrid[:mask.shape[0], :mask.shape[1]]
        for x, y, negative in clicks:
            disk = (rows - y)**2 + (columns - x)**2 <= RADIUS**2
            mask[disk] = 0 if negative else 1
        return mask

    def _reply(self, interaction_id: str) -> dict:
        mask = self._mask(interaction_id)
        prob = np.stack([mask == 0, mask == 1]).astype(np.float16)
        return {'mask': mask, 'prob': prob, 'clicks': len(self.interactions[interaction_id][1])}

    def ritm_click(self, interaction_id, x, y, is_negative) -> dict:
        if interaction_id not in self.interactions:
            raise CommandError('invalid', f'no interaction {interaction_id}')
        with self.running('click'):
            time.sleep(self.step_seconds / 4)
            self.interactions[interaction_id][1].append((int(x), int(y), bool(is_negative)))
            return self._reply(interaction_id)

    def ritm_undo(self, interaction_id) -> dict:
        if interaction_id not in self.interactions:
            raise CommandError('invalid', f'no interaction {interaction_id}')
        with self.running('undo'):
            clicks = self.interactions[interaction_id][1]
            if not clicks:
                return {'mask': None, 'clicks': 0}
            clicks.pop()
            return self._reply(interaction_id)

    def ritm_close(self, interaction_id) -> dict:
        self.interactions.pop(interaction_id, None)
        self.interaction_shapes.pop(interaction_id, None)
        return {}

    # propagation
    def prop_start(self, job_id, token, snapshot) -> dict:
        _, staging = self.check_snapshot(job_id, token, snapshot)
        names = tuple(snapshot['frame_names'])
        shape = (int(snapshot['height']), int(snapshot['width']))
        start = int(snapshot['start_ti'])
        bounds = snapshot.get('bounds')
        frames = policies.frames_after_start(start, snapshot['direction'], len(names),
                                             None if bounds is None else tuple(bounds))
        seed = np.argmax(np.asarray(snapshot['seed_prob'], np.float32), axis=0).astype(np.uint8)
        masks = {name: mask_from_png(data, shape)
                 for name, data in snapshot['anchor_masks'].items()}
        run = _Run([('seed', start)] + [('frame', ti) for ti in frames],
                   dict(snapshot['anchors']), masks, seed)
        self.job = Job(job_id, token, staging, names, run)
        return {'permanent_anchors': []}

    def prop_step(self, job_id, token) -> dict:
        job = self.active_job(job_id, token)
        run = job.runtime
        self.steps += 1
        if self.crash_at_step is not None and self.steps >= self.crash_at_step:
            os._exit(3)  # a crash: no reply, no cleanup
        with self.running('step', job_id):
            time.sleep(self.step_seconds)
            if self.oom_at_step is not None and self.steps >= self.oom_at_step:
                self.oom_at_step = None
                self.prop_release(job_id, token)
                raise CommandError('oom', 'out of GPU memory in a propagation step (fake)')
            kind, ti = run.order[run.position]
            run.position += 1
            name = job.names[ti]
            reply = {'kind': kind, 'ti': ti, 'name': name, 'messages': [], 'memory': None,
                     'finished': False, 'stop_reason': None}
            anchor = run.anchors.get(name) if kind == 'frame' else None
            if anchor == 'empty':
                reply.update(kind='stop', mask=None, finished=True, stop_reason='empty_anchor')
            elif anchor == 'mask':
                run.mask = run.anchor_masks[name]
                reply.update(kind='anchor', mask=run.mask)
            else:
                reply['mask'] = run.mask
                if kind == 'frame':
                    reply['sha256'] = stage_mask(job.staging, name, run.mask, PALETTE)
            if run.position >= len(run.order) and not reply['finished']:
                reply.update(finished=True, stop_reason='end')
            return reply

    def prop_release(self, job_id, token) -> dict:
        if self.job is not None and self.job.job_id == job_id and self.job.token == token:
            self.job = None
        return {'released': True}
