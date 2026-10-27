"""
The cooperation service with its GPU queue in the test process, and a real GPU worker process
(python -m cutieplus.gpu_worker --engine fake) on real sockets: the fake engine answers the same
commands as the real one, without models, and logs when each command ran.
"""
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from cutieplus.cooperation.config import ServiceConfig
from tests.cooperation_harness import Harness

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEIGHT, WIDTH = 24, 32
MEMORY = {'mem_every': 5, 'top_k': 30, 'min_mem_frames': 5, 'max_mem_frames': 10,
          'max_num_tokens': 10000}


class GpuHarness(Harness):

    def __init__(self, tmp_path, frames: int = 12, step_seconds: float = 0.0, **config):
        workspace = tmp_path / 'workspace'
        workspace.mkdir(exist_ok=True)
        self.lock_file = tmp_path / 'gpu.lock'
        self.lock_file.write_text('')
        super().__init__(tmp_path, worker_socket=str(tmp_path / 'worker.sock'),
                         gpu_lock=str(self.lock_file), **config)
        from cutieplus.cooperation.gpu import GpuQueue
        self.service.gpu = GpuQueue(self.service)
        self.service.gpu.start()
        self.tmp_path = tmp_path
        self.log_path = str(tmp_path / 'gpu.log')
        self.step_seconds = step_seconds
        self.config_file = tmp_path / 'service.toml'
        self.config_file.write_text(
            f'state_dir = "{self.config.state_dir}"\nworkspace_root = "{workspace}"\n'
            f'worker_socket = "{self.config.worker_socket}"\ngpu_lock = "{self.lock_file}"\n'
            'verify_peers = false\nrequire_patched_sqlite = false\n')
        # a sequence of frames in the workspace
        self.task = str(workspace / 'cutie_output' / 'main' / '0001')
        frames_dir = workspace / 'cutie_tasks' / 'main' / 'visual' / '0001'
        frames_dir.mkdir(parents=True)
        self.frame_names, self.frame_paths = [], []
        for index in range(frames):
            name = f'{index + 1:06d}'
            file_path = frames_dir / f'{name}.png'
            Image.fromarray(np.full((HEIGHT, WIDTH, 3), index * 7 % 256, np.uint8)).save(file_path)
            self.frame_names.append(name)
            self.frame_paths.append(str(file_path))
        self.workers: List[subprocess.Popen] = []

    def start_worker(self, wait: bool = True, engine: str = 'fake', timeout: float = 30,
                     **fake) -> subprocess.Popen:
        env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
        env.update(CUTIEPLUS_FAKE_LOG=self.log_path,
                   CUTIEPLUS_FAKE_STEP_S=str(self.step_seconds))
        for key, value in fake.items():
            env[f'CUTIEPLUS_FAKE_{key.upper()}'] = str(value)
        generation = self.service.gpu.link.generation
        process = subprocess.Popen([sys.executable, '-m', 'cutieplus.gpu_worker', '--engine',
                                    engine, '--config', str(self.config_file)], cwd=REPO,
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.workers.append(process)
        if wait:
            self.wait_for(lambda: self.service.gpu.link.generation > generation
                          and self.service.gpu.link.ready, timeout)
        return process

    @staticmethod
    def wait_for(predicate, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError('the condition never held')
            time.sleep(0.02)

    def snapshot(self, start: int = 0, direction: str = 'forward', anchors: Optional[dict] = None,
                 anchor_masks: Optional[dict] = None) -> dict:
        seed = np.zeros((2, HEIGHT, WIDTH), np.float16)
        seed[0] = 1
        seed[0, 4:12, 6:16] = 0
        seed[1, 4:12, 6:16] = 1
        return {'frame_names': list(self.frame_names), 'frame_paths': list(self.frame_paths),
                'start_ti': start, 'direction': direction, 'anchors': anchors or {},
                'anchor_masks': anchor_masks or {}, 'height': HEIGHT, 'width': WIDTH,
                'num_objects': 1, 'seed_prob': seed, 'bounds': None,
                'resolution': {'max_internal_size': 480, 'upscale_to_internal_size': False},
                'memory': dict(MEMORY)}

    def submit(self, token: str, key: str, revision: int = 1, **snapshot) -> dict:
        return self.call('submit_propagation', token, idempotency_key=key, workspace=self.task,
                         revision=revision, snapshot=self.snapshot(**snapshot))

    def job(self, token: str, job_id: str) -> dict:
        return self.call('job_events', token, job_id=job_id)

    def wait_job(self, token: str, job_id: str, statuses=('succeeded', 'failed', 'cancelled',
                                                          'interrupted'), timeout=30) -> dict:
        self.wait_for(lambda: self.job(token, job_id)['status'] in statuses, timeout)
        return self.job(token, job_id)

    def log(self) -> List[dict]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def close(self) -> None:
        for process in self.workers:
            if process.poll() is None:
                process.kill()
            process.wait(10)
        self.service.gpu.stop()


def no_overlap(entries: List[dict]) -> bool:
    ordered = sorted(entries, key=lambda entry: entry['start'])
    return all(earlier['end'] <= later['start'] for earlier, later in zip(ordered, ordered[1:]))


def worker_config(tmp_path, harness: GpuHarness) -> ServiceConfig:
    return ServiceConfig.load(str(harness.config_file))
