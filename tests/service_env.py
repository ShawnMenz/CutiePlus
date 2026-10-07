"""A coordinator with its socket server for the tests, in this process; the worker is spawned."""
import os
import shutil
import tempfile
import time
from os import path
from typing import List, Optional

from cutieplus.client import ServiceClient
from cutieplus.coordinator.server import Server
from cutieplus.coordinator.service import Service
from tests.conftest import compose_gui_config


def fake_overrides(**fake) -> dict:
    return {'engine': 'fake', 'allow_test_commands': True, 'fake': fake}


class ServiceEnv:

    def __init__(self, task_root: str, state_dir: str, overrides: Optional[dict] = None,
                 device: str = 'cpu'):
        # a short socket path: Unix socket paths are limited to 108 bytes
        self.socket_dir = tempfile.mkdtemp(prefix='cps')
        settings = {'socket': path.join(self.socket_dir, 's.sock'), 'heartbeat_seconds': 0.5,
                    'session_timeout_seconds': 30, 'request_timeout_seconds': 60}
        settings.update(overrides or {})
        self.task_root = task_root
        self.state_dir = state_dir
        self.settings = settings
        self.cfg = compose_gui_config(device=device)
        self.clients: List[ServiceClient] = []
        self.service: Optional[Service] = None
        self.server: Optional[Server] = None
        self.start()

    def start(self) -> None:
        self.service = Service(self.cfg, self.task_root, self.state_dir, self.settings)
        self.service.start()
        self.server = Server(self.service, self.service.socket_path)
        self.server.start()

    def stop_service(self) -> None:
        if self.server is not None:
            self.server.stop()
        if self.service is not None:
            self.service.stop()
        self.server = self.service = None

    def client(self, name: str, display: str, heartbeat: bool = True) -> ServiceClient:
        client = ServiceClient(self.settings['socket'], name, display, request_timeout=60,
                               reconnect_seconds=10)
        client.connect()
        if heartbeat:
            client.start_heartbeat()
        self.clients.append(client)
        return client

    def close(self) -> None:
        for client in self.clients:
            client.stopping.set()
            try:
                client.close()
            except Exception:
                pass
        self.stop_service()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


def wait_for(condition, timeout: float = 30.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError('timed out waiting')


def run_propagation(client, task_id, epoch, start_frame, direction='forward', memory=None,
                    timeout=60.0) -> dict:
    memory = memory or {'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4, 'max_mem_frames': 9,
                        'max_num_tokens': 10000}
    job = client.call('start_propagation', task_id=task_id, lease_epoch=epoch,
                      start_frame=start_frame, direction=direction, memory=memory,
                      request_id=os.urandom(8).hex())['job_id']
    frames, since, messages = [], 0, []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reply = client.call('poll_propagation', job_id=job, since=since,
                            messages_since=len(messages))
        frames += reply['frames']
        messages += reply['messages']
        since = reply['next']
        if reply['state'] in ('finished', 'cancelled', 'failed'):
            return dict(reply, job_id=job, all_frames=frames, all_messages=messages)
        time.sleep(0.02)
    raise AssertionError('propagation did not finish')
