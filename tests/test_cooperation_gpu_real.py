"""
The shared GPU worker with the real models on the GPU (CUTIEPLUS_TEST_GPU=1 pytest -m gpu): a
propagation through the worker gives the masks the window gave when it ran the models itself, and
another user's clicks are answered between its steps rather than after it.
"""
import os
import shutil
import time
from os import path

import numpy as np
import pytest
import torch

from cutieplus.backends.service import ServiceTaskBackend
from cutieplus.cooperation.client import ServiceClient
from cutieplus.cooperation.server import Server
from tests.conftest import compose_gui_config
from tests.cooperation_harness import PASSWORD
from tests.gpu_harness import GpuHarness
from tests.helpers import make_task

pytestmark = [pytest.mark.gpu, pytest.mark.models, pytest.mark.slow]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a > 0, b > 0).sum()
    return 1.0 if union == 0 else np.logical_and(a > 0, b > 0).sum() / union


def propagated(workspace: str, names):
    from PIL import Image
    return {name: np.array(Image.open(path.join(workspace, 'masks', f'{name}.png')))
            for name in names}


def test_the_worker_propagates_as_the_window_did_and_clicks_wait_one_step(tmp_path, qapp):
    gpu = GpuHarness(tmp_path)
    server = Server(gpu.service.handle, str(tmp_path / 'app.sock'))
    server.start()
    try:
        gpu.start_worker(engine='torch', timeout=180)
        hello = gpu.service.gpu.link.hello
        assert hello['engine'] == 'torch' and hello['device'] == 'cuda', hello
        owner = gpu.owner()
        gpu.add(owner, 'alice')
        gpu.add(owner, 'bob')
        clients = {}
        for login in ('alice', 'bob'):
            client = ServiceClient(str(tmp_path / 'app.sock'), timeout=120)
            client.token = client.call('login', login_name=login, password=PASSWORD)['token']
            clients[login] = client
        root = gpu.config.workspace_root
        task = make_task(root, frame_count=60, mask_anchors=(0, ))
        reference = make_task(str(tmp_path / 'local'), frame_count=60, mask_anchors=(0, ))
        from gui.main_controller import MainController, load_networks

        # the window running the models itself, on the GPU
        cfg = compose_gui_config(reference.workspace, device='cuda')
        local = MainController(cfg, networks=load_networks(cfg))
        try:
            local.on_forward_propagation()
        finally:
            local.gui.close()

        # the same task through the shared worker, while bob clicks on another sequence
        other = make_task(root, sequence='0002', frame_count=4, mask_anchors=())
        window = MainController(
            compose_gui_config(task.workspace, labeler='alice'),
            backend_factory=lambda cfg, res_man, t: ServiceTaskBackend(cfg, res_man, t,
                                                                       clients['alice']))
        bob = ServiceClient(str(tmp_path / 'app.sock'), timeout=120)
        bob.token = clients['bob'].token
        opened = bob.call('open_interaction', workspace=other.workspace,
                          image_path=path.join(other.root, 'main', 'visual', '0002',
                                               '000001.jpg'))
        click_seconds = []
        try:
            backend = window.backend
            window.convert_current_image_mask_torch()
            backend.start_propagation(0, 'forward', window.curr_prob, None)
            gpu.wait_for(lambda: backend.queue_state()['status'] == 'running', 120)
            for index in range(5):
                started = time.monotonic()
                bob.call('click', interaction_id=opened['interaction_id'], x=20 + index, y=20,
                         is_negative=False, idempotency_key=f'bob-real-click-{index}')
                click_seconds.append(time.monotonic() - started)
            while True:
                progress = backend.advance_propagation()
                if progress.done:
                    break
        finally:
            window.gui.close()
        names = [f'{index:06d}' for index in range(2, 61)]
        theirs, ours = propagated(reference.workspace, names), propagated(task.workspace, names)
        scores = [iou(theirs[name], ours[name]) for name in names]
        print(f'IoU with the window\'s own run: min {min(scores):.4f}, mean '
              f'{np.mean(scores):.4f}; clicks during the run: '
              f'{", ".join(f"{1000 * seconds:.0f}" for seconds in click_seconds)} ms on '
              f'{hello["device_name"]}')
        assert min(scores) > 0.97, scores
        assert np.median(click_seconds) < 0.5, click_seconds
        log = gpu.db.query("SELECT status FROM gpu_jobs WHERE operation_type = 'click'")
        assert [row['status'] for row in log] == ['succeeded'] * 5
    finally:
        server.stop()
        gpu.close()
