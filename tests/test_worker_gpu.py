"""
The real GPU worker (spawned, CUDA, both models): hardware detection, the memory profile, and
results equal to the in-process engine. Run with CUTIEPLUS_TEST_GPU=1 pytest -m gpu.
"""
from os import path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from cutie.inference.inference_core import InferenceCore
from cutieplus.coordinator.worker_proxy import WorkerProxy
from cutieplus.engine.images import read_index_mask, read_rgb
from cutieplus.engine.propagation import PropagationPlan, PropagationRuntime
from cutieplus.engine.ritm import RitmEngine
from gui.interactive_utils import index_numpy_to_one_hot_torch
from tests.conftest import compose_gui_config, networks_on
from tests.helpers import make_task
from tests.test_worker_process import job_for, run_job

pytestmark = [pytest.mark.gpu, pytest.mark.models, pytest.mark.slow]

# live annotators share the lab GPU: the test worker may hold at most this much
SERVICE = {'device': 'cuda', 'gpu': {'max_worker_gib': 8}, 'profile_frame_size': [540, 960]}


@pytest.fixture(scope='module')
def worker():
    proxy = WorkerProxy(OmegaConf.to_container(compose_gui_config(device='cuda'), resolve=True),
                        dict(SERVICE))
    proxy.start(timeout=600)
    yield proxy
    proxy.stop()


def test_hello_reports_the_device_and_a_measured_profile(worker):
    hello = worker.hello
    hardware = hello['hardware']
    assert hardware['backend'] == 'cuda'
    assert hardware['name'] == torch.cuda.get_device_name(0)
    assert hardware['total_bytes'] == torch.cuda.get_device_properties(0).total_memory
    assert tuple(hardware['compute_capability']) == torch.cuda.get_device_capability(0)
    model = hello['memory_model']
    assert model['measured'] is True
    assert model['model_bytes'] > 100 << 20
    assert model['step_per_product'] > 0 and model['ritm_per_pixel'] >= 0
    # the configured mixed precision is what the worker computes in (it once was not: an
    # explicit 'cuda:0' turned autocast off)
    assert hello['amp'] is True and hello['compute_dtype'] == 'torch.float16'
    kinds = {sample['kind'] for sample in model['samples']}
    assert kinds == {'ritm_click', 'cutie_run'}
    assert len(hello['model_identity']['cutie_sha256']) == 64


def test_clicks_equal_the_in_process_engine(worker, tmp_path):
    task = make_task(str(tmp_path), frame_count=6, mask_anchors=(2, ))
    image_path = path.join(task.frame_dirs['thermal'], '000003.jpg')
    mask_path = path.join(task.masks_dir, '000003.png')
    clicks = [(50, 48, False), (70, 48, True), (40, 40, False)]
    worker.call('ritm_open', interaction_id='a', image_path=image_path, mask={'path': mask_path},
                num_objects=1, target_object=1)
    remote = [worker.call('ritm_click', interaction_id='a', x=x, y=y, is_negative=n)['mask']
              for x, y, n in clicks]
    remote.append(worker.call('ritm_undo', interaction_id='a')['mask'])
    replayed = worker.call('ritm_replay', interaction_id='a')
    worker.call('ritm_close', interaction_id='a')

    _, click_ctrl = networks_on('cuda')
    engine = RitmEngine(click_ctrl, 'cuda', True)
    engine.open('a', read_rgb(image_path),
                index_numpy_to_one_hot_torch(read_index_mask(mask_path), 2).cuda(), 1)
    local = [engine.click('a', x, y, n).mask for x, y, n in clicks]
    local.append(engine.undo('a').mask)
    for got, want in zip(remote, local):
        assert np.array_equal(got, want)
    assert replayed['reconstruction'] == 'replay' and np.array_equal(replayed['mask'], local[-1])


def test_propagation_equals_the_in_process_runtime_and_frees_its_memory(worker, tmp_path):
    task = make_task(str(tmp_path), frame_count=10, mask_anchors=(0, 6), empty_anchors=(9, ))
    before = worker.call('status')['memory']['allocated']
    job = job_for(task)
    start = worker.call('prop_start', job=job)
    assert start['decision'] == 'run'
    remote = {r['ti']: r['mask'] for r in run_job(worker, job) if r['kind'] == 'frame'}
    after = worker.call('status')
    assert after['job'] is None
    assert after['memory']['allocated'] <= before + (1 << 20)  # the context is gone

    cutie, _ = networks_on('cuda')
    cfg = compose_gui_config(device='cuda')
    names = tuple(task.names)
    plan = PropagationPlan(names, 0, 'forward', job['anchors'],
                           tuple(start['plan']['permanent_tis']), True, 2)
    frames = [read_rgb(p) for p in job['frame_paths']]
    masks = {int(ti): read_index_mask(source['path']) for ti, source in job['anchor_masks'].items()}
    seed = index_numpy_to_one_hot_torch(masks[0], 2).cuda()
    runtime = PropagationRuntime(InferenceCore(cutie, cfg), plan, lambda ti: frames[ti],
                                 lambda ti: masks[ti], seed, 'cuda', True, cfg)
    local = {}
    while not runtime.finished:
        result = runtime.step()
        if result.needs_commit:
            local[result.ti] = result.mask
    assert sorted(remote) == sorted(local) == [1, 2, 3, 4, 5, 7, 8]
    for ti in local:
        assert np.array_equal(remote[ti], local[ti])
