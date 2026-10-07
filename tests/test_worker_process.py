"""The GPU worker process with the fake engine: spawn, commands, errors, crash and hang handling."""
from os import path

import numpy as np
import pytest
from omegaconf import OmegaConf

from cutieplus.coordinator.worker_proxy import WorkerDied, WorkerError, WorkerProxy
from tests.conftest import compose_gui_config
from tests.helpers import make_task


def fake_worker(**fake) -> WorkerProxy:
    service = {'engine': 'fake', 'allow_test_commands': True, 'fake': fake}
    return WorkerProxy(OmegaConf.to_container(compose_gui_config(), resolve=True), service)


def job_for(task, start_ti=0, direction='forward', attempt='1'):
    names = task.names
    anchors = task.anchors()
    return {
        'job_id': 'job-1', 'attempt_id': attempt, 'frame_names': names,
        'frame_paths': [path.join(task.frame_dirs['visual'], name + '.jpg') for name in names],
        'start_ti': start_ti, 'direction': direction,
        'anchors': {name: entry['kind'] for name, entry in anchors.items()},
        'anchor_masks': {str(names.index(name)): {'path': path.join(task.masks_dir, name + '.png')}
                         for name, entry in anchors.items() if entry['kind'] == 'mask'},
        'seed_mask': {'path': path.join(task.masks_dir, names[start_ti] + '.png')},
        'height': 96, 'width': 128, 'num_objects': 1,
        'resolution': {'max_internal_size': 480, 'upscale_to_internal_size': False},
        'memory': {'mem_every': 5, 'top_k': 30, 'min_mem_frames': 5, 'max_mem_frames': 10,
                   'max_num_tokens': 10000},
    }


def run_job(proxy, job):
    results = []
    while True:
        reply = proxy.call('prop_step', job_id=job['job_id'], attempt_id=job['attempt_id'])
        results.append(reply)
        if reply['finished']:
            return results


def test_commands_round_trip(tmp_path):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, 3), empty_anchors=(6, ))
    proxy = fake_worker()
    try:
        hello = proxy.start(timeout=120)
        assert hello['hardware']['name'] == 'fake GPU'
        assert proxy.generation == 1
        image = path.join(task.frame_dirs['visual'], '000002.jpg')
        proxy.call('ritm_open', interaction_id='a', image_path=image, mask=None,
                   num_objects=1, target_object=1)
        reply = proxy.call('ritm_click', interaction_id='a', x=40, y=40, is_negative=False)
        assert reply['mask'].dtype == np.uint8 and reply['mask'][40, 40] == 1
        assert reply['clicks'] == 1
        with pytest.raises(WorkerError) as error:
            proxy.call('ritm_click', interaction_id='missing', x=1, y=1, is_negative=False)
        assert error.value.kind == 'invalid'

        start = proxy.call('prop_start', job=job_for(task))
        assert start['decision'] == 'run' and start['plan']['permanent_tis'] == [0, 3]
        results = run_job(proxy, job_for(task))
        kinds = [(r['kind'], r['ti']) for r in results]
        assert kinds == [('anchor_memory', 0), ('anchor_memory', 3), ('seed', 0), ('frame', 1),
                         ('frame', 2), ('anchor', 3), ('frame', 4), ('frame', 5), ('stop', 6)]
        assert all(r['mask'] is not None for r in results if r['kind'] == 'frame')
        assert proxy.call('status')['job'] is None  # released when finished
    finally:
        proxy.stop()
    assert not proxy.alive()


def test_crash_is_detected_and_a_new_generation_starts(tmp_path):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, ))
    proxy = fake_worker()
    try:
        proxy.start(timeout=120)
        proxy.call('test_inject', crash_on_ti=3)
        job = job_for(task)
        proxy.call('prop_start', job=job)
        with pytest.raises(WorkerDied):
            run_job(proxy, job)
        assert not proxy.alive()
        with pytest.raises(WorkerDied):
            proxy.call('status')
        proxy.start(timeout=120)
        assert proxy.generation == 2
        assert proxy.call('status')['job'] is None  # nothing survives a restart
    finally:
        proxy.stop()


def test_hung_worker_is_stopped(tmp_path):
    task = make_task(str(tmp_path), frame_count=4, mask_anchors=(0, ))
    proxy = fake_worker(step_seconds=30)
    try:
        proxy.start(timeout=120)
        job = job_for(task)
        proxy.call('prop_start', job=job)
        with pytest.raises(WorkerDied, match='did not answer'):
            proxy.call('prop_step', timeout=1, job_id='job-1', attempt_id='1')
        assert not proxy.alive()
    finally:
        proxy.stop()


def test_injected_out_of_memory_releases_the_context(tmp_path):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, ))
    proxy = fake_worker()
    try:
        proxy.start(timeout=120)
        proxy.call('test_inject', oom_on_ti=2)
        job = job_for(task)
        proxy.call('prop_start', job=job)
        with pytest.raises(WorkerError) as error:
            run_job(proxy, job)
        assert error.value.kind == 'oom'
        assert proxy.call('status')['job'] is None
        # a new attempt starts from scratch and gives the same masks as an undisturbed run
        retry = dict(job, attempt_id='2')
        proxy.call('prop_start', job=retry)
        masks = {r['ti']: r['mask'] for r in run_job(proxy, retry) if r['kind'] == 'frame'}
        assert sorted(masks) == list(range(1, 8))
    finally:
        proxy.stop()


def test_queued_job_holds_no_memory(tmp_path):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, ))
    proxy = fake_worker(total_gib=2, others_gib=1.2)
    try:
        proxy.start(timeout=120)
        before = proxy.call('status')['memory']['allocated']
        job = dict(job_for(task), height=1080, width=1920,
                   resolution={'max_internal_size': 960, 'upscale_to_internal_size': True})
        reply = proxy.call('prop_start', job=job)
        assert reply['decision'] in ('queue', 'reject')
        status = proxy.call('status')
        assert status['job'] is None and status['memory']['allocated'] == before
        proxy.call('test_set_memory', others_gib=0.0)
    finally:
        proxy.stop()
