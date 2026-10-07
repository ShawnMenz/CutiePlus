"""
The labeling window in shared mode, driven through the same scripted session as the baseline,
against a coordinator whose GPU worker runs the real models (on the CPU here; on the GPU in the
GPU run): the results are compared with those of the original implementation.
"""
import os
from os import path

import numpy as np
import pytest

from tests.conftest import compose_gui_config
from tests.scenarios import flatten, run_session, scenario_task
from tests.service_env import ServiceEnv

DATA = path.join(path.dirname(path.abspath(__file__)), 'data')

pytestmark = [pytest.mark.models, pytest.mark.slow]


def run_shared_session(tmp_path, device):
    from cutieplus.backends.shared import SharedTaskBackend
    from gui.main_controller import MainController
    task = scenario_task(str(tmp_path))
    settings = {'device': device, 'allow_cpu': device == 'cpu', 'profile': device != 'cpu',
                'profile_frame_size': [96, 128], 'torch_threads': 4}
    env = ServiceEnv(task.root, str(tmp_path / 'state'), settings, device=device)
    try:
        client = env.client('tester', ':9')
        controller = MainController(compose_gui_config(task.workspace, device='cpu'),
                                    backend_factory=SharedTaskBackend.factory(client))
        assert controller.backend.kind == 'shared'
        result = run_session(controller, task)
        events = env.service.db.query('SELECT op, frame FROM edits ORDER BY edit_id')
    finally:
        env.close()
    return result, events


def compare_with_baseline(result):
    expected = np.load(path.join(DATA, 'baseline_local_session.npz'))
    actual = flatten(result)
    assert sorted(actual) == sorted(expected.files)
    differences = {}
    for key in expected.files:
        a, b = actual[key], expected[key]
        if a.shape != b.shape:
            differences[key] = 'presence differs'
        elif not np.array_equal(a, b):
            differences[key] = int((a != b).sum())
    return differences


# The one difference by design: the original window seeds a propagation with the soft probabilities
# of the last click held in its memory; the shared service seeds it with the committed mask (the
# argmax of those probabilities), as a job may only start from committed data. On this scenario
# 10 boundary pixels of the seed differ, which changes 1 pixel in 2 of the 8 propagated frames
# (tests/test_shared_gui.py; confirmed by seeding the original loop both ways). Every click,
# undo, anchor and file is otherwise identical.
SEED_EXPLAINED = {'steps/after_forward/000010', 'steps/after_forward/000013',
                  'steps/after_backward/000006', 'steps/after_backward/000010',
                  'steps/after_backward/000013', 'steps/restored_9', 'masks/000006',
                  'masks/000010', 'masks/000013'}


def check_against_baseline(result):
    differences = compare_with_baseline(result)
    assert set(differences) <= SEED_EXPLAINED, differences
    assert all(isinstance(count, int) and count <= 2 for count in differences.values()), differences


def test_shared_session_matches_the_original(tmp_path, qapp):
    result, events = run_shared_session(tmp_path, 'cpu')
    check_against_baseline(result)
    import json
    with open(path.join(DATA, 'baseline_local_session.json'), encoding='utf-8') as f:
        recorded = json.load(f)
    assert result['anchors'] == recorded['anchors']
    assert result['progress_keys'] == recorded['progress_keys']
    assert result['binary_masks'] == recorded['binary_masks']
    ops = [row['op'] for row in events]
    assert ops.count('click') == 2 and ops.count('undo_click') == 1
    assert 'remove_anchor' in ops and ops.count('restore') == 3


@pytest.mark.gpu
def test_shared_session_on_the_gpu_matches_the_local_mode(tmp_path, qapp):
    """The session with the real GPU worker against the original local mode on the same GPU. The
    shared-mode window runs in a process of its own, which must not initialize CUDA."""
    import json
    import subprocess
    import sys
    from gui.main_controller import MainController
    from tests.conftest import REPO, networks_on
    output = str(tmp_path / 'shared.npz')
    environment = dict(os.environ, CUTIEPLUS_TEST_GPU='1')
    subprocess.run([sys.executable, '-m', 'tests.shared_session_script', str(tmp_path / 'shared'),
                    'cuda', output], cwd=REPO, env=environment, check=True, timeout=900)
    with open(output + '.json') as f:
        report = json.load(f)
    assert report['cuda_initialized'] is False, 'the shared-mode window initialized CUDA'
    shared = dict(np.load(output))

    task = scenario_task(str(tmp_path / 'local'))
    local_controller = MainController(compose_gui_config(task.workspace, device='cuda'),
                                      networks=networks_on('cuda'))
    local = flatten(run_session(local_controller, task))
    assert sorted(shared) == sorted(local)
    assert all(shared[key].shape == local[key].shape for key in local)
    differences = {key: int((shared[key] != local[key]).sum()) for key in local
                   if not np.array_equal(shared[key], local[key])}
    assert set(differences) <= SEED_EXPLAINED, differences
    assert all(count <= 0.002 * 96 * 128 for count in differences.values()), differences
