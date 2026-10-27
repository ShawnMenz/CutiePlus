"""
The labeling window's behaviour in the original local mode, compared with the recording made from
the original implementation (tests/make_baseline.py): masks after each step, anchors, review
status and the work folder (masks/ and metadata.json only).
"""
import json
from os import path

import numpy as np
import pytest

from tests.conftest import compose_gui_config
from tests.scenarios import flatten, run_session, scenario_task

DATA = path.join(path.dirname(path.abspath(__file__)), 'data')

pytestmark = [pytest.mark.models, pytest.mark.slow]


def test_local_session_matches_baseline(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    task = scenario_task(str(tmp_path))
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    result = run_session(controller, task)

    expected = np.load(path.join(DATA, 'baseline_local_session.npz'))
    actual = flatten(result)
    assert sorted(actual) == sorted(expected.files)
    mismatched = [key for key in expected.files if not np.array_equal(actual[key], expected[key])]
    assert not mismatched, f'differs from the original implementation: {mismatched}'

    with open(path.join(DATA, 'baseline_local_session.json'), encoding='utf-8') as f:
        recorded = json.load(f)
    assert result['anchors'] == recorded['anchors']
    assert result['progress_keys'] == recorded['progress_keys']
    assert result['work_folder'] == ['masks', 'metadata.json']


def test_closed_window_releases_the_task(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    from gui.task_state import TaskLocked, lock_holder
    task = scenario_task(str(tmp_path))
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    assert lock_holder(task.workspace)['labeler'] == 'tester'
    with pytest.raises(TaskLocked):
        MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    controller.gui.close()
    assert lock_holder(task.workspace) is None
