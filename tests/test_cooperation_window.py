"""
The labeling window on a host with the cooperation service (cutieplus/backends/service.py): it loads
no model; its clicks and propagations run in the shared GPU worker (here the fake engine, in its own
process); a propagation's masks are applied once its job ends; a propagation waits for the GPU
while the window stays editable, and its result is held for review when the task changed meanwhile;
without the service the window still edits and saves, but infers nothing.
"""
import os
from os import path

import numpy as np
import pytest

from cutieplus.backends.service import OfflineTaskBackend, ServiceTaskBackend
from cutieplus.cooperation.client import ServiceClient
from cutieplus.cooperation.server import Server
from gui.task_state import read_metadata
from tests.conftest import compose_gui_config
from tests.cooperation_harness import PASSWORD
from tests.gpu_harness import GpuHarness
from tests.helpers import make_task

pytestmark = pytest.mark.slow


@pytest.fixture
def world(tmp_path, monkeypatch):
    # loading a model anywhere in this process would fail the test
    from cutieplus.engine import models

    def refuse(*args, **kwargs):
        raise AssertionError('a window loaded a model')

    monkeypatch.setattr(models, 'load_networks', refuse)
    gpu = GpuHarness(tmp_path, step_seconds=0.0)
    server = Server(gpu.service.handle, str(tmp_path / 'app.sock'))
    server.start()
    owner = gpu.owner()
    gpu.add(owner, 'alice')
    client = ServiceClient(str(tmp_path / 'app.sock'), timeout=30)
    client.token = client.call('login', login_name='alice', password=PASSWORD)['token']
    task = make_task(gpu.config.workspace_root, frame_count=8)
    gpu.client = client
    gpu.task_files = task
    yield gpu
    server.stop()
    gpu.close()


def open_window(world, held_answer=None):
    """held_answer: what the user answers when the window offers a held result as it opens."""
    from gui.gui import GUI
    from gui.main_controller import MainController
    client = world.client
    original = GUI.confirm_held_result
    if held_answer is not None:
        GUI.confirm_held_result = lambda self, job: held_answer
    try:
        return MainController(
            compose_gui_config(world.task_files.workspace, labeler='alice'),
            backend_factory=lambda cfg, res_man, task: ServiceTaskBackend(cfg, res_man, task,
                                                                          client))
    finally:
        GUI.confirm_held_result = original


def masks(world):
    folder = path.join(world.task_files.workspace, 'masks')
    return sorted(name[:-4] for name in os.listdir(folder))


def test_clicks_and_propagation_run_in_the_shared_worker(world, qapp):
    world.start_worker()
    controller = open_window(world)
    try:
        assert controller.cutie is None and controller.backend.device == 'cpu'
        assert controller.gui.clear_all_mem_button.isHidden()
        revision = read_metadata(world.task_files.workspace).get('revision', 0)
        controller.click_fn('left', 12, 10)
        assert controller.curr_mask[10, 12] == 1
        assert controller.task.anchors['000001']['kind'] == 'mask'
        controller.on_forward_propagation()  # runs to the end: the worker is free
        assert masks(world) == [f'{index:06d}' for index in range(1, 9)]
        assert read_metadata(world.task_files.workspace)['revision'] > revision
        jobs = world.db.query("SELECT operation_type, status, result_state FROM gpu_jobs "
                              "ORDER BY queue_sequence")
        assert jobs[0] == {'operation_type': 'click', 'status': 'succeeded',
                           'result_state': 'none'}
        assert jobs[-1] == {'operation_type': 'propagate', 'status': 'succeeded',
                            'result_state': 'applied'}
        # the staged files are gone once applied
        assert not os.listdir(world.config.jobs_dir)
    finally:
        controller.gui.close()


def test_a_queued_run_leaves_the_window_editable_and_a_changed_task_holds_its_result(world,
                                                                                    qapp):
    controller = open_window(world)  # no worker yet: the GPU is not available
    try:
        controller.on_forward_propagation()
        assert controller.waiting_for_gpu and not controller.propagating
        assert 'Waiting for the GPU' in controller.gui.console.toPlainText()
        # still editable: an empty anchor on frame 5
        controller.gui.tl_slider.setValue(4)
        controller.on_reset_mask()
        assert controller.task.anchors['000005']['kind'] == 'empty'
        world.start_worker()
        world.wait_for(lambda: controller.backend.queue_state()['status'] != 'queued')
        controller.on_queue_timer()  # follows the run, then holds its result
        assert not controller.waiting_for_gpu
        assert 'its result is held' in controller.gui.console.toPlainText()
        assert masks(world) == ['000001', '000005']  # the anchors only: nothing applied
    finally:
        controller.gui.close()
    # the next window asks; applying keeps the anchors
    reopened = open_window(world, held_answer=True)
    try:
        assert 'saved' in reopened.gui.console.toPlainText()
        assert '000004' in masks(world)
        assert reopened.res_man.get_mask(3).max() == 1  # the propagated mask
        assert reopened.res_man.get_mask(4).max() == 0  # the empty anchor, kept
        assert reopened.task.anchors['000005']['kind'] == 'empty'
        state = world.db.query_one("SELECT result_state FROM gpu_jobs WHERE operation_type = "
                                   "'propagate'")
        assert state['result_state'] == 'applied'
    finally:
        reopened.gui.close()


def test_pause_cancels_a_queued_run(world, qapp):
    controller = open_window(world)
    try:
        controller.on_forward_propagation()
        assert controller.waiting_for_gpu
        controller.on_forward_propagation()  # the button reads Pause meanwhile
        assert not controller.waiting_for_gpu
        status = world.db.query_one("SELECT status FROM gpu_jobs WHERE operation_type = "
                                    "'propagate'")['status']
        assert status == 'cancelled'
    finally:
        controller.gui.close()


def test_without_the_service_the_window_edits_but_infers_nothing(world, qapp):
    from gui.main_controller import MainController
    controller = MainController(compose_gui_config(world.task_files.workspace),
                                backend_factory=OfflineTaskBackend)
    try:
        controller.click_fn('left', 12, 10)
        assert 'Click not applied' in controller.gui.console.toPlainText()
        controller.on_forward_propagation()
        assert 'Propagation not started' in controller.gui.console.toPlainText()
        controller.gui.tl_slider.setValue(3)
        controller.on_reset_mask()  # an edit without the GPU
        assert controller.task.anchors['000004']['kind'] == 'empty'
        assert np.array_equal(controller.curr_mask, np.zeros_like(controller.curr_mask))
    finally:
        controller.gui.close()
