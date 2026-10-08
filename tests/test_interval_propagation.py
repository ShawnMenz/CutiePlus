"""
F and B started inside an interval stay inside it (up to its last / first frame); started outside
every interval they run as before. The coordinator side is in test_service.py.
"""
import json
from os import path

import pytest

from cutieplus.engine.policies import frames_after_start, propagation_bounds
from gui.task_state import INTERVAL_FILE, intervals_payload
from tests.conftest import compose_gui_config
from tests.helpers import make_task

INTERVALS = [(3, 8), (12, 20)]  # 1-indexed, as stored: ti 2-7 and 11-19


def test_the_interval_of_the_start_frame_bounds_the_run():
    assert propagation_bounds(INTERVALS, 2) == (2, 7)  # its first frame
    assert propagation_bounds(INTERVALS, 7) == (2, 7)  # its last frame
    assert propagation_bounds(INTERVALS, 15) == (11, 19)
    assert propagation_bounds(INTERVALS, 9) is None  # between intervals
    assert propagation_bounds(INTERVALS, 0) is None
    assert propagation_bounds([(1, 30)], 29) == (0, 29)

    assert frames_after_start(4, 'forward', 30, (2, 7)) == [5, 6, 7]
    assert frames_after_start(4, 'backward', 30, (2, 7)) == [3, 2]
    assert frames_after_start(7, 'forward', 30, (2, 7)) == []  # at the end of its interval
    # no interval: to the ends of the sequence, as before
    assert frames_after_start(27, 'forward', 30) == [28, 29]
    assert frames_after_start(2, 'backward', 30) == [1, 0]


@pytest.mark.models
@pytest.mark.slow
def test_the_window_propagates_inside_the_interval_of_the_current_frame(tmp_path, qapp,
                                                                          cpu_networks):
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=16, mask_anchors=(0, 8))
    with open(path.join(task.workspace, INTERVAL_FILE), 'w', encoding='utf-8') as f:
        json.dump(intervals_payload([(1, 5), (9, 13)]), f)  # ti 0-4 and 8-12
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui

    def masks():
        return [ti for ti, name in enumerate(task.names) if task.mask(name) is not None]

    try:
        assert masks() == [0, 8]  # the anchors
        controller.on_forward_propagation()  # from frame 1: up to frame 5
        assert masks() == [0, 1, 2, 3, 4, 8]
        log = gui.console.toPlainText()
        assert 'Propagation stays inside the interval 1-5.' in log
        assert 'Reached the end of the interval 1-5.' in log

        gui.tl_slider.setValue(12)  # frame 13: back to frame 9, an anchor that is not written
        controller.on_backward_propagation()
        assert masks() == [0, 1, 2, 3, 4, 8, 9, 10, 11]
        assert 'Reached the start of the interval 9-13.' in gui.console.toPlainText()

        gui.tl_slider.setValue(5)  # frame 6, in no interval: to the last frame, as before
        controller.on_forward_propagation()
        assert masks() == [ti for ti in range(16) if ti != 5]  # the start frame is not written
    finally:
        gui.close()
