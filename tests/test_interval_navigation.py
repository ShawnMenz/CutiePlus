"""
{ / } move to the previous / next interval start of a task, around the ends; Shift+R then plays
the interval moved to.
"""

import pytest

from gui.task_state import interval_index_from, intervals_payload, write_sections
from tests.conftest import compose_gui_config
from tests.helpers import make_task

INTERVALS = [(3, 8), (12, 20), (25, 28)]


def test_the_nearest_start_in_each_direction_around_the_ends():
    # forward: the first start after the frame, the first interval past the last
    assert [interval_index_from(INTERVALS, frame, 1) for frame in (1, 3, 5, 12, 24, 25, 30)] == \
        [0, 1, 1, 2, 2, 0, 0]
    # back: the last start before the frame (inside an interval its own start first), the last
    # interval before the first
    assert [interval_index_from(INTERVALS, frame, -1) for frame in (30, 26, 25, 16, 12, 4, 3, 1)] == \
        [2, 2, 1, 1, 0, 0, 2, 2]
    assert [interval_index_from([(1, 30)], frame, d) for frame in (1, 10) for d in (1, -1)] == \
        [0, 0, 0, 0]


@pytest.mark.models
@pytest.mark.slow
def test_the_keys_move_the_window_between_interval_starts(tmp_path, qapp, cpu_networks):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=30)
    write_sections(task.workspace, intervals=intervals_payload(INTERVALS))
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    slider = gui.tl_slider
    try:
        def press(key):
            shortcuts = [shortcut for shortcut in gui.findChildren(QShortcut)
                         if shortcut.key() == QKeySequence(key)]
            assert len(shortcuts) == 1  # two on one key would be ambiguous and neither would fire
            shortcuts[0].activated.emit()
            return slider.value() + 1  # 1-indexed, as the interval boxes show it

        assert slider.value() == 0
        assert [press(Qt.Key.Key_BraceRight) for _ in range(4)] == [3, 12, 25, 3]
        assert press(Qt.Key.Key_BraceLeft) == 25
        slider.setValue(15)  # inside the second interval
        assert [press(Qt.Key.Key_BraceLeft) for _ in range(3)] == [12, 3, 25]
        assert 'Interval 3 of 3: frames 25-28' in gui.console.toPlainText()

        # Shift+R plays the interval moved to, and stops at its end
        press(Qt.Key.Key_BraceLeft)
        assert slider.value() + 1 == 12
        gui.on_preview_interval()
        assert gui.timer.isActive() and gui.play_until + 1 == 20
        # a jump stops the preview
        press(Qt.Key.Key_BraceRight)
        assert not gui.timer.isActive() and slider.value() + 1 == 25

        # nothing moves during a propagation
        controller.propagating = True
        gui.on_jump_interval(1)
        assert slider.value() + 1 == 25
        controller.propagating = False
    finally:
        gui.close()
