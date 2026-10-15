"""
Right of each interval's boxes, its frame count, both ends included (15 - 650: 636 frames): on the
first interval's row and on every added one, following the numbers as they are typed; empty while
an added interval is not filled in.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task


def test_the_count_includes_both_ends(qapp):
    from gui.gui_utils import IntervalPair
    assert IntervalPair(30, 15, 20).count_label.text() == '6 frames'
    assert IntervalPair(30, 1, 30).count_label.text() == '30 frames'
    assert IntervalPair(30, 7, 7).count_label.text() == '1 frame'


def test_the_count_follows_the_boxes(qapp):
    from gui.gui_utils import IntervalBox, IntervalPair
    pair = IntervalPair(30, removable=True)  # an added interval starts blank
    label = pair.count_label
    assert label.text() == ''
    pair.start_box.setValue(15)
    assert label.text() == ''  # its end is not typed yet
    pair.end_box.setValue(20)
    assert label.text() == '6 frames'
    pair.start_box.setValue(25)  # past the end, until the start is confirmed
    assert label.text() == ''
    pair.order('start')  # the end moves to the start
    assert label.text() == '1 frame'
    pair.end_box.setValue(IntervalBox.BLANK)  # its end box cleared
    assert label.text() == ''


def test_the_count_sits_right_of_the_boxes_at_one_width(qapp):
    from gui.gui_utils import IntervalPair
    pair = IntervalPair(1200, 1, 6)
    pair.show()
    try:
        label = pair.count_label
        assert label.x() > pair.end_box.x() + pair.end_box.width()
        assert label.property('intervalCount') is True
        assert label.toolTip() == 'Frames in this interval, its start and end frames included.'
        width = pair.sizeHint().width()
        pair.end_box.setValue(1200)  # '1200 frames': the pair keeps its width
        assert label.text() == '1200 frames' and pair.sizeHint().width() == width
    finally:
        pair.close()


@pytest.mark.models
@pytest.mark.slow
def test_every_interval_row_shows_its_count(tmp_path, qapp, cpu_networks, monkeypatch):
    from PySide6.QtCore import QSettings
    from gui import theme
    from gui.main_controller import MainController
    stored = str(tmp_path / 'settings.ini')
    monkeypatch.setattr(theme, 'settings', lambda: QSettings(stored, QSettings.Format.IniFormat))
    task = make_task(str(tmp_path), frame_count=30)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        first = gui.interval_pairs[0]
        assert first.count_label.text() == '30 frames'  # the whole sequence
        gui.interval_end.setValue(10)
        controller.on_interval_change('end')
        assert first.count_label.text() == '10 frames'

        gui.on_add_interval()
        added = gui.interval_pairs[1]
        assert added.count_label.text() == ''
        added.start_box.setValue(15)
        added.end_box.setValue(20)
        controller.on_interval_change('end', added)
        assert added.count_label.text() == '6 frames'
        gui.on_add_interval()
        third = gui.interval_pairs[2]
        third.start_box.setValue(21)
        third.end_box.setValue(30)
        controller.on_interval_change('end', third)

        controller.on_remove_interval(added)  # the rows below move up, with their counts
        assert [pair.count_label.text() for pair in gui.interval_pairs] == ['10 frames',
                                                                           '10 frames']
    finally:
        gui.close()
