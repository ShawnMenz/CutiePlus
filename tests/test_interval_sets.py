"""
The labeling window shows one set at a time: a category, a description and an interval. Set 1 is
the sequence's own; every added interval is a set with a category and a description of its own.
The triangle opens the list of every set to pick one; picking selects it, folds the list and goes
to its first frame, and { / } and Shift+R select the interval they reach. The selected interval's
band on the frame slider is bold green, the others light green, and the mask takes the selected
set's category color.
"""
import json
from os import path

import pytest

from gui.task_state import INTERVALS, TaskState, read_section, write_sections
from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]

COLORS = {'car': '#00CF00', 'person': '#4800FF', 'truck': '#685810'}
THREE = {'intervals': [{'start': 1, 'end': 10},
                       {'start': 12, 'end': 20, 'description': 'Second.', 'category': 'person'},
                       {'start': 22, 'end': 30, 'description': 'Third.'}]}


def write_json(file_path, payload):
    with open(file_path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream)


def task_with(tmp_path, intervals=None):
    task = make_task(str(tmp_path), frame_count=30)
    write_json(path.join(task.root, 'main', 'categories.json'),
               [{'name': name, 'color': color} for name, color in COLORS.items()])
    if intervals is not None:
        write_sections(task.workspace, intervals=intervals)
    return task


def open_controller(task, networks):
    from gui.main_controller import MainController
    return MainController(compose_gui_config(task.workspace), networks=networks)


def shown(*widgets):
    return [not widget.isHidden() for widget in widgets]


def rgb(name):
    return tuple(int(COLORS[name][i:i + 2], 16) for i in (1, 3, 5))


def test_one_set_is_shown_and_the_triangle_lists_them_all(tmp_path, qapp, cpu_networks):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui import interactive_utils
    from gui.gui_utils import FrameSlider
    controller = open_controller(task_with(tmp_path, THREE), cpu_networks)
    gui = controller.gui
    first, second, third = gui.interval_pairs
    try:
        assert gui.selected_set == 0 and gui.sets_button.text() == '▶ 1/3'
        assert gui.set_list.isHidden()
        assert shown(first, gui.description_edit, second, second.description_edit,
                     second.remove_button, third) == [True, True, False, False, False, False]
        assert gui.task_label.text() == 'car' and gui.tl_slider.selected_interval == (1, 10)

        gui.toggle_sets()  # ▼: every set, as a row to pick
        assert not gui.set_list.isHidden() and gui.sets_button.text() == '▼ 1/3'
        rows = gui.set_rows
        assert [(row.number.text(), row.category.text(), row.description.full_text,
                 row.frames.text()) for row in rows] == [
            ('1', 'car', 'The red car moving right.', '<b>Interval:</b> 1-10'),
            ('2', 'person', 'Second.', '<b>Interval:</b> 12-20'),
            ('3', 'car', 'Third.', '<b>Interval:</b> 22-30')]  # no category saved: the sequence's
        assert [row.property('selected') for row in rows] == [True, False, False]

        QTest.mouseClick(rows[1], Qt.MouseButton.LeftButton)  # picks set 2
        assert gui.selected_set == 1 and gui.set_list.isHidden()
        assert gui.sets_button.text() == '▶ 2/3'
        assert controller.curr_ti == 11  # its first frame
        assert gui.tl_slider.selected_interval == (12, 20)
        assert shown(first, gui.description_edit, second, second.description_edit,
                     second.remove_button) == [False, False, True, True, True]
        assert gui.task_label.text() == 'person'
        assert tuple(interactive_utils.color_map_np[1]) == rgb('person')  # the mask's color
        assert (FrameSlider.SELECTED_BAND_COLOR, FrameSlider.BAND_COLOR) == ('#00853E', '#509E2F')
        assert not gui.tl_slider.grab().isNull()  # paints the bands
    finally:
        gui.close()


def test_an_added_interval_keeps_its_own_category(tmp_path, qapp, cpu_networks):
    task = task_with(tmp_path)
    controller = open_controller(task, cpu_networks)
    gui = controller.gui
    try:
        gui.interval_end.setValue(10)
        controller.on_interval_change('end')
        gui.on_add_interval()  # a new set, selected, its category the sequence's
        added = gui.interval_pairs[1]
        assert gui.selected_set == 1 and gui.sets_button.text() == '▶ 2/2'
        assert added.category == 'car' and gui.task_label.text() == 'car'
        controller.on_interval_category_chosen(added, 'person')  # before its frames are typed
        assert gui.task_label.text() == 'person'
        assert 'saved with the interval once its start and end' in gui.console.toPlainText()
        added.start_box.setValue(15)
        added.end_box.setValue(20)
        controller.on_interval_change('end', added)
        assert read_section(task.workspace, INTERVALS)['intervals'][1]['category'] == 'person'
        assert controller.task.category == 'car'  # set 1's, the sequence's, is unchanged

        controller.on_interval_category_chosen(added, 'truck')  # saved at once now
        assert 'Category of interval 15-20: truck' in gui.console.toPlainText()
        assert TaskState(task.workspace).interval_categories(30)[(15, 20)] == 'truck'
    finally:
        gui.close()
    reopened = open_controller(task, cpu_networks).gui
    try:
        assert reopened.interval_pairs[1].category == 'truck' and reopened.selected_set == 0
        assert reopened.task_label.text() == 'car'
    finally:
        reopened.close()


def test_jumps_and_previews_select_the_interval_they_reach(tmp_path, qapp, cpu_networks):
    controller = open_controller(task_with(tmp_path, THREE), cpu_networks)
    gui = controller.gui
    try:
        gui.on_jump_interval(1)  # }: the next interval start
        assert controller.curr_ti == 11 and gui.selected_set == 1
        assert gui.tl_slider.selected_interval == (12, 20)
        gui.on_preview_interval()  # Shift+R plays the interval jumped to, then the next
        gui.on_preview_interval()
        assert gui.selected_set == 2 and gui.tl_slider.selected_interval == (22, 30)
        gui.stop_playback()
        assert gui.tl_slider.selected_interval == (22, 30)  # stays selected

        controller.on_remove_interval(gui.interval_pairs[2])  # back to set 1
        assert gui.selected_set == 0 and gui.sets_button.text() == '▶ 1/2'
        assert len(gui.set_rows) == 2
    finally:
        gui.close()


def test_the_window_labels_one_object(tmp_path, qapp, cpu_networks):
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import QLabel, QPushButton
    controller = open_controller(task_with(tmp_path), cpu_networks)
    gui = controller.gui
    try:
        assert controller.curr_object == 1 and controller.num_objects == 1
        for name in ('object_dial', 'object_color', 'reset_object_button'):
            assert not hasattr(gui, name)
        assert not [label for label in gui.findChildren(QLabel)
                    if label.text().startswith('Current object')]
        assert 'Reset object' not in [button.text() for button in gui.findChildren(QPushButton)]
        keys = [shortcut.key() for shortcut in gui.findChildren(QShortcut)]
        assert QKeySequence('1') not in keys and QKeySequence('Ctrl+1') not in keys
    finally:
        gui.close()
