"""
The category of a task (gui/category_menu.py): a click on the labeling window's category pill
opens a menu of the subset's categories, each row its name and a sample of its color, with a
search box that keeps the names starting with the letters typed. Choosing one saves it in the work
folder (metadata.json's category; the input stays as it is) as a change that needs a new review, and the pill
takes the category's color with text that reads on it.
"""
import json
from os import path

import pytest

from gui.task_input import Category
from gui.task_state import CATEGORY, METADATA_FILE, TaskState, read_section, task_status
from tests.conftest import compose_gui_config
from tests.helpers import make_task, summary_of

CATEGORIES = [Category(name, color) for name, color in (
    ('bag', '#0050C7'), ('ball', '#AFBF87'), ('cable car', '#973000'), ('car', '#00CF00'),
    ('cat', '#F700FF'), ('person', '#4800FF'), ('truck', '#685810'))]


def shown(menu):
    return [menu.list.item(row).text() for row in menu.visible_rows()]


def press(widget, key):
    from PySide6.QtTest import QTest
    QTest.keyClick(widget, key)


def test_matching_keeps_the_names_that_start_with_the_letters():
    from gui.category_menu import matching
    names = lambda text: [category.name for category in matching(CATEGORIES, text)]  # noqa: E731
    assert names('p') == ['person']
    assert names('ca') == ['cable car', 'car', 'cat']
    assert names(' CA ') == ['cable car', 'car', 'cat']  # case and spaces aside
    assert names('car') == ['car']  # the start of the name, not of a later word
    assert names('') == [category.name for category in CATEGORIES]
    assert names('xyz') == []


def test_the_menu_filters_as_you_type_and_picks_with_the_keys(qapp):
    from PySide6.QtCore import Qt
    from gui.category_menu import COLOR_ROLE, CategoryMenu
    menu = CategoryMenu(CATEGORIES, current='person')
    picked = []
    menu.chosen.connect(picked.append)
    assert shown(menu) == [category.name for category in CATEGORIES]
    assert menu.list.currentItem().text() == 'person'  # the current category, selected
    assert menu.list.item(3).data(COLOR_ROLE) == '#00CF00'  # car's color sample
    menu.search.setText('ca')
    assert shown(menu) == ['cable car', 'car', 'cat']
    assert menu.list.currentItem().text() == 'cable car'  # the first shown
    press(menu.search, Qt.Key.Key_Down)
    press(menu.search, Qt.Key.Key_Down)
    press(menu.search, Qt.Key.Key_Down)  # stops at the last row shown
    assert menu.list.currentItem().text() == 'cat'
    press(menu.search, Qt.Key.Key_Up)
    press(menu.search, Qt.Key.Key_Return)
    assert picked == ['car']

    menu = CategoryMenu(CATEGORIES)
    menu.chosen.connect(picked.append)
    menu.search.setText('xyz')
    assert shown(menu) == [] and not menu.empty.isHidden()
    assert menu.empty.text() == 'No category starts with "xyz".'
    press(menu.search, Qt.Key.Key_Return)  # nothing to pick
    menu.search.setText('t')
    menu.pick(menu.list.item(6))  # a click on a row
    assert picked == ['car', 'truck']

    menu = CategoryMenu(CATEGORIES)
    menu.chosen.connect(picked.append)
    menu.show()
    press(menu.search, Qt.Key.Key_Escape)  # closes, picks nothing
    assert not menu.isVisible() and picked == ['car', 'truck']
    assert CategoryMenu([]).empty.text().startswith('No categories to choose from')


def test_the_pill_text_reads_on_its_color():
    from gui.style import DARK_TEXT, WHITE, contrast_text
    assert [contrast_text(color) for color in ('#AFBF87', '#BFA7FF', '#00CF00', '#FF9F00')] == \
        [DARK_TEXT] * 4
    assert [contrast_text(color) for color in ('#006858', '#4800FF', '#685810', '#0050C7')] == \
        [WHITE] * 4


@pytest.mark.models
@pytest.mark.slow
def test_a_click_on_the_pill_chooses_and_saves_the_category(tmp_path, qapp, cpu_networks):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6)
    with open(path.join(task.root, 'main', 'categories.json'), 'w', encoding='utf-8') as stream:
        json.dump([{'name': c.name, 'color': c.color} for c in CATEGORIES], stream)
    events = []
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks,
                                on_task_event=lambda _, event: events.append(event))
    gui = controller.gui
    try:
        assert gui.task_label.text() == 'car' and 'Click to choose' in gui.task_label.toolTip()
        assert 'background: #00CF00' in gui.task_label.styleSheet()
        QTest.mouseClick(gui.task_label, Qt.MouseButton.LeftButton)
        menu = gui.category_menu
        assert menu is not None and menu.isVisible()
        assert menu.list.currentItem().text() == 'car'
        menu.search.setText('p')
        press(menu.search, Qt.Key.Key_Return)
        assert not menu.isVisible()
        assert gui.task_label.text() == 'person' and 'background: #4800FF' in \
            gui.task_label.styleSheet()
        saved = read_section(task.workspace, CATEGORY)
        assert (saved['name'], saved['by']) == ('person', 'tester')
        assert task_status(task.workspace) == 'in progress' and events[-1] == 'category'
        assert 'Category: person; the task needs a new review.' in gui.console.toPlainText()
        with open(path.join(task.root, 'main', 'targets.json'), encoding='utf-8') as stream:
            assert json.load(stream)['0001']['category'] == 'car'  # the input stays

        before = path.getmtime(path.join(task.workspace, METADATA_FILE))
        controller.on_category_chosen('person')  # the same: nothing changes
        assert path.getmtime(path.join(task.workspace, METADATA_FILE)) == before
        assert TaskState(task.workspace).category == 'person'
        assert summary_of(task.workspace)['target'].startswith('person: ')
    finally:
        gui.close()


@pytest.mark.models
@pytest.mark.slow
def test_a_task_without_a_category_shows_a_gray_null_pill(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6, in_targets=False)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    try:
        label = controller.gui.task_label
        assert label.text() == 'null' and label.property('pill') == 'null'
        assert label.styleSheet() == ''
    finally:
        controller.gui.close()


@pytest.mark.models
@pytest.mark.slow
def test_the_mask_is_drawn_in_the_category_color(tmp_path, qapp, cpu_networks):
    from gui import interactive_utils
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6)
    with open(path.join(task.root, 'main', 'categories.json'), 'w', encoding='utf-8') as stream:
        json.dump([{'name': c.name, 'color': c.color} for c in CATEGORIES], stream)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    try:
        assert tuple(interactive_utils.color_map_np[1]) == (0x00, 0xCF, 0x00)  # car
        drawn = controller.gui.canvas_image  # frame 1 shows its anchor's mask
        controller.on_category_chosen('person')
        assert tuple(interactive_utils.color_map_np[1]) == (0x48, 0x00, 0xFF)
        assert controller.gui.canvas_image is not drawn  # drawn again in the new color
    finally:
        controller.gui.close()
    other = make_task(str(tmp_path / 'other'), frame_count=6, in_targets=False)
    controller = MainController(compose_gui_config(other.workspace), networks=cpu_networks)
    try:  # no category: cyan, as before
        assert tuple(interactive_utils.color_map_np[1]) == interactive_utils.TARGET_MASK_COLOR
    finally:
        controller.gui.close()
