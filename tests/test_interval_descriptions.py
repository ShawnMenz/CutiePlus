"""
Each interval of a task is a clip of its own, with its own description: the first interval is
described by the sequence's description (the top box); every added interval has a box of its own
on its row, which starts as the text in the top box and is saved with the interval in the
intervals section of metadata.json. Save (Ctrl+S) keeps every interval's description and
category.
"""
import pytest

from gui.task_state import (INTERVALS, PROGRESS, TaskState, added_interval_descriptions,
                            first_interval, interval_descriptions, intervals_from_payload,
                            intervals_payload, read_section, write_sections)
from tests.conftest import compose_gui_config
from tests.helpers import make_task

SEQUENCE = 'The red car moving right.'  # make_task's description


def saved_intervals(task):
    return read_section(task.workspace, INTERVALS)


def test_descriptions_travel_with_their_intervals():
    payload = intervals_payload([(15, 20), (1, 10)], {(15, 20): 'parked'})
    assert payload == {'intervals': [{'start': 1, 'end': 10},
                                     {'start': 15, 'end': 20, 'description': 'parked'}]}
    assert intervals_from_payload(payload) == [(1, 10), (15, 20)]
    # the first interval is the entry without a description, wherever it is
    later_first = intervals_payload([(1, 5), (8, 12)], {(1, 5): 'early'})
    assert first_interval(later_first) == (8, 12)
    assert interval_descriptions([(1, 5), (8, 12)], later_first, SEQUENCE) == \
        {(1, 5): 'early', (8, 12): SEQUENCE}
    # saved before intervals had descriptions: the first entry is the first interval, the others
    # show the sequence's text, which they keep once saved
    old = {'intervals': [{'start': 1, 'end': 5}, {'start': 8, 'end': 12}]}
    assert first_interval(old) == (1, 5)
    assert interval_descriptions([(1, 5), (8, 12)], old, SEQUENCE) == \
        {(1, 5): SEQUENCE, (8, 12): SEQUENCE}
    assert added_interval_descriptions([(1, 5), (8, 12)], old, SEQUENCE) == {(8, 12): SEQUENCE}
    # one interval keeps the {start, end} shape: the sequence's description describes it
    assert intervals_payload([(3, 9)], {(3, 9): 'ignored'}) == {'start': 3, 'end': 9}
    assert first_interval({}) is None
    assert interval_descriptions([(1, 30)], {}, SEQUENCE) == {(1, 30): SEQUENCE}


@pytest.fixture
def open_window(tmp_path, qapp, cpu_networks):
    """open_window(task) -> the labeling window."""
    from gui.main_controller import MainController
    windows = []

    def make(task):
        controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
        windows.append(controller)
        qapp.processEvents()
        return controller

    yield make
    from shiboken6 import isValid
    for controller in windows:
        if isValid(controller.gui):  # not closed (and deleted) by the test
            controller.gui.close()


def shown(*widgets):
    # whether each is shown in the window (one set at a time: see tests/test_interval_sets.py)
    return [not widget.isHidden() for widget in widgets]


@pytest.mark.models
@pytest.mark.slow
def test_each_added_interval_has_its_own_description(tmp_path, open_window):
    from PySide6.QtWidgets import QLabel
    task = make_task(str(tmp_path), frame_count=30)
    controller = open_window(task)
    gui = controller.gui
    console = gui.console.toPlainText
    # no "Intervals" label: the first interval sits right of the sequence's description
    assert not [label for label in gui.findChildren(QLabel) if label.text() == 'Intervals']
    assert shown(gui.sets_button, gui.task_label, gui.description_edit, gui.interval_pairs[0],
                 gui.add_interval_button, gui.frame_name) == [True] * 6
    gui.interval_end.setValue(10)
    controller.on_interval_change('end')
    assert saved_intervals(task) == {'start': 1, 'end': 10}

    # a new set, shown instead of the first, whose description starts as the top box's text
    gui.description_edit.setText('The red car, shown.')
    gui.on_add_interval()
    gui.set_description(SEQUENCE)
    added = gui.interval_pairs[1]
    assert added.description() == 'The red car, shown.'
    assert shown(added.description_edit, added, added.remove_button) == [True] * 3
    assert shown(gui.description_edit, gui.interval_pairs[0]) == [False, False]
    assert added.remove_button.text() == '- Remove interval'
    controller.on_interval_change('end', added)  # blank boxes: nothing is saved yet
    assert saved_intervals(task) == {'start': 1, 'end': 10}
    added.start_box.setValue(15)
    added.end_box.setValue(20)
    controller.on_interval_change('end', added)
    assert saved_intervals(task)['intervals'] == [
        {'start': 1, 'end': 10},
        {'start': 15, 'end': 20, 'description': 'The red car, shown.', 'category': 'car'}]

    # its own box changes its own description only
    added.description_edit.setText('  The red car parked. ')
    controller.on_interval_description_change(added)
    assert added.description_edit.text() == 'The red car parked.'
    assert saved_intervals(task)['intervals'][1]['description'] == 'The red car parked.'
    assert 'Description of interval 15-20 changed' in console()
    # the top box changes the first interval's (the sequence's), not the added one's
    gui.description_edit.setText('A red car.')
    controller.on_description_change()
    assert controller.task.description == 'A red car.'
    assert controller.task.interval_descriptions(30) == {(1, 10): 'A red car.',
                                                         (15, 20): 'The red car parked.'}
    # an emptied box gets its text back
    added.description_edit.setText('  ')
    controller.on_interval_description_change(added)
    assert added.description() == 'The red car parked.'
    assert 'cannot be empty' in console()

    # a third set; removing the second shows the first again
    gui.on_add_interval()
    third = gui.interval_pairs[2]
    assert third.description() == 'A red car.'
    third.start_box.setValue(25)
    third.end_box.setValue(28)
    third.description_edit.setText('The red car leaving.')
    controller.on_interval_change('end', third)
    controller.on_remove_interval(added)
    assert gui.interval_pairs == [gui.interval_pairs[0], third] and gui.selected_set == 0
    assert shown(gui.interval_pairs[0], third, third.description_edit) == [True, False, False]
    assert saved_intervals(task)['intervals'] == [
        {'start': 1, 'end': 10},
        {'start': 25, 'end': 28, 'description': 'The red car leaving.', 'category': 'car'}]

    # { / } name the interval's description; Ctrl+S keeps each interval's description and
    # category, and submits the task
    gui.on_jump_interval(1)
    assert 'Interval 2 of 2: frames 25-28' in console()
    assert 'Description: The red car leaving.' in console()
    assert controller.on_save()
    assert saved_intervals(task)['intervals'] == [
        {'start': 1, 'end': 10},
        {'start': 25, 'end': 28, 'description': 'The red car leaving.', 'category': 'car'}]
    assert read_section(task.workspace, PROGRESS)['review_by'] == 'tester'

    # reopened: the first set shown, the added one with its text, hidden until selected
    gui.close()
    reopened = open_window(task).gui
    assert [pair.entry() for pair in reopened.interval_pairs] == [(1, 10), (25, 28)]
    assert reopened.interval_pairs[1].description() == 'The red car leaving.'
    assert shown(reopened.interval_pairs[0], reopened.interval_pairs[1].description_edit) == \
        [True, False]


@pytest.mark.models
@pytest.mark.slow
def test_intervals_saved_before_descriptions(tmp_path, qapp, cpu_networks):
    # the first entry is the first interval; the others show the sequence's text; opening the
    # window saves nothing, and a new sequence description leaves them the text they show
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=30)
    old = {'intervals': [{'start': 8, 'end': 12}, {'start': 1, 'end': 5}]}
    write_sections(task.workspace, intervals=old)
    progress = read_section(task.workspace, PROGRESS)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        assert [pair.entry() for pair in gui.interval_pairs] == [(1, 5), (8, 12)]
        assert gui.interval_pairs[1].description() == SEQUENCE
        assert not controller.save_interval_pairs()
        assert saved_intervals(task) == old
        assert read_section(task.workspace, PROGRESS) == progress

        gui.description_edit.setText('A blue car.')
        controller.on_description_change()
        assert saved_intervals(task)['intervals'] == [
            {'start': 1, 'end': 5},
            {'start': 8, 'end': 12, 'description': SEQUENCE, 'category': 'car'}]
        assert TaskState(task.workspace).interval_descriptions(30) == {
            (1, 5): 'A blue car.', (8, 12): SEQUENCE}
    finally:
        gui.close()
