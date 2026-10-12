"""
Each interval of a task is a clip of its own, with its own description: the first interval is
described by the sequence's description (the top box); every added interval has a box of its own
on its row, which starts as the text in the top box and is saved with the interval in
interval.json. Saving binary PNGs writes one numbered description per interval.
"""
import json
from os import path

import pytest

from cutieplus.client import ClientError
from gui.task_state import (DESCRIPTION_TEXT, INTERVAL_FILE, INTERVAL_TEXT, TaskState,
                            added_interval_descriptions, first_interval, interval_descriptions,
                            intervals_from_payload, intervals_payload, write_description_text)
from tests.conftest import compose_gui_config
from tests.helpers import make_task
from tests.service_env import ServiceEnv, fake_overrides

SEQUENCE = 'The red car moving right.'  # make_task's description


def read_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def read_text(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read()


def test_descriptions_travel_with_their_intervals(tmp_path):
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

    write_description_text(str(tmp_path), [SEQUENCE])
    assert read_text(tmp_path / DESCRIPTION_TEXT) == f'description={SEQUENCE}\n'
    write_description_text(str(tmp_path), [SEQUENCE, 'parked'])
    assert read_text(tmp_path / DESCRIPTION_TEXT) == \
        f'description_1={SEQUENCE}\ndescription_2=parked\n'


@pytest.mark.slow
def test_the_service_saves_and_exports_interval_descriptions(tmp_path):
    task = make_task(str(tmp_path), frame_count=30)
    env = ServiceEnv(task.root, str(tmp_path / 'state'), fake_overrides())
    try:
        client = env.client('alice', ':2')
        lease = client.call('open_task', workspace=task.workspace)
        ids = dict(task_id=lease['task_id'], lease_epoch=lease['lease_epoch'])
        with pytest.raises(ClientError) as error:
            client.call('set_intervals', intervals=[[1, 10], [15, 20, '  ']], **ids)
        assert error.value.code == 'invalid'
        revision = env.service.tasks.task_row(lease['task_id'])['content_revision']
        reply = client.call('set_intervals', intervals=[[15, 20, ' parked '], [1, 10]],
                            request_id='intervals-1', **ids)
        assert reply['content_revision'] == revision + 1  # a change: it needs a new review
        saved = {'intervals': [{'start': 1, 'end': 10},
                               {'start': 15, 'end': 20, 'description': 'parked'}]}
        assert read_json(path.join(task.workspace, INTERVAL_FILE)) == saved
        assert client.call('set_intervals', intervals=[[1, 10]], request_id='intervals-1',
                           **ids)['replayed']

        client.call('submit_for_review', target_objects=[1], **ids)
        folder = path.join(task.workspace, 'binary_masks')
        assert read_text(path.join(folder, DESCRIPTION_TEXT)) == \
            f'description_1={SEQUENCE}\ndescription_2=parked\n'
        assert 'interval_2_start=15' in read_text(path.join(folder, INTERVAL_TEXT))
        assert read_json(path.join(task.workspace, INTERVAL_FILE)) == saved  # kept
    finally:
        env.close()


@pytest.fixture(params=['local', 'shared'])
def open_window(request, tmp_path, qapp):
    """open_window(task) -> the labeling window, in local mode or through the shared service."""
    from gui.main_controller import MainController
    windows, environments = [], []

    def make(task):
        if request.param == 'local':
            networks = request.getfixturevalue('cpu_networks')
            controller = MainController(compose_gui_config(task.workspace), networks=networks)
        else:
            from cutieplus.backends.shared import SharedTaskBackend
            if not environments:
                environments.append(ServiceEnv(task.root, str(tmp_path / 'state'),
                                               fake_overrides()))
            client = environments[0].client('alice', ':2')
            controller = MainController(compose_gui_config(task.workspace, device='cpu'),
                                        backend_factory=SharedTaskBackend.factory(client))
        windows.append(controller)
        qapp.processEvents()
        return controller

    yield make
    from shiboken6 import isValid
    for controller in windows:
        if isValid(controller.gui):  # not closed (and deleted) by the test
            controller.gui.close()
    for env in environments:
        env.close()


def row_widgets(gui, row):
    grid = gui.interval_grid
    return [grid.itemAtPosition(row, column).widget() if grid.itemAtPosition(row, column) else None
            for column in range(5)]


@pytest.mark.models
@pytest.mark.slow
def test_each_added_interval_has_its_own_description(tmp_path, open_window):
    from PySide6.QtWidgets import QLabel
    task = make_task(str(tmp_path), frame_count=30)
    interval_file = path.join(task.workspace, INTERVAL_FILE)
    controller = open_window(task)
    gui = controller.gui
    console = gui.console.toPlainText
    # no "Intervals" label: the first interval sits right of the sequence's description
    assert not [label for label in gui.findChildren(QLabel) if label.text() == 'Intervals']
    assert row_widgets(gui, 0)[:5] == [gui.task_label, gui.description_edit, gui.interval_pairs[0],
                                       gui.add_interval_button, gui.frame_name]
    gui.interval_end.setValue(10)
    controller.on_interval_change('end')
    assert read_json(interval_file) == {'start': 1, 'end': 10}

    # a new row, whose description starts as the text shown in the top box
    gui.description_edit.setText('The red car, shown.')
    gui.on_add_interval()
    gui.set_description(SEQUENCE)
    added = gui.interval_pairs[1]
    assert added.description() == 'The red car, shown.'
    assert row_widgets(gui, 1)[1:4] == [added.description_edit, added, added.remove_button]
    controller.on_interval_change('end', added)  # blank boxes: nothing is saved yet
    assert read_json(interval_file) == {'start': 1, 'end': 10}
    added.start_box.setValue(15)
    added.end_box.setValue(20)
    controller.on_interval_change('end', added)
    assert read_json(interval_file)['intervals'] == [
        {'start': 1, 'end': 10}, {'start': 15, 'end': 20, 'description': 'The red car, shown.'}]

    # its own box changes its own description only
    added.description_edit.setText('  The red car parked. ')
    controller.on_interval_description_change(added)
    assert added.description_edit.text() == 'The red car parked.'
    assert read_json(interval_file)['intervals'][1]['description'] == 'The red car parked.'
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

    # a third row; removing the second moves it up
    gui.on_add_interval()
    third = gui.interval_pairs[2]
    assert third.description() == 'A red car.'
    third.start_box.setValue(25)
    third.end_box.setValue(28)
    third.description_edit.setText('The red car leaving.')
    controller.on_interval_change('end', third)
    controller.on_remove_interval(added)
    assert gui.interval_pairs == [gui.interval_pairs[0], third]
    assert row_widgets(gui, 1)[1:4] == [third.description_edit, third, third.remove_button]
    assert row_widgets(gui, 2) == [None] * 5
    assert read_json(interval_file)['intervals'] == [
        {'start': 1, 'end': 10}, {'start': 25, 'end': 28, 'description': 'The red car leaving.'}]

    # { / } name the interval's description; Ctrl+S writes one per interval, numbered as the
    # intervals
    gui.on_jump_interval(1)
    assert 'Interval 2 of 2: frames 25-28' in console()
    assert 'Description: The red car leaving.' in console()
    controller.on_export_binary()
    folder = path.join(task.workspace, 'binary_masks')
    assert read_text(path.join(folder, DESCRIPTION_TEXT)) == \
        'description_1=A red car.\ndescription_2=The red car leaving.\n'
    assert read_text(path.join(folder, INTERVAL_TEXT)) == (
        'interval_count=2\ninterval_1_start=1\ninterval_1_end=10\n'
        'interval_2_start=25\ninterval_2_end=28\n')

    # reopened: the first interval on the top row, the added one on its row with its text
    gui.close()
    reopened = open_window(task).gui
    assert [pair.entry() for pair in reopened.interval_pairs] == [(1, 10), (25, 28)]
    assert reopened.interval_pairs[1].description() == 'The red car leaving.'
    assert row_widgets(reopened, 0)[2] is reopened.interval_pairs[0]
    assert row_widgets(reopened, 1)[1] is reopened.interval_pairs[1].description_edit


@pytest.mark.models
@pytest.mark.slow
def test_intervals_saved_before_descriptions(tmp_path, qapp, cpu_networks):
    # the first entry is the first interval; the others show the sequence's text; opening the
    # window saves nothing, and a new sequence description leaves them the text they show
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=30)
    interval_file = path.join(task.workspace, INTERVAL_FILE)
    old = {'intervals': [{'start': 8, 'end': 12}, {'start': 1, 'end': 5}]}
    with open(interval_file, 'w', encoding='utf-8') as f:
        json.dump(old, f)
    progress = read_text(path.join(task.workspace, 'progress.json')) \
        if path.exists(path.join(task.workspace, 'progress.json')) else None
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        assert [pair.entry() for pair in gui.interval_pairs] == [(1, 5), (8, 12)]
        assert gui.interval_pairs[1].description() == SEQUENCE
        assert not controller.save_interval_pairs()
        assert read_json(interval_file) == old
        assert (read_text(path.join(task.workspace, 'progress.json'))
                if path.exists(path.join(task.workspace, 'progress.json')) else None) == progress

        gui.description_edit.setText('A blue car.')
        controller.on_description_change()
        assert read_json(interval_file)['intervals'] == [
            {'start': 1, 'end': 5}, {'start': 8, 'end': 12, 'description': SEQUENCE}]
        assert TaskState(task.workspace).interval_descriptions(30) == {
            (1, 5): 'A blue car.', (8, 12): SEQUENCE}
    finally:
        gui.close()
