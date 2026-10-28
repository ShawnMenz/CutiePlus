"""
Output video in the labeling window's Save card (gui/demo_video.py, MainController.on_output_demo):
with the Output FPS box beside it, it saves the task, writes one demo video per interval to
cutie_demo/<tab>/ beside the input, in the window's visualization mode, and marks the task done
once they are written, unless it changed meanwhile. What Ctrl+Q refuses, it refuses too. The work
folder keeps masks/ and metadata.json only.
"""
import json
import os
import time
from os import path

import av
import pytest

from gui.demo_video import task_clips
from gui.task_state import (TaskState, intervals_payload, read_section, task_status,
                            write_sections)
from tests.conftest import compose_gui_config
from tests.helpers import DESCRIPTION, make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


def open_window(task, networks, events):
    from gui.main_controller import MainController
    return MainController(compose_gui_config(task.workspace), networks=networks,
                          on_task_event=lambda _, event: events.append(event))


def wait_for_output(qapp, controller, timeout: float = 60) -> str:
    deadline = time.monotonic() + timeout
    while controller.demo_worker is not None:
        assert time.monotonic() < deadline, 'the output never finished'
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    return controller.gui.console.toPlainText()


def card_title(widget) -> str:
    from gui.style import Card
    while widget is not None and not isinstance(widget, Card):
        widget = widget.parentWidget()
    return widget.title_label.text() if widget is not None else ''


def two_intervals(tmp_path):
    task = make_task(str(tmp_path), frame_count=12)
    categories = path.join(task.root, task.split, 'categories.json')
    with open(categories, encoding='utf-8') as f:
        listed = json.load(f)
    with open(categories, 'w', encoding='utf-8') as f:
        json.dump(listed + [{'name': 'bus', 'color': '#40BFF7'}], f)
    write_sections(task.workspace, intervals=intervals_payload(
        [(1, 5), (8, 12)], {(8, 12): 'the bus leaving'}, {(8, 12): 'bus'}))
    return task


def test_output_writes_a_video_per_interval_and_marks_the_task_done(tmp_path, qapp,
                                                                    cpu_networks):
    task = two_intervals(tmp_path)
    events = []
    controller = open_window(task, cpu_networks, events)
    try:
        gui = controller.gui
        assert card_title(gui.output_video_button) == 'Save' == card_title(gui.fps_dial)
        clips = task_clips(controller.task, controller.res_man, str(tmp_path), 'davis')
        assert [(clip.category, clip.description, clip.color, len(clip.frames))
                for clip in clips] == [('car', DESCRIPTION, (0x00, 0xCF, 0x00), 5),
                                       ('bus', 'the bus leaving', (0x40, 0xBF, 0xF7), 5)]
        assert clips[0].modalities == ['visual', 'thermal']
        gui.fps_dial.setValue(7)
        controller.on_output_demo()
        console = wait_for_output(qapp, controller)
        demo = path.join(str(tmp_path), 'cutie_demo', 'main')
        assert sorted(os.listdir(demo)) == ['0001_1-5_davis.mp4', '0001_8-12_davis.mp4'], console
        for name, frames in (('0001_1-5_davis.mp4', 5), ('0001_8-12_davis.mp4', 5)):
            with av.open(path.join(demo, name)) as video:
                stream = video.streams.video[0]
                assert len(list(video.decode(stream))) == frames and stream.average_rate == 7
        assert task_status(task.workspace) == 'done', console
        assert read_section(task.workspace, 'progress')['done_by'] == 'tester'
        assert events[-1] == 'done' and '0001 is done' in console
        assert sorted(os.listdir(task.workspace)) == ['masks', 'metadata.json']
        assert gui.output_video_button.text() == 'Output video (Ctrl+E)'
    finally:
        controller.gui.close()


def test_a_sequence_that_needs_attention_is_refused(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    TaskState(task.workspace, 'tester').save_attention(True)
    controller = open_window(task, cpu_networks, [])
    try:
        controller.on_output_demo()
        assert controller.demo_worker is None
        assert 'No video output: this sequence needs attention' in \
            controller.gui.console.toPlainText()
        assert not path.exists(path.join(str(tmp_path), 'cutie_demo'))
        assert task_status(task.workspace) != 'done'
    finally:
        controller.gui.close()


def test_a_change_while_writing_keeps_the_videos_but_not_the_done_mark(tmp_path, qapp,
                                                                       cpu_networks):
    task = make_task(str(tmp_path), frame_count=6)
    controller = open_window(task, cpu_networks, [])
    try:
        controller.on_output_demo()
        TaskState(task.workspace, 'tester').save_description('changed meanwhile')
        console = wait_for_output(qapp, controller)
        assert os.listdir(path.join(str(tmp_path), 'cutie_demo', 'main')) == \
            ['0001_1-6_davis.mp4']
        assert 'Not marked done: the sequence changed' in console
        assert task_status(task.workspace) != 'done'
    finally:
        controller.gui.close()


def test_closing_the_window_cancels_the_output(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=40)
    controller = open_window(task, cpu_networks, [])
    controller.on_output_demo()
    controller.gui.close()
    assert controller.demo_worker is None
    demo = path.join(str(tmp_path), 'cutie_demo', 'main')
    assert not path.exists(demo) or not [name for name in os.listdir(demo)
                                         if name.endswith('.partial.mp4')]
    assert task_status(task.workspace) != 'done'
