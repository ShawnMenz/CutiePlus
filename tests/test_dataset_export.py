"""
The dataset export (gui/dataset_export.py, the task browser's Export): every interval of every done
sequence, in every tab, is a clip of cutie_export/, numbered from 1 and zero-padded to the digits
of the count. A clip holds the interval's frames (copied, numbered from 1), its masks (L 0/255:
saved, else pregenerated, else empty) and a seqinfo naming the sequence it was cut from.
"""
import filecmp
import os
from os import path

import numpy as np
import pytest
from PIL import Image

from gui.dataset_export import (ExportCancelled, ExportPlan, clip_names, export_root_of,
                                free_space_problem, lock_clips, plan_export, write_export)
from gui.task_state import (TaskLock, TaskLocked, discard_task, intervals_payload, lock_holder,
                            work_folder, write_sections)
from tests.conftest import compose_gui_config
from tests.helpers import DESCRIPTION, make_task

SUBMITTED = {'last_edited': '2026-10-12T10:00:00+00:00', 'last_edited_by': 'alice',
             'edited_by': ['alice'], 'review_at': '2026-10-12T11:00:00+00:00', 'review_by': 'alice'}
DONE = dict(SUBMITTED, done_at='2026-10-12T12:00:00+00:00', done_by='bob')
HEIGHT, WIDTH = 24, 32


def task(base, sequence, split='main', progress=DONE, intervals=None, **options):
    made = make_task(base, sequence=sequence, split=split, frame_count=30, height=HEIGHT,
                     width=WIDTH, **options)
    write_sections(made.workspace, progress=progress)
    if intervals is not None:
        write_sections(made.workspace, intervals=intervals)
    return made


@pytest.fixture
def root(tmp_path):
    base = str(tmp_path)
    # two intervals: the added one with a description and a category of its own
    task(base, '0001', intervals=intervals_payload([(1, 5), (8, 12)], {(8, 12): 'Parked.'},
                                                   {(8, 12): 'truck'}))
    task(base, '0002', prefill=(2, ))  # the whole sequence; frame 3 has a pregenerated mask
    task(base, '0003', progress=SUBMITTED)  # under review: not exported
    discarded = task(base, '0004')
    discard_task(discarded.workspace, 'carol', True)  # left out
    # the same sequence name in another tab, whose input seqinfo names its dataset
    task(base, '0001', split='long', intervals=intervals_payload([(3, 6)]), mask_anchors=())
    seqinfo = path.join(base, 'tasks', 'long', 'seqinfo')
    os.makedirs(seqinfo)
    with open(path.join(seqinfo, '0001_seqinfo.ini'), 'w', encoding='utf-8') as stream:
        stream.write('[Sequence]\nname=0001\ndataset=VTUAV\noriginalSource=pedestrian_122\n')
    return path.join(base, 'tasks')


def frames_of(root, split, sequence):
    folder = path.join(root, split, 'visual', sequence)
    return [path.join(folder, name) for name in sorted(os.listdir(folder))]


def test_clips_are_numbered_from_one_and_padded_to_their_count():
    assert clip_names(9) == [str(number) for number in range(1, 10)]
    assert clip_names(12)[:2] == ['01', '02'] and clip_names(12)[-1] == '12'
    names = clip_names(3000)
    assert (names[0], names[998], names[-1]) == ('0001', '0999', '3000')


def test_the_plan_takes_every_interval_of_the_done_sequences(root):
    plan = plan_export(root)
    assert not plan.problems and plan.sequences == 3
    assert [(clip.original_source, clip.start, clip.end, clip.category, clip.description)
            for clip in plan.clips] == [
        ('main/0001', 1, 5, 'car', DESCRIPTION), ('main/0001', 8, 12, 'truck', 'Parked.'),
        ('main/0002', 1, 30, 'car', DESCRIPTION), ('long/0001', 3, 6, 'car', DESCRIPTION)]
    assert plan.names() == ['1', '2', '3', '4'] and plan.frames == 5 + 5 + 30 + 4
    sizes = [path.getsize(file) for file in frames_of(root, 'main', '0002')]
    assert plan.bytes > 2 * sum(sizes)  # both modalities of every clip


def test_the_export_crops_the_frames_and_writes_masks_and_seqinfo(root):
    plan = plan_export(root)
    export = export_root_of(root)
    progress = []
    report = write_export(plan, export, lambda done, total: progress.append((done, total)))
    assert sorted(os.listdir(export)) == ['annotation', 'seqinfo', 'thermal', 'visual']
    assert sorted(os.listdir(path.join(export, 'visual'))) == ['1', '2', '3', '4']
    assert progress[-1] == (44, 44) and (report.clips, report.frames) == (4, 44)

    # clip 2: frames 8-12 of main/0001, numbered from 1, the same bytes
    clip = path.join(export, 'visual', '2')
    assert sorted(os.listdir(clip)) == [f'00000{n}.jpg' for n in range(1, 6)]
    source = frames_of(root, 'main', '0001')
    assert all(filecmp.cmp(path.join(clip, f'00000{n}.jpg'), source[n + 6], shallow=False)
               for n in range(1, 6))
    assert len(os.listdir(path.join(export, 'thermal', '2'))) == 5

    # masks: L 0/255; clip 3 (main/0002) has the saved anchor, the pregenerated mask, the rest empty
    masks = path.join(export, 'annotation', '3')
    assert len(os.listdir(masks)) == 30
    first = Image.open(path.join(masks, '000001.png'))
    assert first.mode == 'L' and first.size == (WIDTH, HEIGHT)
    assert set(np.unique(np.array(first))) == {0, 255}
    assert np.array(Image.open(path.join(masks, '000003.png'))).max() == 255  # pregenerated
    assert np.array(Image.open(path.join(masks, '000002.png'))).max() == 0  # no mask: empty
    # saved anchors on frame 1 of main/0001 and main/0002, a pregenerated one on 0002's frame 3
    assert (report.pregenerated, report.empty) == (1, 44 - 2 - 1)

    with open(path.join(export, 'seqinfo', '2_seqinfo.ini'), encoding='utf-8') as stream:
        assert stream.read() == (
            '[Sequence]\nname=2\ndataset=\noriginalSource=main/0001\nsourceLsId=0001\n'
            'category=truck\ndescription=Parked.\nproblem=\nseqLength=5\nintervalStart=8\n'
            'intervalEnd=12\nimExtVisual=.jpg\nimExtThermal=.jpg\n'
            f'imWidthVisual={WIDTH}\nimHeightVisual={HEIGHT}\n'
            f'imWidthThermal={WIDTH}\nimHeightThermal={HEIGHT}\n')
    with open(path.join(export, 'seqinfo', '4_seqinfo.ini'), encoding='utf-8') as stream:
        text = stream.read()
    # the input's seqinfo names the dataset; the tab tells the two sequences 0001 apart
    assert 'dataset=VTUAV\noriginalSource=long/0001\nsourceLsId=0001\n' in text
    assert 'intervalStart=3\nintervalEnd=6\n' in text


def test_a_new_export_replaces_the_previous_one_and_a_cancel_keeps_it(root):
    export = export_root_of(root)
    write_export(plan_export(root), export)
    open(path.join(export, 'stale.txt'), 'w').close()
    with pytest.raises(ExportCancelled):
        write_export(plan_export(root), export, cancelled=lambda: True)
    assert path.exists(path.join(export, 'stale.txt'))  # unchanged
    assert sorted(os.listdir(path.dirname(export))) == ['cutie_export', 'cutie_output', 'tasks']

    write_sections(work_folder(root, 'main', '0003'), progress=DONE)  # one more clip
    write_export(plan_export(root), export)
    assert not path.exists(path.join(export, 'stale.txt'))
    assert sorted(os.listdir(path.join(export, 'visual'))) == ['1', '2', '3', '4', '5']
    with open(path.join(export, 'seqinfo', '4_seqinfo.ini'), encoding='utf-8') as stream:
        assert 'originalSource=main/0003' in stream.read()  # in the browser's order


def test_a_sequence_open_in_a_window_stops_the_export(root):
    plan = plan_export(root)
    window = TaskLock.acquire(work_folder(root, 'main', '0002'), 'alice')
    try:
        with pytest.raises(TaskLocked):
            lock_clips(plan, 'bob')
        assert lock_holder(work_folder(root, 'main', '0001')) is None  # none kept
    finally:
        window.release()
    locks = lock_clips(plan, 'bob')
    assert lock_holder(work_folder(root, 'main', '0001'))['activity'] == 'exporting'
    for lock in locks:
        lock.release()


def test_problems_stop_the_export_before_anything_is_written(root, monkeypatch):
    write_sections(work_folder(root, 'main', '0002'), intervals=intervals_payload([(25, 40)]))
    plan = plan_export(root)
    assert plan.problems == ['main/0002: 25-40 is not an interval of frames 1-30.']
    plan = plan_export(path.join(path.dirname(root), 'nowhere'))
    assert plan.problems and not plan.clips

    # the new export is written before the previous one goes: its frames must fit
    from gui import dataset_export
    monkeypatch.setattr(dataset_export.shutil, 'disk_usage',
                        lambda folder: type('Usage', (), {'free': 10})())
    assert 'make room' in free_space_problem(ExportPlan(bytes=11), export_root_of(root))
    assert free_space_problem(ExportPlan(bytes=10), export_root_of(root)) is None


@pytest.mark.slow
def test_the_browser_exports_after_a_confirmation(root, qapp):
    import time
    from PySide6.QtWidgets import QPushButton
    from gui.task_browser import TaskBrowser
    browser = TaskBrowser(compose_gui_config(device='cpu'), root)
    asked = []
    browser.confirm_export = lambda plan, export: asked.append(len(plan.clips)) or True
    try:
        export_button = next(button for button in browser.findChildren(QPushButton)
                             if button.text() == 'Export')
        assert export_button.isEnabled()
        export_button.click()
        deadline = time.monotonic() + 30
        while browser.export_worker is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert asked == [4] and browser.export_worker is None
        log = browser.console.toPlainText()
        assert f'Exported 4 clips (1 to 4, 44 frames) into {export_root_of(root)}.' in log
        assert '1 mask frames are the pregenerated masks' in log
        assert lock_holder(work_folder(root, 'main', '0001')) is None  # released
    finally:
        browser.close()
