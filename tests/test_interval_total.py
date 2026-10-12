"""
The task browser's statistics end with "Total Intervals: N": the intervals of the sequences marked
done, in every tab (main and long), whatever the tab shown; each is a clip of the dataset. A
sequence flagged for attention after Mark done still counts; a discarded one never does. The
statistics sit in a card of their own, apart from the messages.
"""
import json
from os import path

import pytest

from gui.task_state import (ATTENTION_FILE, INTERVAL_FILE, PROGRESS_FILE, TASK_SET_FILE,
                            discard_task, intervals_payload, summarize_task)
from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = pytest.mark.slow

SUBMITTED = {'last_edited': '2026-10-12T10:00:00+00:00', 'last_edited_by': 'alice',
             'edited_by': ['alice'], 'review_at': '2026-10-12T11:00:00+00:00', 'review_by': 'alice'}
DONE = dict(SUBMITTED, done_at='2026-10-12T12:00:00+00:00', done_by='bob')


def write(workspace, name, payload):
    with open(path.join(workspace, name), 'w', encoding='utf-8') as f:
        json.dump(payload, f)


def task(base, split, sequence, progress=None, intervals=None):
    made = make_task(base, sequence=sequence, split=split, frame_count=30, height=24, width=32)
    if progress:
        write(made.workspace, PROGRESS_FILE, progress)
    if intervals:
        write(made.workspace, INTERVAL_FILE, intervals_payload(intervals))
    return made


@pytest.fixture
def root(tmp_path):
    base = str(tmp_path)
    task(base, 'main', '0001', DONE, [(1, 5), (8, 12), (20, 25)])  # 3
    task(base, 'main', '0002', DONE)  # the curated interval: 1
    task(base, 'main', '0003', SUBMITTED, [(1, 5), (8, 12)])  # under review: 0
    flagged = task(base, 'main', '0004', DONE)  # flagged after Mark done: 1
    write(flagged.workspace, ATTENTION_FILE, {'at': '2026-10-12T13:00:00+00:00', 'by': 'carol'})
    task(base, 'long', '0005', DONE, [(1, 10), (15, 30)])  # 2
    discarded = task(base, 'long', '0006', DONE, [(1, 10), (15, 30)])  # discarded: 0
    discard_task(discarded.workspace, 'carol', True)
    task(base, 'main', '0007')  # not started: 0
    task_root = path.join(base, 'tasks')
    with open(path.join(task_root, TASK_SET_FILE), encoding='utf-8') as f:
        task_set = json.load(f)
    task_set['splits'] = ['main', 'long']
    write(task_root, TASK_SET_FILE, task_set)
    return task_root


def test_the_summary_tells_done_and_how_many_intervals(root):
    first = summarize_task(path.join(root, 'main', '0001'))
    assert first['done'] and first['intervals'] == 3
    flagged = summarize_task(path.join(root, 'main', '0004'))
    assert flagged['status'] == 'Attention' and flagged['done'] and flagged['intervals'] == 1
    assert not summarize_task(path.join(root, 'long', '0006'))['done']
    assert not summarize_task(path.join(root, 'main', '0003'))['done']


def test_the_browser_totals_the_done_intervals_of_every_tab(root, qapp):
    from gui.style import Card
    from gui.task_browser import TaskBrowser
    browser = TaskBrowser(compose_gui_config(device='cpu'), root)
    try:
        lines = browser.stats.text().split('<br>')
        assert len(lines) == 3 and lines[0].startswith('<b>main</b>')
        assert lines[1].startswith('By annotator:')
        assert lines[2] == 'Total Intervals: 7'
        browser.tabs.setCurrentWidget(browser.tables['long'])  # the same total on every tab
        assert browser.stats.text().endswith('Total Intervals: 7')

        # Mark done on another sequence adds its intervals
        write(path.join(root, 'main', '0003'), PROGRESS_FILE, DONE)
        browser.refresh_task('main', '0003')
        assert browser.stats.text().endswith('Total Intervals: 9')

        # the statistics and the messages are two cards
        def card_of(widget):
            while widget is not None and not isinstance(widget, Card):
                widget = widget.parentWidget()
            return widget

        messages, statistics = card_of(browser.console), card_of(browser.stats)
        assert messages is not None and statistics is not None and messages is not statistics
        assert messages.title_label.text() == 'Messages'
        assert statistics.title_label.text() == 'Statistics'
    finally:
        browser.close()
