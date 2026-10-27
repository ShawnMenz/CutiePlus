"""
The task browser's Frames column: the sequence's frame count until it is marked done, then the
frames of each of its intervals (the dataset's clips) in order, joined by '/'. A click on its header
does not sort; the other columns still do.
"""
import pytest

from gui.task_state import discard_task, intervals_payload, work_folder, write_sections
from tests.conftest import compose_gui_config
from tests.helpers import make_task, summary_of

pytestmark = pytest.mark.slow

SUBMITTED = {'last_edited': '2026-10-12T10:00:00+00:00', 'last_edited_by': 'alice',
             'edited_by': ['alice'], 'review_at': '2026-10-12T11:00:00+00:00', 'review_by': 'alice'}
DONE = dict(SUBMITTED, done_at='2026-10-12T12:00:00+00:00', done_by='bob')

# sequence: (progress, saved intervals, Frames text)
EXPECTED = {
    '0001': (DONE, [(1, 5), (8, 12), (20, 25)], '5/5/6'),
    '0002': (DONE, [(3, 12)], '10'),
    '0003': (DONE, [(5, 24)], '20'),
    '0004': (DONE, None, '30'),  # no interval: the whole sequence
    '0005': (SUBMITTED, [(1, 5), (8, 12)], '30'),  # not done yet
    '0006': (DONE, [(1, 10), (15, 30)], '30'),  # discarded: no done mark
    '0007': (None, None, '30'),  # not started
}


@pytest.fixture
def root(tmp_path):
    base = str(tmp_path)
    for sequence, (progress, intervals, _) in EXPECTED.items():
        made = make_task(base, sequence=sequence, frame_count=30, height=24, width=32)
        if progress:
            write_sections(made.workspace, progress=progress)
        if intervals:
            write_sections(made.workspace, intervals=intervals_payload(intervals))
    root = str(tmp_path / 'tasks')
    discard_task(work_folder(root, 'main', '0006'), 'carol', True)
    return root


def test_the_summary_lists_the_clips(root):
    assert summary_of(work_folder(root, 'main', '0001'))['clips'] == [(1, 5), (8, 12), (20, 25)]
    assert summary_of(work_folder(root, 'main', '0003'))['clips'] == [(5, 24)]
    summary = summary_of(work_folder(root, 'main', '0004'))
    assert summary['clips'] == [(1, 30)] and summary['intervals'] == 1 and summary['frames'] == 30


def test_frames_shows_the_interval_lengths_once_done_and_does_not_sort(root, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from gui.task_browser import FRAMES_COLUMN, SEQUENCE_COLUMN, TaskBrowser
    browser = TaskBrowser(compose_gui_config(device='cpu'), root)
    try:
        browser.show()
        table = browser.tables['main']

        def column(number):
            return [table.item(row, number).text() for row in range(table.rowCount())]

        shown = dict(zip(column(SEQUENCE_COLUMN), column(FRAMES_COLUMN)))
        assert shown == {sequence: expected[2] for sequence, expected in EXPECTED.items()}
        first = next(row for row in range(table.rowCount())
                     if table.item(row, SEQUENCE_COLUMN).text() == '0001')
        assert table.item(first, FRAMES_COLUMN).toolTip() == \
            'Marked done: the frames of each interval (1-5, 8-12, 20-25) of 30.'

        # sorted by sequence, newest first; a click on Frames moves no row and keeps the arrow
        table.sortItems(SEQUENCE_COLUMN, Qt.SortOrder.DescendingOrder)
        order = column(SEQUENCE_COLUMN)
        assert order == sorted(EXPECTED, reverse=True)
        header = table.horizontalHeader()
        for _ in range(2):  # the second click would flip the order
            x = header.sectionViewportPosition(FRAMES_COLUMN) + header.sectionSize(FRAMES_COLUMN) // 2
            QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, 5))
            assert column(SEQUENCE_COLUMN) == order
            assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == \
                (SEQUENCE_COLUMN, Qt.SortOrder.DescendingOrder)

        # the other columns still sort, Sequence included
        x = header.sectionViewportPosition(SEQUENCE_COLUMN) + header.sectionSize(SEQUENCE_COLUMN) // 2
        QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, 5))
        assert column(SEQUENCE_COLUMN) == sorted(EXPECTED)

        # a refresh keeps the order and the texts
        browser.refresh()
        assert column(SEQUENCE_COLUMN) == sorted(EXPECTED)
        assert dict(zip(column(SEQUENCE_COLUMN), column(FRAMES_COLUMN))) == shown
    finally:
        browser.close()
