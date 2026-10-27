"""
Task browser: lists the sequences of the input (gui/task_input.py), one tab per subset, and opens
one at a time in the labeling window; Import (gui/import_dialog.py) adds subsets, and Export
writes the dataset from the done sequences (gui/dataset_export.py). Each sequence's work is kept
in its work folder (gui/task_state.py: work_folder), made when it is first opened. The models are
loaded on the first open and reused for every task after that.

A task is open in one window at a time (TaskLock), and Open by shows which window has it.
Annotators lists everyone who has changed a task, and the Statistics card counts their tasks.
"""
import copy
import html
import os
import threading
from collections import Counter
from os import path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, open_dict
from PySide6.QtCore import QEvent, QObject, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QProgressDialog, QPushButton, QStyle,
                               QStackedWidget, QStyledItemDelegate, QStyleOptionViewItem,
                               QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from cutieplus.backends.base import BackendError
from gui.dataset_export import (ExportCancelled, ExportPlan, export_root_of, free_space_problem,
                                lock_clips, plan_export, release_locks, write_export)
from gui.gui_utils import fit_window, is_dark
from gui.main_controller import MainController, load_networks
from gui.import_dialog import ImportDialog
from gui.style import GAP, Card, set_variant
from gui.task_input import InputError, SequenceInput, find_tabs, read_tab, unorganized_parts
from gui.task_state import (ATTENTION_STATUS, DISCARDED_STATUS, TaskLocked, attention_of,
                            describe_attention, describe_holder, discard_of, discard_task,
                            ensure_work_folder, labeler_name, lock_holder, mark_done,
                            summarize_task, task_status, work_folder)
from gui.theme import (THEMES, apply_theme, has_theme_package, next_theme, save_theme,
                       saved_theme)

COLUMNS = ('Sequence', 'Frames', 'Anchors', 'Status', 'Annotators', 'Open by', 'Target')
SEQUENCE_COLUMN, FRAMES_COLUMN, STATUS_COLUMN, ANNOTATORS_COLUMN, OPEN_BY_COLUMN, TARGET_COLUMN = \
    0, 1, 3, 4, 5, 6
# on a Status item: how many intervals the sequence adds to the dataset (0 unless marked done)
CONFIRMED_INTERVALS_ROLE = Qt.ItemDataRole.UserRole + 1
LOCK_REFRESH_MS = 10000  # Open by follows the other windows
STATUS_COLORS = {
    'not started': QColor(140, 140, 140),
    'in progress': QColor(230, 150, 30),
    'under review': QColor(70, 130, 230),
    'done': QColor(60, 170, 80),
    ATTENTION_STATUS: QColor(211, 47, 47),  # Need Attention in the labeling window
    DISCARDED_STATUS: QColor(100, 100, 100),  # Discard, left out of the dataset; shown in italics
}
# discarded: dark gray, lighter in a dark theme where dark gray would vanish
DISCARDED_COLORS = (STATUS_COLORS[DISCARDED_STATUS], QColor(150, 150, 150))
HIDDEN_BY_HIDE_DONE = ('done', DISCARDED_STATUS)


class ExportWorker(QObject):
    """Writes the export in a thread of its own; the signals reach the browser in its thread."""
    progress = Signal(int, int)  # frames written, frames
    finished = Signal(object, str)  # the ExportReport, else None and the error ('' if cancelled)

    def __init__(self) -> None:
        super().__init__()
        self.cancel = threading.Event()

    def run(self, plan: ExportPlan, export_root: str) -> None:
        try:
            report = write_export(plan, export_root, self.progress.emit, self.cancel.is_set)
        except ExportCancelled:
            self.finished.emit(None, '')
            return
        except Exception as exc:  # shown in the messages; the previous export stays
            self.finished.emit(None, str(exc) or type(exc).__name__)
            return
        self.finished.emit(report, '')


def theme_button_text(theme: str) -> str:
    return f'Theme: {theme.capitalize()}'


def frames_text(summary: dict) -> str:
    """The Frames column: the sequence's frames until it is marked done, then the frames of each
    of its intervals (the dataset's clips) in order, e.g. 120/85/40."""
    if not summary['done']:
        return str(summary['frames'])
    return '/'.join(str(end - start + 1) for start, end in summary['clips'])


class UnsortedItem(QTableWidgetItem):
    """A cell of a column that does not sort: every row compares equal, so sorting on it (the
    sort is stable) leaves the rows where they are."""

    def __lt__(self, other) -> bool:
        return False


class TargetDelegate(QStyledItemDelegate):
    """Draws the Target column with the category in bold. The item's own text stays the plain
    'category: description', which sorting, filtering and the tooltip use; the parts come from
    the item's user data."""

    TEXT_MARGIN = 3

    def paint(self, painter, option, index):
        parts = index.data(Qt.ItemDataRole.UserRole)
        if not parts:
            return super().paint(painter, option, index)
        category, description = parts
        shown = QStyleOptionViewItem(option)
        self.initStyleOption(shown, index)
        shown.text = ''  # the style draws the background and the selection; the text is ours
        widget = shown.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, shown, painter, widget)

        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, shown, widget)
        rect = rect.adjusted(self.TEXT_MARGIN, 0, -self.TEXT_MARGIN, 0)
        # the selection is a light tint (gui/style.py): the text keeps its color
        color = shown.palette.color(QPalette.ColorRole.Text)
        bold = QFont(shown.font)
        bold.setBold(True)
        head = f'{category}: '
        head_width = QFontMetrics(bold).horizontalAdvance(head)
        room = max(rect.width() - head_width, 0)
        tail = QFontMetrics(shown.font).elidedText(description, Qt.TextElideMode.ElideRight, room)

        align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.save()
        painter.setPen(color)
        painter.setFont(bold)
        painter.drawText(QRect(rect.left(), rect.top(), head_width, rect.height()), align, head)
        painter.setFont(shown.font)
        painter.drawText(QRect(rect.left() + head_width, rect.top(), room, rect.height()),
                         align, tail)
        painter.restore()


class TaskTable(QTableWidget):
    """A tab's table of sequences, with a notice in the middle of its rows' area while it lists
    none (no sequence in the subset, all done or discarded and hidden, or none matching the
    filter)."""

    def __init__(self, rows: int, columns: int) -> None:
        super().__init__(rows, columns)
        self.notice = empty_notice(QLabel(self.viewport()))
        self.notice.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.notice.hide()

    def show_notice(self, text: Optional[str]) -> None:
        self.notice.setText(text or '')
        self.notice.setGeometry(self.viewport().rect())
        self.notice.setVisible(bool(text))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.notice.setGeometry(self.viewport().rect())


def empty_notice(label: QLabel) -> QLabel:
    # a muted message, centered both ways (gui/style.py)
    label.setProperty('emptyNotice', True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label


class StatusDelegate(QStyledItemDelegate):
    """Draws the Status column as a pill: the status in its color on a tint of that color, in
    italics for a discarded sequence. The item keeps its text, color and font, which sorting,
    filtering and the counts read."""

    MARGIN = 6  # px from the cell's left edge
    PADDING = 10  # px around the text, left and right

    def paint(self, painter, option, index):
        shown = QStyleOptionViewItem(option)
        self.initStyleOption(shown, index)
        text = shown.text
        shown.text = ''  # the style draws the background and the selection; the pill is ours
        widget = shown.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, shown, painter, widget)
        if not text:
            return
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        color = brush.color() if brush is not None else shown.palette.color(QPalette.ColorRole.Text)
        if widget is not None and is_dark(widget):
            color = color.lighter(135)  # the status colors are dark for a dark card
        metrics = QFontMetrics(shown.font)
        height = min(metrics.height() + 6, shown.rect.height() - 4)
        width = min(metrics.horizontalAdvance(text) + 2 * self.PADDING + 2,
                    shown.rect.width() - 2 * self.MARGIN)
        pill = QRectF(shown.rect.left() + self.MARGIN, shown.rect.center().y() - height / 2 + 1,
                      width, height)
        tint = QColor(color)
        tint.setAlpha(40)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tint)
        painter.drawRoundedRect(pill, height / 2, height / 2)
        painter.setPen(color)
        painter.setFont(shown.font)
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter,
                         metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                            int(width) - 2 * self.PADDING))
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QSize(hint.width() + 2 * (self.MARGIN + self.PADDING), hint.height())

    def column_width(self, font: QFont) -> int:
        # wide enough for every status, in italics too (discarded)
        italic = QFont(font)
        italic.setItalic(True)
        text = max(QFontMetrics(italic).horizontalAdvance(status) for status in STATUS_COLORS)
        return text + 2 * (self.MARGIN + self.PADDING) + 8


class TaskBrowser(QWidget):

    def __init__(self, cfg: DictConfig, task_root: str, initial_split: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.task_root = path.abspath(task_root)  # the input root
        self.networks = None
        self.controller: Optional[MainController] = None
        # (tab, sequence) -> the input sequence, and the frames of its anchor modality
        self.sources: Dict[Tuple[str, str], SequenceInput] = {}
        self.frame_counts: Dict[Tuple[str, str], int] = {}
        self.labeler = labeler_name(cfg['labeler'] if 'labeler' in cfg else None)
        self.setWindowTitle(f'Cutie tasks: {self.task_root} · {self.labeler}')
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter by sequence, annotator or target text')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        self.hide_done = QCheckBox('Hide done / discarded')
        self.hide_done.setToolTip('Hide the sequences that are done or discarded (in every tab).')
        self.hide_done.toggled.connect(self.apply_filter)
        # top right: cycles Auto, Light, Dark, Eye care for every window of this program
        self.theme = saved_theme()
        self.theme_button = QPushButton()
        self.theme_button.setToolTip('Color theme: Auto follows the desktop; Eye care is Light '
                                     'with every white or light-gray surface in #FAF9DE. Click to '
                                     'cycle Auto, Light, Dark, Eye care. Remembered.')
        self.theme_button.setMinimumWidth(
            max(QFontMetrics(self.theme_button.font()).horizontalAdvance(theme_button_text(name))
                for name in THEMES) + 32)
        self.theme_button.setEnabled(has_theme_package())
        self.theme_button.clicked.connect(self.on_theme)
        self.show_theme()
        self.target_delegate = TargetDelegate(self)
        self.status_delegate = StatusDelegate(self)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # no frame around the tables: they sit on a card
        # one table per subset of the input, in the order of find_tabs (refresh keeps it)
        self.tables: Dict[str, QTableWidget] = {}
        # each tab's sort, (column, order): a click on Frames, which does not sort, keeps it
        self.sort_keys: Dict[str, Tuple[int, Qt.SortOrder]] = {}

        open_button = set_variant(QPushButton('Open'), 'primary')
        open_button.clicked.connect(self.on_open)
        export_button = QPushButton('Export')
        export_button.setToolTip('Write the dataset: every interval of every done sequence, in '
                                 'every tab, as a clip of workspace/cutie_export/ (numbered from '
                                 '1). It replaces the previous export once complete.')
        export_button.clicked.connect(self.on_export)
        # a running export: its worker, progress dialog and the locks of its sequences
        self.export_worker: Optional[ExportWorker] = None
        self.export_progress: Optional[QProgressDialog] = None
        self.export_locks = []
        self.export_plan: Optional[ExportPlan] = None
        done_button = set_variant(QPushButton('Mark done (D)'), 'tonal')
        done_button.setToolTip('D: mark the selected sequence done; it must be under review (Save '
                               'in the labeling window).')
        done_button.clicked.connect(self.on_mark_done)
        # the filter box keeps the letter for itself while it has the focus
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(self.on_mark_done)
        # leaves the selected sequence out of the dataset; Restore on a discarded one
        self.discard_button = set_variant(QPushButton('Discard'), 'danger')
        self.discard_button.setToolTip(
            'Leave the selected sequence out of the dataset: its done mark and Need Attention are '
            'removed, and it cannot be marked done. On a discarded sequence the button restores '
            'it.')
        self.discard_button.clicked.connect(self.on_discard)
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh)
        import_button = QPushButton('Import')
        import_button.setToolTip('Add subsets of the input (each is a tab), or move the input '
                                 'found directly in the input folder into main/.')
        import_button.clicked.connect(self.on_import)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(140)
        # status and annotator counts of the current tab
        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setWordWrap(True)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.tabs.currentChanged.connect(self.update_stats)
        self.tabs.currentChanged.connect(self.show_discard_action)

        # the input and the export on the left, the selected sequence's actions on the right
        buttons = QHBoxLayout()
        for button in (import_button, export_button, refresh_button):
            set_variant(button, 'info')  # blue; a disabled one grey, as every disabled button
            buttons.addWidget(button)
        buttons.addStretch(1)
        for button in (open_button, done_button, self.discard_button):
            buttons.addWidget(button)
        # rounded cards (gui/style.py): the filter, the tasks with their actions, the messages
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(12)
        filter_row.addWidget(self.search, 1)
        filter_row.addWidget(self.hide_done)
        filter_row.addWidget(self.theme_button)
        layout.addWidget(Card(filter_row))
        # the tabs, or, before any subset is imported, a message in their place
        self.no_data = empty_notice(QLabel())
        self.task_pages = QStackedWidget()
        self.task_pages.setObjectName('taskPages')  # on the card's color (gui/style.py)
        self.task_pages.addWidget(self.tabs)
        self.task_pages.addWidget(self.no_data)
        tasks = QVBoxLayout()
        tasks.setSpacing(GAP)
        tasks.addWidget(self.task_pages, 1)
        tasks.addLayout(buttons)
        layout.addWidget(Card(tasks, stretch=True), 1)
        # the messages and the statistics, side by side so the table keeps its height
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        bottom.addWidget(Card(self.console, title='Messages'), 3)
        bottom.addWidget(Card(self.stats, title='Statistics'), 2)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self.refresh()
        if initial_split in self.tables:
            self.tabs.setCurrentWidget(self.tables[initial_split])
        self.lock_timer = QTimer(self)
        self.lock_timer.timeout.connect(self.refresh_locks)
        self.lock_timer.start(LOCK_REFRESH_MS)
        fit_window(self, QSize(1100, 750))

    def log(self, text: str) -> None:
        self.console.appendPlainText(text)

    @staticmethod
    def annotators_tip(names: List[str]) -> str:
        return 'changed by ' + ', '.join(names) if names else ''

    def status_color(self, status: str) -> QColor:
        if status == DISCARDED_STATUS:
            return DISCARDED_COLORS[int(is_dark(self))]
        return STATUS_COLORS[status]

    def changeEvent(self, event) -> None:
        # a new theme: the discarded gray follows it
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, 'stats'):
            for table in self.tables.values():
                for row in range(table.rowCount()):
                    item = table.item(row, STATUS_COLUMN)
                    if item is not None and item.text() == DISCARDED_STATUS:
                        item.setForeground(self.status_color(DISCARDED_STATUS))
            self.update_stats()

    def on_sort_indicator(self, split: str, column: int, order: Qt.SortOrder) -> None:
        # Frames does not sort (its cells compare equal): the arrow stays on the previous column
        if column != FRAMES_COLUMN:
            self.sort_keys[split] = (column, order)
            return
        header = self.tables[split].horizontalHeader()
        header.blockSignals(True)
        header.setSortIndicator(*self.sort_keys[split])
        header.blockSignals(False)

    def show_theme(self) -> None:
        self.theme_button.setText(theme_button_text(self.theme))

    def on_theme(self) -> None:
        self.theme = next_theme(self.theme)
        apply_theme(self.theme)
        save_theme(self.theme)
        self.show_theme()

    def task_dir(self, split: str, sequence: str) -> str:
        # the sequence's work folder (it exists once the sequence has been opened)
        return work_folder(self.task_root, split, sequence)

    def summary(self, split: str, sequence: str) -> dict:
        key = (split, sequence)
        return summarize_task(self.sources[key], self.task_dir(split, sequence),
                              self.frame_counts[key])

    def make_table(self, split: str) -> QTableWidget:
        table = TaskTable(0, len(COLUMNS))
        table.setHorizontalHeaderLabels(COLUMNS)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        # airy rows, no grid lines, alternate rows tinted (gui/style.py)
        table.verticalHeader().setDefaultSectionSize(32)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setHighlightSections(False)
        table.setItemDelegateForColumn(STATUS_COLUMN, self.status_delegate)
        table.setFrameShape(QFrame.Shape.NoFrame)
        table.horizontalHeader().setSectionResizeMode(TARGET_COLUMN, QHeaderView.ResizeMode.Stretch)
        table.setColumnWidth(STATUS_COLUMN, self.status_delegate.column_width(table.font()))
        table.doubleClicked.connect(self.on_open)
        table.itemSelectionChanged.connect(self.show_discard_action)
        table.setItemDelegateForColumn(TARGET_COLUMN, self.target_delegate)
        # sorting moves the rows, but the view keeps its hidden rows by position
        table.model().layoutChanged.connect(self.apply_filter)
        header = table.horizontalHeader()
        self.sort_keys[split] = (header.sortIndicatorSection(), header.sortIndicatorOrder())
        header.sortIndicatorChanged.connect(
            lambda column, order, split=split: self.on_sort_indicator(split, column, order))
        return table

    def refresh(self) -> None:
        """Reads the input again: its subsets (a tab each, a new one added, a gone one removed)
        and every sequence's row."""
        try:
            tabs = find_tabs(self.task_root)
        except InputError as exc:
            tabs = []
            self.log(str(exc))
        if not tabs and unorganized_parts(self.task_root):
            self.log(f'The input in {self.task_root} is not in a subset yet: press Import, then '
                     'Save, to move it into main/.')
            self.no_data.setText('No data imported.\nPress Import, then Save, to move the input '
                                 'into main/.')
        else:
            self.no_data.setText('No data imported.\nPress Import to add a subset.')
        self.task_pages.setCurrentWidget(self.tabs if tabs else self.no_data)
        names = [tab.name for tab in tabs]
        for split in [split for split in self.tables if split not in names]:
            self.tabs.removeTab(self.tabs.indexOf(self.tables.pop(split)))
        for index, tab in enumerate(tabs):
            if tab.name not in self.tables:
                self.tables[tab.name] = self.make_table(tab.name)
                self.tabs.insertTab(index, self.tables[tab.name], tab.name)
        self.tables = {name: self.tables[name] for name in names}
        self.sources, self.frame_counts = {}, {}
        for tab in tabs:
            contents = read_tab(tab)
            for problem in contents.problems:
                self.log(problem)
            table = self.tables[tab.name]
            table.setSortingEnabled(False)
            table.setRowCount(len(contents.sequences))
            for row, source in enumerate(contents.sequences):
                key = (tab.name, source.sequence)
                self.sources[key] = source
                summary = summarize_task(source, self.task_dir(*key))
                self.frame_counts[key] = summary['frames']
                self.fill_row(table, row, summary)
            table.setSortingEnabled(True)
            self.tabs.setTabText(self.tabs.indexOf(table),
                                 f'{tab.name} ({len(contents.sequences)})')
        self.apply_filter()
        self.update_stats()

    def update_stats(self) -> None:
        # the whole tab, whatever the filter hides; read from the Status column already loaded
        table = self.tabs.currentWidget()
        split = next((name for name, candidate in self.tables.items() if candidate is table), None)
        if split is None:
            self.stats.clear()
            return
        self.stats.setToolTip(f'Total Intervals: the intervals of the sequences marked done, in '
                              f'every tab ({", ".join(self.tables)}); each is a clip of the '
                              'dataset. Discarded sequences never count.')
        counts = Counter(table.item(row, STATUS_COLUMN).text() for row in range(table.rowCount()))
        parts = [f'<span style="color:{self.status_color(status).name()}">{status} '
                 f'{counts[status]}</span>' for status in reversed(STATUS_COLORS)]
        # the sequences each annotator has worked on, whatever their status now; a sequence
        # counts for each of its annotators
        annotators = Counter()
        unrecorded = 0
        for row in range(table.rowCount()):
            names = table.item(row, ANNOTATORS_COLUMN).data(Qt.ItemDataRole.UserRole) or []
            annotators.update(names)
            if not names and table.item(row, STATUS_COLUMN).text() != 'not started':
                unrecorded += 1  # changed before labelers were recorded
        by_annotator = [f'{html.escape(name)} {count}' for name, count in
                        sorted(annotators.items(), key=lambda entry: (-entry[1], entry[0]))]
        if unrecorded:
            by_annotator.append(f'not recorded {unrecorded}')
        self.stats.setText(f'<b>{split}</b>: {table.rowCount()} tasks · ' + ' · '.join(parts) +
                           '<br>By annotator: ' + (' · '.join(by_annotator) or 'none yet') +
                           f'<br>Total Intervals: {self.confirmed_intervals()}')

    def confirmed_intervals(self) -> int:
        """The intervals of the sequences marked done, in every tab: the dataset's clips."""
        return sum(table.item(row, STATUS_COLUMN).data(CONFIRMED_INTERVALS_ROLE) or 0
                   for table in self.tables.values() for row in range(table.rowCount()))

    @staticmethod
    def open_by_text(holder: Optional[dict]) -> str:
        if holder is None:
            return ''
        if not holder:
            return 'another window'
        text = f'{holder.get("labeler", "?")} (pid {holder.get("pid", "?")})'
        return f'{text} {holder["activity"]}' if holder.get('activity') else text

    def fill_row(self, table: QTableWidget, row: int, summary: dict) -> None:
        values = (summary['sequence'], frames_text(summary),
                  f'{summary["mask_anchors"]} + {summary["empty_anchors"]} empty', summary['status'],
                  ', '.join(summary['annotators']), self.open_by_text(summary['open_by']),
                  summary['target'])
        for column, value in enumerate(values):
            item = UnsortedItem() if column == FRAMES_COLUMN else QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, value)
            if column == FRAMES_COLUMN and summary['done']:
                ranges = ', '.join(f'{start}-{end}' for start, end in summary['clips'])
                item.setToolTip(f'Marked done: the frames of each interval ({ranges}) of '
                                f'{summary["frames"]}.')
            if column == STATUS_COLUMN:
                item.setForeground(self.status_color(value))
                item.setData(CONFIRMED_INTERVALS_ROLE,
                             summary['intervals'] if summary['done'] else 0)
                if value == DISCARDED_STATUS:
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                tips = []
                if summary['discard']:
                    tips.append('left out of the dataset, '
                                f'{describe_attention(summary["discard"], "discarded")}')
                if summary['attention']:
                    tips.append(f'needs attention, {describe_attention(summary["attention"])}')
                if summary['last_edited_by']:
                    tips.append(f'last edited by {summary["last_edited_by"]}')
                item.setToolTip('\n'.join(tips))
            if column == ANNOTATORS_COLUMN:
                # the names, for the per-annotator counts
                item.setData(Qt.ItemDataRole.UserRole, summary['annotators'])
                item.setToolTip(self.annotators_tip(summary['annotators']))
            if column == OPEN_BY_COLUMN and summary['open_by'] is not None:
                item.setToolTip(describe_holder(summary['open_by']))
            if column == TARGET_COLUMN:
                item.setToolTip(value)
                item.setData(Qt.ItemDataRole.UserRole, (summary['category'], summary['description']))
            table.setItem(row, column, item)

    def refresh_locks(self) -> None:
        # other windows open and close tasks; a row whose holder changed is reread whole, since
        # its anchors and status may have changed too
        for split, table in self.tables.items():
            changed = []
            for row in range(table.rowCount()):
                sequence = table.item(row, SEQUENCE_COLUMN).text()
                holder = lock_holder(self.task_dir(split, sequence))
                if table.item(row, OPEN_BY_COLUMN).text() != self.open_by_text(holder):
                    changed.append((row, sequence))
            if changed:
                table.setSortingEnabled(False)
                for row, sequence in changed:
                    self.fill_row(table, row, self.summary(split, sequence))
                table.setSortingEnabled(True)
        self.apply_filter()
        self.update_stats()

    def refresh_task(self, split: str, sequence: str) -> None:
        table = self.tables.get(split)
        if table is None:
            return
        for row in range(table.rowCount()):
            if table.item(row, SEQUENCE_COLUMN).text() == sequence:
                table.setSortingEnabled(False)
                self.fill_row(table, row, self.summary(split, sequence))
                table.setSortingEnabled(True)
                self.apply_filter()  # e.g. a sequence just marked done leaves a list that hides done
                self.update_stats()
                self.show_discard_action()
                return

    def apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        hide_done = self.hide_done.isChecked()
        for table in self.tables.values():
            for row in range(table.rowCount()):
                if table.item(row, SEQUENCE_COLUMN) is None:
                    continue  # a row still being filled
                searched = ' '.join(table.item(row, column).text().lower() for column in
                                    (SEQUENCE_COLUMN, ANNOTATORS_COLUMN, TARGET_COLUMN))
                hidden = ((bool(text) and text not in searched)
                          or (hide_done
                              and table.item(row, STATUS_COLUMN).text() in HIDDEN_BY_HIDE_DONE))
                table.setRowHidden(row, hidden)
            # a hidden row cannot stay selected: Open / Mark done / Discard would act on it unseen
            if any(table.isRowHidden(index.row()) for index in table.selectionModel().selectedRows()):
                table.clearSelection()
        for split, table in self.tables.items():
            listed = any(not table.isRowHidden(row) for row in range(table.rowCount()))
            table.show_notice(None if listed else self.empty_text(split, table))

    def empty_text(self, split: str, table: QTableWidget) -> str:
        # why a tab lists no sequence
        if table.rowCount() == 0:
            return f'No sequences in {split}.'
        statuses = [table.item(row, STATUS_COLUMN).text() for row in range(table.rowCount())
                    if table.item(row, STATUS_COLUMN) is not None]
        if self.hide_done.isChecked() and all(status in HIDDEN_BY_HIDE_DONE for status in statuses):
            return (f'Every sequence in {split} is done or discarded: nothing left to label.\n'
                    'Untick Hide done / discarded to list them.')
        return f'No sequence matches "{self.search.text().strip()}".'

    def selected_task(self) -> Optional[Tuple[str, str]]:
        table = self.tabs.currentWidget()
        rows = table.selectionModel().selectedRows() if table is not None else []
        if not rows:
            self.log('Select a task first.')
            return None
        split = next(name for name, candidate in self.tables.items() if candidate is table)
        return split, table.item(rows[0].row(), SEQUENCE_COLUMN).text()

    def is_open(self, split: str, sequence: str) -> bool:
        return (self.controller is not None
                and (self.controller.task.split, self.controller.task.sequence_id) == (split, sequence))

    def refuse_if_held_elsewhere(self, split: str, sequence: str, action: str) -> bool:
        # another window has the task: its unsaved work is out of reach
        holder = lock_holder(self.task_dir(split, sequence))
        if holder is None or holder.get('pid') == os.getpid():
            return False
        self.log(f'{split}/{sequence} is open by {describe_holder(holder)}; {action} once they '
                 'close it.')
        self.refresh_task(split, sequence)
        return True

    def on_open(self) -> None:
        selection = self.selected_task()
        if selection is None:
            return
        split, sequence = selection
        if self.is_open(split, sequence):
            self.controller.gui.raise_()
            self.controller.gui.activateWindow()
            return
        # checked before closing the current task, which would be closed for nothing
        if self.refuse_if_held_elsewhere(split, sequence, 'open it'):
            return
        if self.controller is not None:
            if self.controller.propagating:
                self.log('Pause the propagation before switching tasks.')
                return
            # writes pending saves; on_task_event clears the controller
            self.controller.gui.close()
        if self.networks is None:
            self.log('Loading models...')
            QApplication.processEvents()
            self.networks = load_networks(self.cfg)
        cfg = copy.deepcopy(self.cfg)
        with open_dict(cfg):
            cfg['workspace'] = ensure_work_folder(self.task_root, self.sources[(split, sequence)])
            cfg['labeler'] = self.labeler
        self.log(f'Opening {split}/{sequence}.')
        try:
            self.controller = MainController(cfg, networks=self.networks,
                                             on_task_event=self.on_task_event)
        except TaskLocked as exc:
            # another window took it after the check above
            self.log(str(exc))
            self.refresh_task(split, sequence)
            return
        except BackendError as exc:
            self.log(f'{split}/{sequence} could not be opened: {exc}')
            return
        self.controller.gui.refresh_reset_button_state()
        self.refresh_task(split, sequence)

    def on_task_event(self, controller: MainController, event: str) -> None:
        if event == 'closed' and self.controller is controller:
            self.controller = None
        self.refresh_task(controller.task.split, controller.task.sequence_id)

    def on_import(self) -> None:
        dialog = ImportDialog(self.task_root, self)
        dialog.imported.connect(self.on_imported)
        dialog.exec()

    def on_imported(self, summary: str) -> None:
        self.log(summary)
        self.refresh()

    def on_mark_done(self) -> None:
        # accepts a task under review
        selection = self.selected_task()
        if selection is None:
            return
        split, sequence = selection
        if discard_of(self.task_dir(split, sequence)):
            self.log(f'{split}/{sequence} is discarded: restore it first.')
            return
        if attention_of(self.task_dir(split, sequence)):
            self.log(f'{split}/{sequence} needs attention: open it and toggle Need Attention off '
                     'first.')
            return
        if self.refuse_if_held_elsewhere(split, sequence, 'mark it done'):
            return
        if self.is_open(split, sequence):
            if self.controller.propagating:
                self.log('Pause the propagation first.')
                return
            self.controller.res_man.save_queue.join()
        workspace = self.task_dir(split, sequence)
        status = task_status(workspace)
        if status != 'under review':
            self.log(f'{split}/{sequence} is {status}: save it in the labeling window (Ctrl+S) to '
                     'submit it for review first.')
            return
        mark_done(workspace, self.labeler)
        self.log(f'{split}/{sequence} is done.')
        self.refresh_task(split, sequence)

    def selected_status(self) -> Optional[str]:
        # the Status of the selected row, without asking for a selection
        table = self.tabs.currentWidget()
        rows = table.selectionModel().selectedRows() if table is not None else []
        return table.item(rows[0].row(), STATUS_COLUMN).text() if rows else None

    def show_discard_action(self) -> None:
        # Discard, or Restore on a discarded sequence
        restore = self.selected_status() == DISCARDED_STATUS
        self.discard_button.setText('Restore' if restore else 'Discard')
        if self.discard_button.property('variant') != ('tonal' if restore else 'danger'):
            set_variant(self.discard_button, 'tonal' if restore else 'danger')

    def confirm_discard(self, split: str, sequence: str) -> bool:
        answer = QMessageBox.question(
            self, 'Discard', f'Discard {split}/{sequence}?\n\nIt is left out of the dataset: its '
            'done mark and Need Attention are removed, and it cannot be marked done. Its masks '
            'and anchors stay; Restore brings it back.',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def on_discard(self) -> None:
        # the selected sequence is left out of the dataset, or restored when it is discarded
        selection = self.selected_task()
        if selection is None:
            return
        split, sequence = selection
        workspace = self.task_dir(split, sequence)
        discarded = discard_of(workspace) is None  # what this click does
        action = 'discard it' if discarded else 'restore it'
        if self.refuse_if_held_elsewhere(split, sequence, action):
            return
        if self.is_open(split, sequence) and self.controller.propagating:
            self.log('Pause the propagation first.')
            return
        if discarded and not self.confirm_discard(split, sequence):
            return
        ensure_work_folder(self.task_root, self.sources[(split, sequence)])
        discard_task(workspace, self.labeler, discarded)
        if discarded:
            if self.is_open(split, sequence):
                self.controller.task.attention = None  # its window's Need Attention is off
                self.controller.gui.refresh_attention_button()
            self.log(f'{split}/{sequence} is discarded: it is left out of the dataset.')
        else:
            self.log(f'{split}/{sequence} is restored; it needs a review and Mark done again.')
        self.refresh_task(split, sequence)

    def on_export(self) -> None:
        """The dataset (gui/dataset_export.py): the intervals of every done sequence, in every tab,
        after a confirmation. Every exported sequence is locked while it is written."""
        if self.export_worker is not None:
            self.export_progress.raise_()
            return
        self.log('Export: reading the done sequences...')
        QApplication.processEvents()
        plan = plan_export(self.task_root)
        for warning in plan.warnings:
            self.log(warning)
        if plan.problems:
            for problem in plan.problems:
                self.log(problem)
            self.log('Not exported: fix the problems above first.')
            return
        if not plan.clips:
            self.log('Nothing to export: no sequence is done.')
            return
        export_root = export_root_of(self.task_root)
        problem = free_space_problem(plan, export_root)
        if problem:
            self.log(f'Not exported: {problem}')
            return
        if not self.confirm_export(plan, export_root):
            return
        try:
            self.export_locks = lock_clips(plan, self.labeler)
        except TaskLocked as exc:
            self.log(f'Not exported: {exc} Close it, then press Export again.')
            return
        self.refresh_locks()  # Open by: exporting
        self.export_plan = plan
        self.export_progress = QProgressDialog(
            f'Exporting {len(plan.clips)} clips into {export_root}...', 'Cancel', 0, plan.frames,
            self)
        self.export_progress.setWindowTitle('Export')
        self.export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_progress.setMinimumDuration(0)
        self.export_progress.setAutoClose(False)
        self.export_progress.setAutoReset(False)
        self.export_worker = ExportWorker()
        self.export_progress.canceled.connect(self.export_worker.cancel.set)
        self.export_worker.progress.connect(self.on_export_progress)
        self.export_worker.finished.connect(self.on_exported)
        self.export_progress.show()
        threading.Thread(target=self.export_worker.run, args=(plan, export_root),
                         daemon=True).start()

    def confirm_export(self, plan: ExportPlan, export_root: str) -> bool:
        names = plan.names()
        replaced = (' The previous export there is replaced once this one is complete.'
                    if path.exists(export_root) else '')
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle('Export')
        dialog.setText(f'Export {len(plan.clips)} clip{"" if len(plan.clips) == 1 else "s"}, '
                       f'{names[0]} to {names[-1]}, into {export_root}?')
        dialog.setInformativeText(
            f'Each is an interval of one of the {plan.sequences} done sequences, in every tab: '
            f'{plan.frames} frames, whose images are copied ({plan.bytes / 2**30:.1f} GB), with '
            f'their masks and a seqinfo file.{replaced}')
        export_button = dialog.addButton('Export', QMessageBox.ButtonRole.AcceptRole)
        cancel_button = dialog.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is export_button

    def on_export_progress(self, done: int, total: int) -> None:
        if self.export_progress is not None:
            self.export_progress.setValue(done)

    def on_exported(self, report, error: str) -> None:
        release_locks(self.export_locks)
        self.export_locks = []
        self.export_progress.close()
        self.export_worker = self.export_progress = None
        export_root = export_root_of(self.task_root)
        if report is None:
            self.log(f'Export stopped: {error}. The previous export is unchanged.' if error else
                     'Export cancelled; the previous export is unchanged.')
        else:
            names = self.export_plan.names()
            notes = []
            if report.pregenerated:
                notes.append(f'{report.pregenerated} mask frames are the pregenerated masks')
            if report.empty:
                notes.append(f'{report.empty} frames had no mask and are empty')
            self.log(f'Exported {report.clips} clips ({names[0]} to {names[-1]}, {report.frames} '
                     f'frames) into {export_root}.' + (f' {"; ".join(notes)}.' if notes else ''))
        self.export_plan = None
        self.refresh_locks()

    def closeEvent(self, event):
        if self.export_worker is not None:
            self.log('Wait for the export to finish, or cancel it, before closing.')
            event.ignore()
            return
        # the open task writes its pending saves as its window closes
        if self.controller is not None:
            self.controller.gui.close()
        event.accept()
