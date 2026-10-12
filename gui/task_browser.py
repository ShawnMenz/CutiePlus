"""
Task browser: lists the Cutie tasks under a task root (<root>/<split>/<sequence>/, made by
RGBT_MOT's tools/cutie_tasks.py), opens one at a time in the labeling window, and exports them.
The models are loaded on the first open and reused for every task after that.

Several browsers (one per labeler session) can share a task root: a task is open in one window
at a time (TaskLock), and Open by shows who has it. Annotators lists everyone who has changed a
task, and the bottom line counts their tasks.
"""
import copy
import html
import os
from collections import Counter
from os import path
from typing import Dict, Optional, Tuple

from omegaconf import DictConfig, open_dict
from PySide6.QtCore import QEvent, QProcess, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QTabWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from cutieplus.backends.base import BackendError
from gui.annotator_manager import AnnotatorManager
from gui.gui_utils import fit_window, is_dark
from gui.main_controller import MainController, load_networks
from gui.style import GAP, Card, set_variant
from gui.task_export import start_export
from gui.task_state import (ATTENTION_STATUS, DISCARDED_STATUS, TASK_FILE, TaskLock, TaskLocked,
                            attention_of, curated_split, describe_attention, describe_holder,
                            discard_of, discard_task, export_command,
                            export_is_current, labeler_name, load_task_set, lock_holder, mark_done,
                            summarize_task, task_status)
from gui.theme import (THEMES, apply_theme, has_theme_package, next_theme, save_theme,
                       saved_theme)

COLUMNS = ('Sequence', 'Frames', 'Anchors', 'Status', 'Annotators', 'Open by', 'Modality', 'Target')
SEQUENCE_COLUMN, STATUS_COLUMN, ANNOTATORS_COLUMN, OPEN_BY_COLUMN, TARGET_COLUMN = 0, 3, 4, 5, 7
LOCK_REFRESH_MS = 10000  # Open by follows the other sessions
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


def theme_button_text(theme: str) -> str:
    return f'Theme: {theme.capitalize()}'


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

    def __init__(self, cfg: DictConfig, task_root: str, initial_split: Optional[str] = None,
                 service=None):
        """service: the shared mode's connection (cutieplus.client.ServiceClient); without it the
        browser loads the models itself on the first Open (local mode)."""
        super().__init__()
        self.cfg = cfg
        self.task_root = path.abspath(task_root)
        self.task_set = load_task_set(self.task_root)
        self.service = service
        self.networks = None
        self.controller: Optional[MainController] = None
        self.exports: Dict[Tuple[str, str], QProcess] = {}
        # shared mode: (task id, lease epoch, whether the browser took the lease, export id)
        self.service_exports: Dict[Tuple[str, str], tuple] = {}
        # tasks this browser exports without having them open in its window
        self.export_locks: Dict[Tuple[str, str], TaskLock] = {}
        # only what --labeler said: the name itself follows the annotator list (refresh_identity)
        self.cli_labeler = cfg['labeler'] if 'labeler' in cfg else None
        self.labeler = ''
        self.refresh_identity()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter by sequence, annotator or target text')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        self.hide_done = QCheckBox('Hide done / discarded')
        self.hide_done.setToolTip('Hide the sequences that are done or discarded (in every tab).')
        self.hide_done.toggled.connect(self.apply_filter)
        # top right: cycles Auto, Light, Dark for every window of this program
        self.theme = saved_theme()
        self.theme_button = QPushButton()
        self.theme_button.setToolTip('Color theme: Auto follows the desktop. Click to cycle Auto, '
                                     'Light, Dark. Remembered for this screen.')
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
        self.tables: Dict[str, QTableWidget] = {}
        for split in self.task_set['splits']:
            if not path.isdir(path.join(self.task_root, split)):
                continue
            table = QTableWidget(0, len(COLUMNS))
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
            table.doubleClicked.connect(self.on_open)
            table.itemSelectionChanged.connect(self.show_discard_action)
            table.setItemDelegateForColumn(TARGET_COLUMN, self.target_delegate)
            # sorting moves the rows, but the view keeps its hidden rows by position
            table.model().layoutChanged.connect(self.apply_filter)
            self.tables[split] = table
            self.tabs.addTab(table, split)

        open_button = set_variant(QPushButton('Open'), 'primary')
        open_button.clicked.connect(self.on_open)
        export_button = QPushButton('Export')
        export_button.clicked.connect(self.on_export)
        done_button = set_variant(QPushButton('Mark done (D)'), 'tonal')
        done_button.setToolTip('D: mark the selected sequence done (it is exported first when '
                               'its export is missing or older than its last change).')
        done_button.clicked.connect(self.on_mark_done)
        # the filter box keeps the letter for itself while it has the focus
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(self.on_mark_done)
        # leaves the selected sequence out of the dataset; Restore on a discarded one
        self.discard_button = set_variant(QPushButton('Discard'), 'danger')
        self.discard_button.setToolTip(
            'Leave the selected sequence out of the dataset: its done mark and Need Attention are '
            'removed, and it cannot be exported or marked done. On a discarded sequence the '
            'button restores it.')
        self.discard_button.clicked.connect(self.on_discard)
        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.refresh)
        manage_button = QPushButton('Manage annotators')
        manage_button.clicked.connect(self.on_manage_annotators)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(140)
        # status and annotator counts of the current tab
        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setWordWrap(True)
        self.tabs.currentChanged.connect(self.update_stats)
        self.tabs.currentChanged.connect(self.show_discard_action)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for button in (open_button, export_button, done_button, self.discard_button,
                       refresh_button, manage_button):
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
        tasks = QVBoxLayout()
        tasks.setSpacing(GAP)
        tasks.addWidget(self.tabs, 1)
        tasks.addLayout(buttons)
        layout.addWidget(Card(tasks, stretch=True), 1)
        messages = QVBoxLayout()
        messages.setSpacing(GAP)
        messages.addWidget(self.console)
        messages.addWidget(self.stats)
        layout.addWidget(Card(messages, title='Messages'))
        self.setLayout(layout)

        self.refresh()
        for table in self.tables.values():
            table.setColumnWidth(STATUS_COLUMN, self.status_delegate.column_width(table.font()))
        if initial_split in self.tables:
            self.tabs.setCurrentWidget(self.tables[initial_split])
        self.lock_timer = QTimer(self)
        self.lock_timer.timeout.connect(self.refresh_locks)
        self.lock_timer.start(LOCK_REFRESH_MS)
        fit_window(self, QSize(1100, 750))

    def log(self, text: str) -> None:
        self.console.appendPlainText(text)

    def refresh_identity(self) -> str:
        """Who this session labels as: --labeler, else the annotator assigned to this screen's
        port, else $CUTIE_LABELER, else the account name. Read again whenever it matters, so a
        saved change of the annotator list is effective without a restart."""
        self.labeler = labeler_name(self.cli_labeler, self.task_root)
        title = f'Cutie tasks: {self.task_root} · {self.labeler}'
        if self.windowTitle() != title:
            self.setWindowTitle(title)
        return self.labeler

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

    def show_theme(self) -> None:
        self.theme_button.setText(theme_button_text(self.theme))

    def on_theme(self) -> None:
        self.theme = next_theme(self.theme)
        apply_theme(self.theme)
        save_theme(self.theme)
        self.show_theme()

    def on_manage_annotators(self) -> None:
        dialog = AnnotatorManager(self.task_root, self)
        dialog.saved.connect(self.on_annotators_saved)
        dialog.exec()

    def on_annotators_saved(self, count: int) -> None:
        self.log(f'Annotators saved ({count}). This screen records as {self.refresh_identity()}.')

    def task_dir(self, split: str, sequence: str) -> str:
        return path.join(self.task_root, split, sequence)

    def refresh(self) -> None:
        for split, table in self.tables.items():
            split_dir = path.join(self.task_root, split)
            # hidden names are tasks still being built
            sequences = sorted(name for name in os.listdir(split_dir) if not name.startswith('.')
                               and path.isfile(path.join(split_dir, name, TASK_FILE)))
            table.setSortingEnabled(False)
            table.setRowCount(len(sequences))
            for row, sequence in enumerate(sequences):
                self.fill_row(table, row, summarize_task(self.task_dir(split, sequence)))
            table.setSortingEnabled(True)
            self.tabs.setTabText(self.tabs.indexOf(table), f'{split} ({len(sequences)})')
        self.apply_filter()
        self.update_stats()

    def update_stats(self) -> None:
        # the whole tab, whatever the filter hides; read from the Status column already loaded
        table = self.tabs.currentWidget()
        split = next((name for name, candidate in self.tables.items() if candidate is table), None)
        if split is None:
            self.stats.clear()
            return
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
                           '<br>By annotator: ' + (' · '.join(by_annotator) or 'none yet'))

    @staticmethod
    def open_by_text(holder: Optional[dict]) -> str:
        if holder is None:
            return ''
        if not holder:
            return 'another window'
        text = f'{holder.get("labeler", "?")} ({holder.get("display") or "no display"})'
        return f'{text} {holder["activity"]}' if holder.get('activity') else text

    def fill_row(self, table: QTableWidget, row: int, summary: dict) -> None:
        values = (summary['sequence'], summary['frames'],
                  f'{summary["mask_anchors"]} + {summary["empty_anchors"]} empty', summary['status'],
                  ', '.join(summary['annotators']), self.open_by_text(summary['open_by']),
                  summary['modality'], summary['target'])
        for column, value in enumerate(values):
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DisplayRole, value)  # numbers sort numerically
            if column == STATUS_COLUMN:
                item.setForeground(self.status_color(value))
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
                if value:
                    item.setToolTip(f'changed by {value}')
            if column == OPEN_BY_COLUMN and summary['open_by'] is not None:
                item.setToolTip(describe_holder(summary['open_by']))
            if column == TARGET_COLUMN:
                item.setToolTip(value)
                item.setData(Qt.ItemDataRole.UserRole, (summary['category'], summary['description']))
            table.setItem(row, column, item)

    def refresh_locks(self) -> None:
        # other sessions open and close tasks; a row whose holder changed is reread whole, since
        # its anchors and status may have changed too
        self.refresh_identity()  # the annotator list may have been saved from another screen
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
                    self.fill_row(table, row, summarize_task(self.task_dir(split, sequence)))
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
                self.fill_row(table, row, summarize_task(self.task_dir(split, sequence)))
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
            # a hidden row cannot stay selected: Open / Export / Mark done would act on it unseen
            if any(table.isRowHidden(index.row()) for index in table.selectionModel().selectedRows()):
                table.clearSelection()

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

    def window_exporting(self, split: str, sequence: str) -> bool:
        # the labeling window's own Export task masks; a second export would write the same tree
        return self.is_open(split, sequence) and self.controller.export_process is not None

    def refuse_if_held_elsewhere(self, split: str, sequence: str, action: str) -> bool:
        # another session's window (or export) has the task: its unsaved work is out of reach
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
        if (split, sequence) in self.exports:
            self.log(f'{split}/{sequence} is being exported; open it once the export finishes.')
            return
        # checked before closing the current task, which would be closed for nothing
        if self.refuse_if_held_elsewhere(split, sequence, 'open it'):
            return
        if self.controller is not None:
            if self.controller.propagating:
                self.log('Pause the propagation before switching tasks.')
                return
            # writes pending saves; on_task_event clears the controller
            if not self.controller.gui.close():
                self.log('The open task is exporting; switch tasks once the export finishes.')
                return
        backend_factory = None
        if self.service is not None:
            from cutieplus.backends.shared import SharedTaskBackend
            backend_factory = SharedTaskBackend.factory(self.service)
        elif self.networks is None:
            self.log('Loading models...')
            QApplication.processEvents()
            self.networks = load_networks(self.cfg)
        cfg = copy.deepcopy(self.cfg)
        with open_dict(cfg):
            cfg['workspace'] = self.task_dir(split, sequence)
            cfg['labeler'] = self.refresh_identity()
        self.log(f'Opening {split}/{sequence}.')
        try:
            self.controller = MainController(cfg, networks=self.networks,
                                             on_task_event=self.on_task_event,
                                             backend_factory=backend_factory)
        except TaskLocked as exc:
            # another session took it after the check above
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

    def on_export(self) -> None:
        selection = self.selected_task()
        if selection is not None:
            self.export_task(*selection, then_mark_done=False)

    def on_mark_done(self) -> None:
        # accepts a task under review; exports it first when the export is missing or stale
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
        self.refresh_identity()
        if self.window_exporting(split, sequence):
            self.log(f'{split}/{sequence} is being exported from its labeling window; '
                     'mark it done once that finishes.')
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
            self.log(f'{split}/{sequence} is {status}: save its binary PNGs in the labeling window '
                     'to submit it for review first.')
            return
        if self.service is not None:
            # the review and the export must be of the task's current content
            status = self.service.call('export_status', task_id=f'{split}/{sequence}')
            if not status['review_current']:
                self.log(f'{split}/{sequence} changed after it was submitted; save its binary PNGs '
                         'again to submit it for review.')
            elif status['export_current']:
                self.shared_mark_done(split, sequence)
            else:
                self.export_task(split, sequence, then_mark_done=True)
            return
        if export_is_current(workspace, self.task_set):
            self.finish_mark_done(split, sequence)
        else:
            self.export_task(split, sequence, then_mark_done=True)

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
            'done mark and Need Attention are removed, and it cannot be exported or marked done. '
            'Its masks and anchors stay; Restore brings it back.',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def on_discard(self) -> None:
        # the selected sequence is left out of the dataset, or restored when it is discarded
        from cutieplus.client import ClientError, new_request_id
        selection = self.selected_task()
        if selection is None:
            return
        split, sequence = selection
        workspace = self.task_dir(split, sequence)
        discarded = discard_of(workspace) is None  # what this click does
        action = 'discard it' if discarded else 'restore it'
        if self.refuse_if_held_elsewhere(split, sequence, action):
            return
        if (split, sequence) in self.exports or self.window_exporting(split, sequence):
            self.log(f'{split}/{sequence} is being exported; {action} once that finishes.')
            return
        if self.is_open(split, sequence) and self.controller.propagating:
            self.log('Pause the propagation first.')
            return
        if discarded and not self.confirm_discard(split, sequence):
            return
        self.refresh_identity()
        if self.service is not None:
            lease = self.service_lease(split, sequence)
            if lease is None:
                return
            task_id, epoch, taken = lease
            try:
                self.service.call('set_discard', task_id=task_id, lease_epoch=epoch,
                                  discarded=discarded, request_id=new_request_id())
            except ClientError as exc:
                self.log(f'{split}/{sequence}: not {"discarded" if discarded else "restored"}: '
                         f'{exc}')
                return
            finally:
                self.release_service_lease(task_id, epoch, taken)
        else:
            discard_task(workspace, self.labeler, discarded)
        if discarded:
            if self.is_open(split, sequence):
                self.controller.task.attention = None  # its window's Need Attention is off
                self.controller.gui.refresh_attention_button()
            self.log(f'{split}/{sequence} is discarded: it is left out of the dataset.')
            if curated_split(self.task_root, sequence) == 'done':
                self.log(f'Its row in the curated CSV says done: {split}/{sequence} stays in the '
                         'dataset until that row changes (Cutie never edits the CSV).')
        else:
            self.log(f'{split}/{sequence} is restored; it needs a review and Mark done again.')
        self.refresh_task(split, sequence)

    def service_lease(self, split: str, sequence: str) -> Optional[Tuple[str, int, bool]]:
        """(task id, lease epoch, taken here) of a task the browser exports or marks done in shared
        mode: the open window's lease, else a lease of the browser's own session."""
        from cutieplus.client import ClientError
        if self.is_open(split, sequence):
            backend = self.controller.backend
            return backend.task_id, backend.epoch, False
        try:
            reply = self.service.call('open_task', workspace=self.task_dir(split, sequence))
        except ClientError as exc:
            self.log(f'{split}/{sequence}: {exc}')
            self.refresh_task(split, sequence)
            return None
        return reply['task_id'], reply['lease_epoch'], True

    def release_service_lease(self, task_id: str, epoch: int, taken: bool) -> None:
        if taken:
            try:
                self.service.call('close_task', task_id=task_id, lease_epoch=epoch)
            except Exception as exc:
                self.log(f'Releasing {task_id}: {exc}')

    def shared_mark_done(self, split: str, sequence: str, lease=None) -> None:
        from cutieplus.client import ClientError, new_request_id
        lease = lease or self.service_lease(split, sequence)
        if lease is None:
            return
        task_id, epoch, taken = lease
        try:
            self.service.call('mark_done', task_id=task_id, lease_epoch=epoch,
                              request_id=new_request_id())
            self.log(f'{split}/{sequence} is done.')
        except ClientError as exc:
            self.log(f'{split}/{sequence} is not marked done: {exc}')
        finally:
            if lease is not None and not self.service_exports.get((split, sequence)):
                self.release_service_lease(task_id, epoch, taken)
        self.refresh_task(split, sequence)

    def finish_mark_done(self, split: str, sequence: str) -> None:
        workspace = self.task_dir(split, sequence)
        if attention_of(workspace):  # flagged while it was being exported
            self.log(f'{split}/{sequence} is exported, but it needs attention: it is not marked '
                     'done.')
        elif task_status(workspace) == 'under review':
            mark_done(workspace, self.labeler)
            self.log(f'{split}/{sequence} is done.')
        else:
            self.log(f'{split}/{sequence} changed during the export; it needs a new review.')
        self.refresh_task(split, sequence)

    def export_task(self, split: str, sequence: str, then_mark_done: bool) -> None:
        if discard_of(self.task_dir(split, sequence)):
            self.log(f'{split}/{sequence} is discarded: it is not exported.')
            return
        if (split, sequence) in self.exports or self.window_exporting(split, sequence):
            self.log(f'{split}/{sequence} is already being exported.')
            return
        if self.service is not None:
            if self.is_open(split, sequence) and self.controller.propagating:
                self.log('Pause the propagation before exporting.')
                return
            lease = self.service_lease(split, sequence)
            if lease is None:
                return
            try:
                export_id = self.service.call('begin_export', task_id=lease[0],
                                              lease_epoch=lease[1])['export_id']
            except Exception as exc:
                self.log(f'{split}/{sequence} is not exported: {exc}')
                self.release_service_lease(*lease)
                return
            self.service_exports[(split, sequence)] = lease + (export_id, )
            if self.is_open(split, sequence):
                self.controller.browser_exporting = True
                self.controller.gui.refresh_reset_button_state()
        elif self.is_open(split, sequence):
            if self.controller.propagating:
                self.log('Pause the propagation before exporting.')
                return
            self.controller.res_man.save_queue.join()
            self.controller.browser_exporting = True
            self.controller.gui.refresh_reset_button_state()
        else:
            # hold the task while exporting, so no other session edits or resets it meanwhile
            self.refresh_identity()
            try:
                self.export_locks[(split, sequence)] = TaskLock.acquire(
                    self.task_dir(split, sequence), self.labeler, activity='exporting')
            except TaskLocked as exc:
                self.log(f'{exc} Export it once they close it.')
                self.refresh_task(split, sequence)
                return
        self.log(f'Exporting {split}/{sequence}...')
        self.exports[(split, sequence)] = start_export(
            export_command(self.task_set, split, sequence), self.log,
            lambda ok: self.on_export_finished(split, sequence, ok, then_mark_done))
        self.refresh_task(split, sequence)

    def on_export_finished(self, split: str, sequence: str, ok: bool, then_mark_done: bool) -> None:
        self.exports.pop((split, sequence), None)
        if self.is_open(split, sequence):
            self.controller.browser_exporting = False
            self.controller.gui.refresh_reset_button_state()
        service_export = self.service_exports.pop((split, sequence), None)
        if service_export is not None:
            # the coordinator accepts it only complete and of the content it started from
            task_id, epoch, taken, export_id = service_export
            result = self.service.call('finish_export', export_id=export_id, exit_ok=bool(ok))
            if ok and result['status'] != 'complete':
                self.log(f'The export of {split}/{sequence} is not accepted ({result["status"]}): '
                         f'{result["detail"]}')
                ok = False
            self.log(f'Exported {split}/{sequence}.' if ok else
                     f'Export of {split}/{sequence} failed.')
            if ok and then_mark_done:
                self.shared_mark_done(split, sequence, (task_id, epoch, False))
            self.release_service_lease(task_id, epoch, taken)
            self.refresh_task(split, sequence)
            return
        self.log(f'Exported {split}/{sequence}.' if ok else f'Export of {split}/{sequence} failed.')
        if ok and then_mark_done:
            self.finish_mark_done(split, sequence)  # still holding the task
        export_lock = self.export_locks.pop((split, sequence), None)
        if export_lock is not None:
            export_lock.release()
        self.refresh_task(split, sequence)

    def closeEvent(self, event):
        # quitting destroys the export processes, which kills them halfway through a tree
        if self.exports:
            running = ', '.join(f'{split}/{sequence}' for split, sequence in self.exports)
            self.log(f'Wait for the export of {running} to finish before closing.')
            event.ignore()
            return
        if self.controller is not None and not self.controller.gui.close():
            self.log('The open task is exporting; close once the export finishes.')
            event.ignore()
            return
        event.accept()
