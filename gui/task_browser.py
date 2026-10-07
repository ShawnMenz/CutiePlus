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
from PySide6.QtCore import QProcess, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QTabWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from cutieplus.backends.base import BackendError
from gui.annotator_manager import AnnotatorManager
from gui.gui_utils import fit_window
from gui.main_controller import MainController, load_networks
from gui.task_export import start_export
from gui.task_state import (TASK_FILE, TaskLock, TaskLocked, describe_holder, export_command,
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
}


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
        selected = bool(shown.state & QStyle.StateFlag.State_Selected)
        color = shown.palette.color(QPalette.ColorRole.HighlightedText if selected
                                    else QPalette.ColorRole.Text)
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
        # tasks this browser exports without having them open in its window
        self.export_locks: Dict[Tuple[str, str], TaskLock] = {}
        # only what --labeler said: the name itself follows the annotator list (refresh_identity)
        self.cli_labeler = cfg['labeler'] if 'labeler' in cfg else None
        self.labeler = ''
        self.refresh_identity()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filter by sequence, annotator or target text')
        self.search.textChanged.connect(self.apply_filter)
        self.hide_done = QCheckBox('Hide done')
        self.hide_done.setToolTip('Hide the sequences that are done (in every tab).')
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

        self.tabs = QTabWidget()
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
            table.horizontalHeader().setSectionResizeMode(TARGET_COLUMN, QHeaderView.ResizeMode.Stretch)
            table.doubleClicked.connect(self.on_open)
            table.setItemDelegateForColumn(TARGET_COLUMN, self.target_delegate)
            # sorting moves the rows, but the view keeps its hidden rows by position
            table.model().layoutChanged.connect(self.apply_filter)
            self.tables[split] = table
            self.tabs.addTab(table, split)

        open_button = QPushButton('Open')
        open_button.clicked.connect(self.on_open)
        export_button = QPushButton('Export')
        export_button.clicked.connect(self.on_export)
        done_button = QPushButton('Mark done')
        done_button.clicked.connect(self.on_mark_done)
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

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for button in (open_button, export_button, done_button, refresh_button, manage_button):
            buttons.addWidget(button)
        layout = QVBoxLayout()
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.search, 1)
        filter_row.addWidget(self.hide_done)
        filter_row.addWidget(self.theme_button)
        layout.addLayout(filter_row)
        layout.addWidget(self.tabs)
        layout.addLayout(buttons)
        layout.addWidget(self.console)
        layout.addWidget(self.stats)
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

    def refresh_identity(self) -> str:
        """Who this session labels as: --labeler, else the annotator assigned to this screen's
        port, else $CUTIE_LABELER, else the account name. Read again whenever it matters, so a
        saved change of the annotator list is effective without a restart."""
        self.labeler = labeler_name(self.cli_labeler, self.task_root)
        title = f'Cutie tasks: {self.task_root} · {self.labeler}'
        if self.windowTitle() != title:
            self.setWindowTitle(title)
        return self.labeler

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
        parts = [f'<span style="color:{color.name()}">{status} {counts[status]}</span>'
                 for status, color in reversed(STATUS_COLORS.items())]
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
                item.setForeground(STATUS_COLORS[value])
                if summary['last_edited_by']:
                    item.setToolTip(f'last edited by {summary["last_edited_by"]}')
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
                          or (hide_done and table.item(row, STATUS_COLUMN).text() == 'done'))
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
        if export_is_current(workspace, self.task_set):
            self.finish_mark_done(split, sequence)
        else:
            self.export_task(split, sequence, then_mark_done=True)

    def finish_mark_done(self, split: str, sequence: str) -> None:
        workspace = self.task_dir(split, sequence)
        if task_status(workspace) == 'under review':
            mark_done(workspace, self.labeler)
            self.log(f'{split}/{sequence} is done.')
        else:
            self.log(f'{split}/{sequence} changed during the export; it needs a new review.')
        self.refresh_task(split, sequence)

    def export_task(self, split: str, sequence: str, then_mark_done: bool) -> None:
        if (split, sequence) in self.exports or self.window_exporting(split, sequence):
            self.log(f'{split}/{sequence} is already being exported.')
            return
        if self.is_open(split, sequence):
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
