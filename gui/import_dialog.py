"""
The Import window (the task browser's Import button): one row per subset of the input, each a tab
of the browser. The subsets already imported are listed and checked again; + Add a subset adds a
row, a tab name and the folder holding the subset's visual/, thermal/ and categories.json. When
the parts of a subset sit directly in the input folder, the first row is main, <input>/main:
Save moves them there. Every row is checked as it is typed (gui/task_import.py); Save is enabled
once all pass, and moves the new subsets into the input folder in the background. A
categories.json may list names only: Save then picks a distinct color for each and writes them
into it (gui/task_import.py: complete_colors), in imported subsets too.
"""
import html
import threading
from os import path
from typing import List, Optional

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from gui.gui_utils import is_dark
from gui.style import ATTENTION, DANGER, GAP, PRIMARY, Card, set_variant
from gui.task_import import ImportRow, RowPlan, apply_import, plan_import
from gui.task_input import (CATEGORIES_FILE, MAIN_TAB, TARGETS_FILE, InputError, find_tabs,
                            unorganized_parts)

CHECK_DELAY_MS = 300  # after the last keystroke
PREFERRED_SIZE = QSize(820, 520)
NAME_WIDTH = 130


class SubsetRow(QWidget):
    """A tab name, its folder (Browse…), ✕ for a new row, and the folder's check below."""

    def __init__(self, dialog: 'ImportDialog', tab: str = '', folder: str = '',
                 imported: bool = False) -> None:
        super().__init__()
        self.dialog = dialog
        self.imported = imported
        self.name_auto = not tab  # the name follows the folder until it is typed
        self.name_edit = QLineEdit(tab)
        self.name_edit.setPlaceholderText('tab name')
        self.name_edit.setFixedWidth(NAME_WIDTH)
        self.folder_edit = QLineEdit(folder)
        self.folder_edit.setPlaceholderText('folder holding visual/, thermal/ and '
                                            f'{CATEGORIES_FILE}')
        self.browse_button = QPushButton('Browse…')
        self.browse_button.clicked.connect(self.on_browse)
        self.remove_button = QPushButton('✕')
        self.remove_button.setToolTip('Remove this row')
        self.remove_button.setFixedWidth(36)
        self.remove_button.clicked.connect(lambda: dialog.remove_row(self))
        if imported:  # an imported subset stays as it is
            for edit in (self.name_edit, self.folder_edit):
                edit.setReadOnly(True)
            self.browse_button.hide()
            self.remove_button.hide()
        self.name_edit.textEdited.connect(self.on_name_edited)
        self.folder_edit.textChanged.connect(self.on_folder_changed)
        self.check_label = QLabel()
        self.check_label.setTextFormat(Qt.TextFormat.RichText)
        self.check_label.setWordWrap(True)
        self.check_label.setContentsMargins(NAME_WIDTH + GAP, 0, 0, 0)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(GAP)
        line.addWidget(self.name_edit)
        line.addWidget(self.folder_edit, 1)
        line.addWidget(self.browse_button)
        line.addWidget(self.remove_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)
        layout.addLayout(line)
        layout.addWidget(self.check_label)

    def row(self) -> ImportRow:
        return ImportRow(self.name_edit.text(), self.folder_edit.text(), self.imported)

    def on_name_edited(self, _text: str) -> None:
        self.name_auto = False
        self.dialog.schedule_check()

    def on_folder_changed(self, text: str) -> None:
        if self.name_auto and text.strip():
            self.name_edit.setText(path.basename(path.normpath(text.strip())))
        self.dialog.schedule_check()

    def on_browse(self) -> None:
        start = self.folder_edit.text().strip() or self.dialog.root
        folder = QFileDialog.getExistingDirectory(self, 'The folder of a subset',
                                                  start if path.isdir(start) else '')
        if folder:
            self.folder_edit.setText(folder)

    def show_plan(self, plan: RowPlan) -> None:
        dark = is_dark(self)
        lines = []
        if plan.errors or (plan.check is not None and plan.check.errors):
            problems = plan.errors + (plan.check.errors if plan.check is not None else [])
            lines.append(f'<span style="color:{DANGER[int(dark)]}">✗ '
                         f'{html.escape(" ".join(problems))}</span>')
        elif plan.check is not None:
            check = plan.check
            parts = [f'{check.sequences} sequences']
            if check.has_targets:
                parts.append(f'targets {check.with_targets}')
            else:
                parts.append(f'no {TARGETS_FILE}: descriptions empty, category null')
            parts += [f'masks {check.with_masks}', f'{check.categories} categories']
            if check.uncolored:
                parts.append(f'colors for {check.uncolored} picked on Save')
            if plan.note:
                parts.append(plan.note)
            lines.append(f'<span style="color:{PRIMARY}">✓</span> '
                         f'{html.escape(" · ".join(parts))}')
        if plan.check is not None:
            lines += [f'<span style="color:{ATTENTION}">⚠ {html.escape(warning)}</span>'
                      for warning in plan.check.warnings]
        self.check_label.setText('<br>'.join(lines))


class ImportWorker(QObject):
    """Makes the moves in a thread of its own; the signals reach the window in its thread."""
    progress = Signal(str)
    finished = Signal(str)  # the error, empty when every move was made

    def run(self, moves, color_folders) -> None:
        try:
            apply_import(moves, self.progress.emit, color_folders)
        except Exception as exc:  # shown in the window; the moves before it stay made
            self.finished.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit('')


class ImportDialog(QDialog):
    imported = Signal(str)  # what was imported, for the browser's messages

    def __init__(self, root: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.root = path.abspath(root)
        self.setWindowTitle('Import')
        self.rows: List[SubsetRow] = []
        self.plans: List[RowPlan] = []
        self.moving = False
        self.worker: Optional[ImportWorker] = None
        self.check_timer = QTimer(self)
        self.check_timer.setSingleShot(True)
        self.check_timer.setInterval(CHECK_DELAY_MS)
        self.check_timer.timeout.connect(self.recheck)

        intro = QLabel(f'Each subset is a tab. A subset folder needs <b>visual/</b>, '
                       f'<b>thermal/</b> and <b>{CATEGORIES_FILE}</b>; {TARGETS_FILE}, '
                       'annotation/ (pregenerated masks) and seqinfo/ are optional. A folder '
                       f'outside {html.escape(self.root)} is moved into it on Save.')
        intro.setWordWrap(True)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(GAP)
        rows_widget = QWidget()
        rows_box = QVBoxLayout(rows_widget)
        rows_box.setContentsMargins(0, 0, 0, 0)
        rows_box.addLayout(self.rows_layout)
        rows_box.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(rows_widget)
        self.add_button = set_variant(QPushButton('+ Add a subset'), 'tonal')
        self.add_button.clicked.connect(lambda: self.add_row())
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = set_variant(QPushButton('Save'), 'primary')
        self.save_button.clicked.connect(self.save)

        header = QHBoxLayout()
        header.setSpacing(GAP)
        for text, width in (('Tab', NAME_WIDTH), ('Folder', None)):
            label = QLabel(f'<b>{text}</b>')
            if width:
                label.setFixedWidth(width)
            header.addWidget(label, 0 if width else 1)
        body = QVBoxLayout()
        body.setSpacing(GAP)
        body.addWidget(intro)
        body.addLayout(header)
        body.addWidget(scroll, 1)
        add_line = QHBoxLayout()
        add_line.addWidget(self.add_button)
        add_line.addStretch(1)
        body.addLayout(add_line)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status, 1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(Card(body, stretch=True), 1)
        layout.addLayout(buttons)

        try:
            tabs = find_tabs(self.root)
        except InputError:
            tabs = []
        for tab in tabs:
            self.add_row(tab.name, tab.folder, imported=True)
        if unorganized_parts(self.root) or not tabs:
            # the default: the input found in the root, or the first subset, goes to main/
            self.add_row(MAIN_TAB, path.join(self.root, MAIN_TAB))
        self.recheck()
        available = self.screen().availableGeometry().size() if self.screen() else PREFERRED_SIZE
        self.resize(PREFERRED_SIZE.boundedTo(available * 0.9))

    def add_row(self, tab: str = '', folder: str = '', imported: bool = False) -> SubsetRow:
        row = SubsetRow(self, tab, folder, imported)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        self.schedule_check()
        if not imported and not tab:
            row.folder_edit.setFocus()
        return row

    def remove_row(self, row: SubsetRow) -> None:
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.schedule_check()

    def schedule_check(self) -> None:
        self.check_timer.start()

    def recheck(self) -> None:
        """Checks every row again; Save is enabled when all pass and one brings something new."""
        self.check_timer.stop()
        self.plans = plan_import(self.root, [row.row() for row in self.rows])
        for row, plan in zip(self.rows, self.plans):
            row.show_plan(plan)
        new = [plan for plan in self.plans if not plan.row.imported]
        colors = [plan for plan in self.plans if plan.colors_folder]
        failing = [plan for plan in self.plans if not plan.ok]
        self.save_button.setEnabled(not self.moving and bool(new or colors) and not failing)
        if failing:
            self.save_button.setToolTip('Fix the rows marked ✗ first.')
        elif not new and not colors:
            self.save_button.setToolTip('Nothing to import: + Add a subset first.')
        else:
            self.save_button.setToolTip('Import the new subsets, and pick the missing category '
                                        'colors.' if colors else 'Import the new subsets.')

    def save(self) -> None:
        self.recheck()  # the folders may have changed since the last check
        if not self.save_button.isEnabled():
            return
        moves = [plan.move for plan in self.plans if plan.move is not None]
        color_folders = [plan.colors_folder for plan in self.plans if plan.colors_folder]
        self.added = [plan.row.tab.strip() for plan in self.plans if not plan.row.imported]
        self.colored = [plan.row.tab.strip() for plan in self.plans if plan.colors_folder]
        self.moving = True
        for widget in [self.add_button, self.save_button, self.cancel_button] + \
                [child for row in self.rows for child in (row.name_edit, row.folder_edit,
                                                          row.browse_button, row.remove_button)]:
            widget.setEnabled(False)
        self.worker = ImportWorker()
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.on_finished)
        threading.Thread(target=self.worker.run, args=(moves, color_folders),
                         daemon=True).start()

    def on_finished(self, error: str) -> None:
        self.moving = False
        if error:
            self.status.setText(f'Import stopped: {error}')
            for widget in [self.add_button, self.cancel_button] + \
                    [child for row in self.rows if not row.imported
                     for child in (row.name_edit, row.folder_edit, row.browse_button,
                                   row.remove_button)]:
                widget.setEnabled(True)
            self.recheck()
            return
        said = [f'Imported {", ".join(self.added)} into {self.root}.'] if self.added else []
        if self.colored:
            said.append(f'Picked the missing category colors of {", ".join(self.colored)}.')
        self.imported.emit(' '.join(said))
        self.accept()

    def reject(self) -> None:
        if not self.moving:  # a move cannot be stopped half way
            super().reject()

    def closeEvent(self, event) -> None:
        if self.moving:
            event.ignore()
            return
        super().closeEvent(event)
