"""
The Manage annotators page of the task browser: one row per annotator, 'Port: [5902] --> [name]'.
Only the admin's session may change the list (gui/annotators.py, admin_port); every other session
opens the same page read-only, as the Annotator list. The admin's row and the row of the screen the
page is on cannot be removed.
Edits are staged in the page; Save checks them, writes annotators.json (see gui/annotators.py) and
makes them effective for the sessions that read it: the screen of every saved port runs, the screen
of every removed port stops (gui/vnc_screens.py), and removed names become inactive.
"""
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from gui import vnc_screens
from gui.annotators import (ANNOTATORS_FILE, MAX_NAME_LENGTH, MAX_PORT, MIN_PORT, VNC_BASE_PORT,
                            annotators_path, check_annotators, load_admin_port, load_annotators,
                            load_inactive, may_manage, next_inactive, save_annotators,
                            session_port)
from gui.style import Card, set_pill, set_variant

FIRST_ANNOTATOR_PORT = VNC_BASE_PORT + 2  # screen :1 is the real monitor


class AnnotatorRow(QWidget):
    """Port: [number] --> [name] [Remove], or [Admin] in Remove's place on the admin's row, which
    has no Remove. editable False: shown only. locked: why its port cannot change and it cannot be
    removed (the admin's row, this screen's row), or None."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, port: int, name: str, editable: bool = True, admin: bool = False,
                 locked: Optional[str] = None) -> None:
        super().__init__()
        self.port_box = QSpinBox()
        self.port_box.setRange(MIN_PORT, MAX_PORT)
        self.port_box.setValue(port)
        self.port_box.setToolTip(f'The port this annotator connects to: {VNC_BASE_PORT} plus the '
                                 'virtual screen number.')
        self.name_box = QLineEdit(name)
        self.name_box.setMaxLength(MAX_NAME_LENGTH)
        self.name_box.setPlaceholderText('Annotator name')
        self.remove_button = set_variant(QPushButton('Remove'), 'danger')
        self.remove_button.setAutoDefault(False)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(QLabel('Port:'))
        layout.addWidget(self.port_box)
        layout.addWidget(QLabel('-->'))
        layout.addWidget(self.name_box, 1)
        self.admin_label = None
        if admin:
            # as wide as Remove on the other rows, so the column lines up
            self.admin_label = set_pill(QLabel('Admin'), 'accent')
            self.admin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.admin_label.setFixedWidth(self.remove_button.sizeHint().width())
            layout.addWidget(self.admin_label)
            self.remove_button.hide()
        else:
            layout.addWidget(self.remove_button)
        self.setLayout(layout)
        if not editable:
            for box in (self.port_box, self.name_box):
                box.setReadOnly(True)
            self.port_box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            self.remove_button.hide()
        elif locked:
            self.port_box.setEnabled(False)
            self.port_box.setToolTip(locked)
            self.remove_button.setEnabled(False)
            self.remove_button.setToolTip(locked)
        self.port_box.valueChanged.connect(lambda _: self.changed.emit())
        self.name_box.textChanged.connect(lambda _: self.changed.emit())
        self.remove_button.clicked.connect(lambda: self.removed.emit(self))

    def entry(self) -> dict:
        return {'port': self.port_box.value(), 'name': self.name_box.text().strip()}

    def matches(self, text: str) -> bool:
        return text in f'{self.port_box.value()} {self.name_box.text().lower()}'


class AnnotatorManager(QDialog):
    """Create, update, remove and look up annotators; nothing is written until Save."""

    saved = Signal(int)  # the number of annotators now in the file

    def __init__(self, task_root: str, parent: Optional[QWidget] = None,
                 screens=vnc_screens) -> None:
        """screens: what starts and stops the VNC screens (gui/vnc_screens.py; tests pass a fake)."""
        super().__init__(parent)
        self.task_root = task_root
        self.screens = screens
        self.admin_port = load_admin_port(task_root)
        self.own_port = session_port()
        self.editable = may_manage(task_root)
        self.setWindowTitle('Manage annotators' if self.editable else 'Annotator list')
        self.rows: List[AnnotatorRow] = []
        # what the file held when this page opened or last saved: a change underneath is noticed
        self.loaded = load_annotators(task_root)
        self.saved_state: List[dict] = []  # set once the rows exist

        if self.editable:
            header = QLabel('Each annotator connects to their own port: port 59NN is virtual screen '
                            ':NN. Changes take effect when you press Save: it starts the screen of '
                            'every new port and stops the screen of every removed one.')
        else:
            admin = next((entry['name'] for entry in self.loaded
                          if entry['port'] == self.admin_port), '?')
            header = QLabel('Each annotator connects to their own port: port 59NN is virtual screen '
                            f':NN. Only the admin (port {self.admin_port}, {admin}) can change '
                            'this list.')
        header.setWordWrap(True)
        self.lookup = QLineEdit()
        self.lookup.setPlaceholderText('Look up by port or name')
        self.lookup.textChanged.connect(self.apply_lookup)

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.addStretch(1)
        rows_widget = QWidget()
        rows_widget.setLayout(self.rows_layout)
        scroll = QScrollArea()
        scroll.setWidget(rows_widget)
        scroll.setWidgetResizable(True)

        self.add_button = set_variant(QPushButton('Add annotator'), 'tonal')
        self.add_button.clicked.connect(lambda: self.add_row(self.next_free_port(), ''))
        self.save_button = set_variant(QPushButton('Save'), 'primary')
        self.save_button.clicked.connect(self.save)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        for button in (self.add_button, self.save_button, self.close_button):
            button.setAutoDefault(False)  # Enter in a name box must not press a button
        self.add_button.setVisible(self.editable)
        self.save_button.setVisible(self.editable)
        self.status = QLabel()
        self.status.setWordWrap(True)
        hint = QLabel(f'A screen with no entry records the account name. Stored in {ANNOTATORS_FILE} '
                      'in the task root.')
        hint.setWordWrap(True)

        lookup_row = QHBoxLayout()
        lookup_row.addWidget(QLabel('Look up:'))
        lookup_row.addWidget(self.lookup, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.close_button)
        # the list in a rounded card (gui/style.py)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        listing = QVBoxLayout()
        listing.addLayout(lookup_row)
        listing.addWidget(scroll, 1)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(Card(listing, stretch=True), 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addWidget(hint)
        self.setLayout(layout)

        for entry in self.loaded:
            self.add_row(entry['port'], entry['name'])
        self.saved_state = self.current_entries()
        self.update_status()
        self.resize(620, 460)

    # ---- the staged list ----------------------------------------------------------------------
    def add_row(self, port: int, name: str) -> AnnotatorRow:
        admin = port == self.admin_port and any(entry['port'] == port for entry in self.loaded)
        locked = None
        if admin:
            locked = 'The admin cannot be removed.'
        elif port == self.own_port and any(entry['port'] == port for entry in self.loaded):
            locked = 'This is the screen you are on: it cannot remove itself.'
        row = AnnotatorRow(port, name, self.editable, admin, locked)
        row.changed.connect(self.on_edited)
        row.removed.connect(self.remove_row)
        self.rows.append(row)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        self.apply_lookup()
        self.update_status()
        return row

    def remove_row(self, row: AnnotatorRow) -> None:
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        self.update_status()

    def next_free_port(self) -> int:
        used = {row.port_box.value() for row in self.rows}
        return next((port for port in range(FIRST_ANNOTATOR_PORT, MAX_PORT + 1) if port not in used),
                    MAX_PORT)

    def current_entries(self) -> List[dict]:
        return sorted((row.entry() for row in self.rows),
                      key=lambda entry: (entry['port'], entry['name']))

    def is_dirty(self) -> bool:
        return self.current_entries() != self.saved_state

    def on_edited(self) -> None:
        self.apply_lookup()
        self.update_status()

    def apply_lookup(self) -> None:
        text = self.lookup.text().strip().lower()
        for row in self.rows:
            row.setVisible(not text or row.matches(text))

    def update_status(self) -> None:
        if not self.editable:
            self.set_status('', 'gray')
            return
        problems = check_annotators(self.current_entries(), self.admin_port)
        if problems:
            self.set_status('\n'.join(problems), 'red')
        elif self.is_dirty():
            self.set_status('Unsaved changes: press Save to make them effective.', 'orange')
        else:
            self.set_status('', 'gray')

    def set_status(self, text: str, color: str) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f'QLabel {{ color: {color}; }}')

    # ---- saving and closing -------------------------------------------------------------------
    def save(self) -> bool:
        if not self.editable or not may_manage(self.task_root):
            self.set_status('Not saved: only the admin can change the annotator list.', 'red')
            return False
        self.admin_port = load_admin_port(self.task_root)
        entries = self.current_entries()
        problems = check_annotators(entries, self.admin_port)
        if problems:
            self.set_status('Not saved:\n' + '\n'.join(problems), 'red')
            return False
        if load_annotators(self.task_root) != self.loaded:
            answer = QMessageBox.question(
                self, 'Annotators changed',
                'Someone saved a different list since this page was opened. Overwrite it with '
                'yours?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return False
        current = load_annotators(self.task_root)  # what the screens follow now
        ports = {entry['port'] for entry in entries}
        removed = {entry['port']: entry['name'] for entry in current if entry['port'] not in ports}
        if self.own_port in removed:
            self.set_status(f'Not saved: port {self.own_port} is this screen, which cannot remove '
                            'itself.', 'red')
            return False
        if removed and QMessageBox.question(
                self, 'Remove annotators',
                'Their screens stop now, and anyone connected is disconnected.\n\n'
                + '\n'.join(f'Port {port}: {name}' for port, name in sorted(removed.items())),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return False
        try:
            saved = save_annotators(self.task_root, entries, next_inactive(
                load_inactive(self.task_root), current + self.loaded, entries))
        except (OSError, ValueError) as exc:
            self.set_status(f'Not saved: {exc}', 'red')
            return False
        self.loaded = saved
        self.saved_state = self.current_entries()
        count = len(saved)
        messages = [f'Saved {count} annotator{"" if count == 1 else "s"}.']
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.screens.sync([entry['port'] for entry in saved], sorted(removed))
        finally:
            QApplication.restoreOverrideCursor()
        # the screens that changed or failed; the ones already running go unsaid
        messages += [vnc_screens.describe(port, result) for port, result in sorted(results.items())
                     if result[0] not in ('running', 'absent')]
        failed = any(problem for _, problem in results.values())
        self.set_status(' '.join(messages), 'orange' if failed else 'green')
        self.saved.emit(count)
        return True

    def confirm_close(self) -> bool:
        if not self.is_dirty():
            return True
        answer = QMessageBox.question(
            self, 'Unsaved changes', 'Close without saving your changes?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def reject(self) -> None:
        # Close and Esc
        if self.confirm_close():
            super().reject()

    def closeEvent(self, event) -> None:
        # the window's own close button
        if self.confirm_close():
            event.accept()
        else:
            event.ignore()
