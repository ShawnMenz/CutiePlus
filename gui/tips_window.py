"""
The labeling guide (TIPS.md) in a window of its own, opened by the labeling window's Tips button.
It is not modal: the guide can stay open beside the labeling window.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

TIPS_FILE = Path(__file__).parent / 'TIPS.md'
PREFERRED_SIZE = (780, 900)


class TipsWindow(QWidget):

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('Cutie labeling guide')
        self.browser = QTextBrowser()
        self.browser.setMarkdown(TIPS_FILE.read_text(encoding='utf-8'))
        layout = QVBoxLayout()
        layout.addWidget(self.browser)
        self.setLayout(layout)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(self.close)

        self.resize(*PREFERRED_SIZE)
        self.placed = False
        # a VNC viewer resizes the remote screen whenever its own window changes
        parent.screen().availableGeometryChanged.connect(self.on_screen_changed)

    def fit_on_screen(self) -> None:
        # as large as the guide likes, but inside the screen, and entirely on it
        available = self.parentWidget().screen().availableGeometry()
        self.resize(min(PREFERRED_SIZE[0], available.width() - 40),
                    min(PREFERRED_SIZE[1], available.height() - 80))
        frame = self.frameGeometry()
        self.move(min(max(frame.x(), available.left()), available.right() + 1 - frame.width()),
                  min(max(frame.y(), available.top()), available.bottom() + 1 - frame.height()))

    def on_screen_changed(self, available) -> None:
        if self.isVisible():
            self.fit_on_screen()

    def show_raised(self) -> None:
        # the first time, centered on the labeling window's screen; afterwards where it was left
        self.fit_on_screen()
        if not self.placed and not self.isVisible():
            available = self.parentWidget().screen().availableGeometry()
            frame = self.frameGeometry()
            frame.moveCenter(available.center())
            self.move(frame.topLeft())
            self.placed = True
        self.show()
        self.fit_on_screen()  # again now that the window manager's frame is known
        self.raise_()
        self.activateWindow()
