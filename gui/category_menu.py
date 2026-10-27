"""
The category menu of the labeling window, opened by a click on the category pill: the subset's
categories (categories.json, see gui/task_input.py), each row its name on the left and a sample of
its color on the right, under a search box that keeps the categories whose names start with the
letters typed (p: person; ca: cable car, car, cat). Down / Up move through the rows shown, Enter or
a click picks one, Esc or a click outside closes the menu. It is a popup of its own, so the
labeling window's keys do not fire while typing.
"""
from typing import List, Optional

from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QStyledItemDelegate, QVBoxLayout, QWidget)

from gui.task_input import Category

COLOR_ROLE = Qt.ItemDataRole.UserRole + 1  # a row's #RRGGBB
SWATCH = QSize(36, 16)
VISIBLE_ROWS = 10  # the list's height, in rows; more scroll


def matching(categories: List[Category], text: str) -> List[Category]:
    """The categories whose names start with the text, case aside; all for an empty text."""
    prefix = text.strip().lower()
    return [category for category in categories if category.name.lower().startswith(prefix)]


class SwatchDelegate(QStyledItemDelegate):
    """A row: the name as any list draws it, and a rounded sample of the color on the right."""

    MARGIN = 8

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        color = index.data(COLOR_ROLE)
        if not color:
            return
        rect = option.rect
        swatch = QRectF(rect.right() - self.MARGIN - SWATCH.width(),
                        rect.center().y() - SWATCH.height() / 2 + 0.5,
                        SWATCH.width(), SWATCH.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(0, 0, 0, 60))  # an edge, for colors close to the background
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(swatch, 4, 4)
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QSize(hint.width() + SWATCH.width() + 3 * self.MARGIN, max(hint.height(), 28))


class CategoryMenu(QWidget):
    chosen = Signal(str)  # the name picked

    def __init__(self, categories: List[Category], current: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.categories = categories
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search: the first letters')
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter)
        self.search.installEventFilter(self)
        self.list = QListWidget()
        self.list.setItemDelegate(SwatchDelegate(self.list))
        self.list.setUniformItemSizes(True)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the keys stay in the search box
        for category in categories:
            item = QListWidgetItem(category.name)
            item.setData(COLOR_ROLE, category.color)
            item.setToolTip(f'{category.name} ({category.color})')
            self.list.addItem(item)
        self.list.itemClicked.connect(lambda item: self.pick(item))
        self.empty = QLabel()
        self.empty.setWordWrap(True)
        self.empty.setEnabled(False)  # drawn muted

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.search)
        layout.addWidget(self.list)
        layout.addWidget(self.empty)
        self.setMinimumWidth(260)
        if categories:
            row_height = self.list.sizeHintForRow(0)
            self.list.setFixedHeight(row_height * min(len(categories), VISIBLE_ROWS) + 4)
        else:
            self.list.hide()
            self.empty.setText('No categories to choose from: the subset\'s categories.json is '
                               'missing or cannot be read.')
        names = [category.name for category in categories]
        if current in names:
            self.list.setCurrentRow(names.index(current))
        self.filter('')

    def visible_rows(self) -> List[int]:
        return [row for row in range(self.list.count()) if not self.list.isRowHidden(row)]

    def filter(self, text: str) -> None:
        shown = {category.name for category in matching(self.categories, text)}
        for row in range(self.list.count()):
            self.list.setRowHidden(row, self.list.item(row).text() not in shown)
        visible = self.visible_rows()
        if visible and self.list.currentRow() not in visible:
            self.list.setCurrentRow(visible[0])  # Enter picks the first shown
        elif not visible:
            self.list.setCurrentRow(-1)
        if visible:
            self.list.scrollToItem(self.list.currentItem())
        if self.categories:
            self.empty.setText(f'No category starts with "{text.strip()}".')
            self.empty.setVisible(not visible)

    def step(self, delta: int) -> None:
        # Down / Up: the next / previous row shown, stopping at the ends
        visible = self.visible_rows()
        if not visible:
            return
        current = self.list.currentRow()
        index = visible.index(current) + delta if current in visible else 0
        self.list.setCurrentRow(visible[max(0, min(index, len(visible) - 1))])

    def pick(self, item: Optional[QListWidgetItem] = None) -> None:
        item = item or self.list.currentItem()
        if item is None or self.list.isRowHidden(self.list.row(item)):
            return
        self.close()
        self.chosen.emit(item.text())

    def eventFilter(self, watched, event) -> bool:
        if watched is self.search and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Down:
                self.step(1)
                return True
            if key == Qt.Key.Key_Up:
                self.step(-1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.pick()
                return True
            if key == Qt.Key.Key_Escape:
                self.close()
                return True
        return super().eventFilter(watched, event)

    def popup_below(self, widget: QWidget) -> None:
        """Shows the menu right under the widget, the cursor in its search box."""
        self.adjustSize()
        self.move(widget.mapToGlobal(widget.rect().bottomLeft()) + QPoint(0, 4))
        self.show()
        self.search.setFocus()


class CategoryPill(QLabel):
    """The labeling window's category: a pill that opens the category menu when clicked."""
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)
