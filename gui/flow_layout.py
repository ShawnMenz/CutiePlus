"""
A layout that places its items left to right and wraps them onto new rows when the width runs
out (after Qt's Flow Layout example). The items of a row are spread across the width, or packed
to the left with spread=False, or widened to fill the row with fill=True (cards).
"""
from typing import List

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget, QWidgetItem


class FlowLayout(QLayout):

    def __init__(self, spacing: int = 12, spread: bool = True, fill: bool = False) -> None:
        super().__init__()
        self.items: List[QLayoutItem] = []
        self.spread = spread
        self.fill = fill
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:
        self.items.append(item)

    def insert_widget(self, index: int, widget: QWidget) -> None:
        # addWidget, at a position
        self.addChildWidget(widget)
        self.items.insert(index, QWidgetItem(widget))
        self.invalidate()

    def count(self) -> int:
        return len(self.items)

    def itemAt(self, index: int):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index: int):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self.arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        # every item in one row
        hints = [item.sizeHint() for item in self.shown_items()]
        width = sum(hint.width() for hint in hints) + self.spacing() * max(len(hints) - 1, 0)
        height = max((hint.height() for hint in hints), default=0)
        return QSize(width, height) + self.margin_size()

    def minimumSize(self) -> QSize:
        # the widest item alone in a row
        size = QSize(0, 0)
        for item in self.shown_items():
            size = size.expandedTo(item.minimumSize())
        return size + self.margin_size()

    def shown_items(self) -> List[QLayoutItem]:
        return [item for item in self.items if not item.isEmpty()]

    def margin_size(self) -> QSize:
        margins = self.contentsMargins()
        return QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def arrange(self, rect: QRect, apply: bool) -> int:
        """Places the items in rect when apply is set; returns the height they take."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        spacing = self.spacing()

        # rows by minimum width
        rows: List[List[QLayoutItem]] = []
        row_width = 0
        for item in self.shown_items():
            width = item.minimumSize().width()
            if rows and row_width + spacing + width <= area.width():
                rows[-1].append(item)
                row_width += spacing + width
            else:
                rows.append([item])
                row_width = width

        y = area.y()
        for index, row in enumerate(rows):
            height = max(item.sizeHint().height() for item in row)
            if apply:
                # the items grow towards their preferred width, then what is left goes between
                # them; a lone item stays on the left
                minimums = [item.minimumSize().width() for item in row]
                wanted = [max(item.sizeHint().width() - minimum, 0)
                          for item, minimum in zip(row, minimums)]
                spare = max(area.width() - sum(minimums) - spacing * (len(row) - 1), 0)
                grow = min(spare / sum(wanted), 1) if sum(wanted) else 0
                widths = [round(minimum + extra * grow) for minimum, extra in zip(minimums, wanted)]
                if self.fill:  # the width left over goes to the items, evenly
                    left = area.width() - sum(widths) - spacing * (len(row) - 1)
                    widths = [width + left // len(row) for width in widths]
                    widths[-1] += left % len(row)
                if self.spread and len(row) > 1 and not self.fill:
                    gap = max((area.width() - sum(widths)) / (len(row) - 1), spacing)
                else:
                    gap = spacing
                x = area.x()
                for item, width in zip(row, widths):
                    item.setGeometry(QRect(round(x), y, width, height))
                    x += width + gap
            y += height + (spacing if index < len(rows) - 1 else 0)
        return y - area.y() + margins.top() + margins.bottom()
