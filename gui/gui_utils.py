import html
from typing import List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPalette, QPen, QRegion, QValidator
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLayout, QLineEdit, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSlider, QSpinBox, QProgressBar, QStyle,
                               QStyleOptionSlider, QWidget)

from gui.style import PRIMARY, SECONDARY, category_pill, restyle

# room for the title bar and borders that the window manager adds around a window
WINDOW_FRAME = QSize(16, 48)

# the frame label's colors, (light theme, dark theme); pure blue is unreadable on the dark theme
MODALITY_COLORS = {
    'R': ('#d32f2f', '#ff6b6b'),
    'G': ('#2e7d32', '#5ad66f'),
    'B': ('#1565c0', '#5c9dff'),
    'thermal': ('#ff4500', '#ff4500'),
}
# by the kind of mask source, the text before the colon
STATUS_COLORS = {
    'anchor': ('#7b1fa2', '#c38bff'),
    'empty anchor': ('#c2185b', '#ff7eb3'),
    'propagated': ('#00796b', '#4dd0c4'),
    'pregenerated': ('#8d7b00', '#e3cf4a'),
    'no mask': ('#757575', '#b0b0b0'),
}
STATUS_MEANING = {
    'anchor': 'a human decision, never overwritten',
    'empty anchor': 'target absent: propagation stops here',
    'propagated': 'written by Cutie',
    'pregenerated': 'model mask, not reviewed yet',
    'no mask': 'nothing yet',
}


def is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


def status_kind(source: str) -> str:
    return source.split(':')[0].strip()


def frame_label_html(file_name: str, modality: Optional[str], source: Optional[str],
                     dark: bool) -> str:
    """'000001.jpg (RGB) [anchor]' with the modality and the status colored: visual is shown
    as RGB in red, green and blue, thermal in orange-red. A plain workspace has no modality or
    source: just the file name."""
    text = html.escape(file_name)
    if modality is None:
        return text
    theme = int(dark)
    if modality == 'visual':
        shown = ''.join(f'<b style="color:{MODALITY_COLORS[letter][theme]}">{letter}</b>'
                        for letter in 'RGB')
    elif modality == 'thermal':
        shown = f'<b style="color:{MODALITY_COLORS["thermal"][theme]}">thermal</b>'
    else:
        shown = html.escape(modality)
    colors = STATUS_COLORS.get(status_kind(source or ''))
    status = html.escape(source or '')
    if colors is not None:
        status = f'<span style="color:{colors[theme]}">{status}</span>'
    return f'{text} ({shown}) [{status}]'


def frame_label_legend(dark: bool) -> str:
    theme = int(dark)
    rows = [f'<b style="color:{MODALITY_COLORS["B"][theme]}">RGB</b>: the visible-light frame; '
            f'<b style="color:{MODALITY_COLORS["thermal"][theme]}">thermal</b>: the thermal frame',
            'Between the brackets, where the shown mask comes from:']
    rows += [f'<span style="color:{STATUS_COLORS[kind][theme]}">{kind}</span>: {meaning}'
             for kind, meaning in STATUS_MEANING.items()]
    return '<br>'.join(rows)


class LogView(QPlainTextEdit):
    """The log area. A text box keeps its top line when it is resized, so a log that shrinks with
    its window would push the newest messages out of view: this one keeps the end in view, unless
    the end was already scrolled out of it (someone reading older messages)."""

    def resizeEvent(self, event):
        bar = self.verticalScrollBar()
        following = bar.value() >= bar.maximum() - 2
        super().resizeEvent(event)
        if following:
            bar.setValue(bar.maximum())


class FrameSlider(QSlider):
    """The frame slider. Frames are stepped with A / D, so the arrow keys are left alone (a slider
    would otherwise step on them); Home / End and PageUp / PageDown still move it. In task mode it
    shows the task's intervals as bands along the groove, the selected one (the set the window
    shows) in bold green, and its anchors as small hollow rings on the groove: black (white in a
    dark theme) for a mask anchor, magenta for an empty one."""

    # rounded segments on the groove: the selected interval in the primary green and a little
    # taller, the others in the secondary green (gui/style.py)
    BAND_COLOR, SELECTED_BAND_COLOR = SECONDARY, PRIMARY
    BAND_HEIGHT, SELECTED_BAND_HEIGHT = 10, 12  # px
    BAND_GAP = 2  # px between two intervals that touch, so they read as two clips
    ANCHOR_DIAMETER = 10  # px: a ring smaller than the handle
    ANCHOR_PEN = 2  # px: its outline
    MASK_ANCHOR_COLORS = ('#000000', '#ffffff')  # light, dark theme

    def __init__(self, orientation) -> None:
        super().__init__(orientation)
        self.intervals: List[Tuple[int, int]] = []  # 1-indexed (start, end)
        self.selected_interval: Optional[Tuple[int, int]] = None
        self.mask_anchors: Tuple[int, ...] = ()  # 0-indexed frames
        self.empty_anchors: Tuple[int, ...] = ()

    def set_anchors(self, mask_tis, empty_tis) -> None:
        mask_tis, empty_tis = tuple(sorted(mask_tis)), tuple(sorted(empty_tis))
        if (mask_tis, empty_tis) != (self.mask_anchors, self.empty_anchors):
            self.mask_anchors, self.empty_anchors = mask_tis, empty_tis
            self.update()

    def set_intervals(self, intervals) -> None:
        self.intervals = [tuple(interval) for interval in intervals]
        self.update()

    def set_selected_interval(self, interval: Optional[Tuple[int, int]]) -> None:
        if interval != self.selected_interval:
            self.selected_interval = interval
            self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            event.ignore()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        # the bands and the anchor rings go on top (the theme paints an opaque background), around
        # the handle
        super().paintEvent(event)
        self.paint_interval_bands()
        self.paint_anchor_marks()

    def anchor_ring(self, ti: int, centre) -> QRectF:
        # the ring's path for frame ti (0-indexed), centred on the groove; the pen straddles it
        radius = (self.ANCHOR_DIAMETER - self.ANCHOR_PEN) / 2
        y = QRectF(self.groove_rect()).center().y()
        return QRectF(centre(ti + 1) - radius, y - radius, 2 * radius, 2 * radius)

    def paint_anchor_marks(self) -> None:
        if not (self.mask_anchors or self.empty_anchors) or self.maximum() <= self.minimum():
            return
        centre, _, _, _, handle = self.frame_geometry()
        painter = self.mark_painter(handle)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        dark = int(is_dark(self))
        for frames, color in ((self.mask_anchors, self.MASK_ANCHOR_COLORS[dark]),
                              (self.empty_anchors, STATUS_COLORS['empty anchor'][dark])):
            painter.setPen(QPen(QColor(color), self.ANCHOR_PEN))
            for ti in frames:
                painter.drawEllipse(self.anchor_ring(ti, centre))
        painter.end()

    def paint_interval_bands(self) -> None:
        frames = self.maximum() - self.minimum() + 1
        # one interval over every frame would only fill the bar
        if not self.intervals or frames < 2 or self.intervals == [(1, frames)]:
            return
        centre, step, _, _, handle = self.frame_geometry()
        painter = self.mark_painter(handle)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        groove = self.groove_rect()
        for start, end in self.intervals:
            selected = (start, end) == self.selected_interval
            height = self.SELECTED_BAND_HEIGHT if selected else self.BAND_HEIGHT
            top = groove.center().y() - height / 2 + 0.5
            radius = height / 2
            left = max(centre(start) - step / 2, groove.left()) + self.BAND_GAP / 2
            right = min(centre(end) + step / 2, groove.right()) - self.BAND_GAP / 2
            painter.setBrush(QColor(self.SELECTED_BAND_COLOR if selected else self.BAND_COLOR))
            width = max(right - left, 2)
            painter.drawRoundedRect(QRectF(left, top, width, height), min(radius, width / 2),
                                    radius)
        painter.end()

    def groove_rect(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                           QStyle.SubControl.SC_SliderGroove, self)

    def mark_painter(self, handle) -> QPainter:
        # paints around the handle, which stays as the theme draws it
        painter = QPainter(self)
        painter.setClipRegion(QRegion(self.rect()) - QRegion(handle, QRegion.RegionType.Ellipse))
        return painter

    def frame_geometry(self):
        """(centre(frame) -> x of the handle's centre on that 1-indexed frame, px per frame, top
        and height of the marks, the handle's rectangle)."""
        frames = self.maximum() - self.minimum() + 1
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                      QStyle.SubControl.SC_SliderHandle, self)
        span = groove.width() - handle.width()
        step = span / max(1, frames - 1)

        def centre(frame: int) -> float:  # x of the handle's centre on that frame (1-indexed)
            position = QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), frame - 1,
                                                      span, option.upsideDown)
            return groove.left() + position + handle.width() / 2

        height = max(handle.height() - 2, groove.height() + 6)
        top = groove.center().y() - height / 2
        return centre, step, top, height, handle


def frame_key_tip(key: str, plain: str, shift: str, alt: str, ctrl: str) -> str:
    """The hover tip of a frame key box: each key combination and what it does."""
    rows = ((key, plain), (f'Shift+{key}', shift), (f'Alt+{key}', alt), (f'Ctrl+{key}', ctrl))
    table = ''.join(f'<tr><td><b>{keys}</b>&nbsp;&nbsp;</td><td>{what}</td></tr>'
                    for keys, what in rows)
    return (f'<table>{table}</table>'
            'The slider also takes Home / End (first / last frame) and PageUp / PageDown '
            '(10 frames).')


def key_box(text: str, tooltip: str) -> QLabel:
    # a pill that names a key (gui/style.py)
    box = QLabel(text)
    box.setToolTip(tooltip)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.setProperty('pill', 'key')
    return box


class TextBox(QLineEdit):
    """A description box. While it has the cursor it keeps the keys: the usual text keys edit the
    text (Ctrl+C / X / V, Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y, Ctrl+A, Home / End, Ctrl + arrows, Shift
    to select), and no window shortcut acts on the frames, anchors, zoom or masks, except Ctrl+S
    (save, which saves the text too), Ctrl+Q (quit and mark done, which saves it too) and F1
    (tips). The Mac keys work too: Command (Meta) and Option (Alt) as in a Mac text field."""

    # the window shortcuts that still work while typing
    WINDOW_KEYS = (QKeySequence('Ctrl+S'), QKeySequence('Ctrl+Q'), QKeySequence(Qt.Key.Key_F1))

    def __init__(self, *args) -> None:
        super().__init__(*args)
        # what a Linux text box lacks: Ctrl+Y (the window's redo key) and the Mac keys
        keys = {
            'Ctrl+Y': self.redo,
            'Meta+Z': self.undo,
            'Meta+Shift+Z': self.redo,
            'Meta+X': self.cut,
            'Meta+C': self.copy,
            'Meta+V': self.paste,
            'Meta+A': self.selectAll,
            'Meta+Left': lambda: self.home(False),
            'Meta+Right': lambda: self.end(False),
            'Meta+Shift+Left': lambda: self.home(True),
            'Meta+Shift+Right': lambda: self.end(True),
            'Meta+Backspace': self.delete_to_start,
            # (Option + Shift + arrows are not bound)
            'Alt+Left': lambda: self.cursorWordBackward(False),
            'Alt+Right': lambda: self.cursorWordForward(False),
            'Alt+Backspace': self.delete_word_back,
            # one line: up / down go to its start / end, as on a Mac
            'Up': lambda: self.home(False),
            'Down': lambda: self.end(False),
        }
        self.extra_keys = {QKeySequence(text).toString(): action for text, action in keys.items()}

    def event(self, event) -> bool:
        # a key that is a window shortcut comes here first: the box keeps it (but the window keys)
        if (event.type() == QEvent.Type.ShortcutOverride
                and QKeySequence(event.keyCombination()) not in self.WINDOW_KEYS):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        action = self.extra_keys.get(QKeySequence(event.keyCombination()).toString())
        if action is None:
            super().keyPressEvent(event)
        else:
            action()
            event.accept()

    def delete_to_start(self) -> None:
        self.home(True)
        if self.hasSelectedText():
            self.del_()

    def delete_word_back(self) -> None:
        if not self.hasSelectedText():
            self.cursorWordBackward(True)
        if self.hasSelectedText():
            self.del_()


class IntervalBox(QSpinBox):
    """A frame-number box that also takes 'm' for its limit: the first frame in the start box, the
    last frame in the end box. Confirming (Enter, or leaving the box) turns the 'm' into that
    number. The box of an added interval can also be blank (value 0, shown empty with 'start' or
    'end' in grey) until a frame number is typed."""

    BLANK = 0

    def __init__(self, which: str, blank_allowed: bool = False) -> None:
        super().__init__()
        self.which = which  # 'start' or 'end'
        self.blank_allowed = blank_allowed
        if blank_allowed:
            self.lineEdit().setPlaceholderText(which)

    def is_blank(self) -> bool:
        return self.blank_allowed and self.value() == self.BLANK

    @staticmethod
    def is_limit_letter(text: str) -> bool:
        return text.strip().lower() == 'm'

    def textFromValue(self, value: int) -> str:
        return '' if self.blank_allowed and value == self.BLANK else super().textFromValue(value)

    def valueFromText(self, text: str) -> int:
        if self.blank_allowed and not text.strip():
            return self.BLANK
        return super().valueFromText(text)

    def validate(self, text: str, pos: int):
        if self.is_limit_letter(text):
            return QValidator.State.Intermediate, text, pos  # typed, but not a number yet
        if self.blank_allowed and not text.strip():
            return QValidator.State.Acceptable, text, pos  # cleared: the box is blank again
        return super().validate(text, pos)

    def fixup(self, text: str) -> str:
        # called on confirming a text that is not a number yet
        if self.is_limit_letter(text):
            return str(max(self.minimum(), 1) if self.which == 'start' else self.maximum())
        super().fixup(text)
        return text


REMOVE_INTERVAL_LABEL = '- Remove interval'
# the label in a message, kept on one line when the line wraps: a word joiner (no break after
# the '-') and a no-break space
REMOVE_INTERVAL_TEXT = REMOVE_INTERVAL_LABEL.replace('- ', '-\u2060\u00a0', 1)


class IntervalPair(QWidget):
    """One kept interval of a task: [start] - [end], each interval a clip of its own. Every pair
    but the first has a button that removes it (or cancels it, when it was never complete); an
    added pair starts blank. An added pair also has its own description box and category; the
    first interval's are the sequence's. The window shows one set (category, description,
    interval) at a time and places an added pair's description_edit and remove_button."""

    removed = Signal(object)
    invalid = False  # overlaps another pair, so it is not saved

    def __init__(self, frame_count: int, start: Optional[int] = None, end: Optional[int] = None,
                 removable: bool = False, description: Optional[str] = None,
                 category: Optional[str] = None) -> None:
        super().__init__()
        self.removable = removable
        self.category = category  # an added interval's; None for the first (the sequence's)
        self.description_edit: Optional[QLineEdit] = None
        # the text an emptied description box gets back: the last one kept
        self.kept_description = description or ''
        if removable:
            self.description_edit = TextBox()
            self.set_description(self.kept_description)
        self.start_box = IntervalBox('start', blank_allowed=removable)
        self.end_box = IntervalBox('end', blank_allowed=removable)
        for box, value in ((self.start_box, start), (self.end_box, end)):
            box.setRange(IntervalBox.BLANK if removable else 1, frame_count)
            box.setValue(value if value is not None else IntervalBox.BLANK)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.start_box)
        layout.addWidget(QLabel('-'))
        layout.addWidget(self.end_box)
        # the interval's frames, both ends included, as the numbers are typed; as wide as the
        # longest count, so the pair keeps its width
        self.count_label = QLabel()
        self.count_label.setProperty('intervalCount', True)  # muted (gui/style.py)
        self.count_label.setToolTip('Frames in this interval, its start and end frames included.')
        self.count_label.setMinimumWidth(
            self.count_label.fontMetrics().horizontalAdvance(f'{frame_count} frames'))
        layout.addSpacing(6)
        layout.addWidget(self.count_label)
        for box in self.boxes():
            box.valueChanged.connect(self.update_count)
        self.update_count()
        self.remove_button: Optional[QPushButton] = None
        if removable:
            self.remove_button = QPushButton(REMOVE_INTERVAL_LABEL)
            self.remove_button.setToolTip('Remove this interval (cancel it, if it was never complete).')
            self.remove_button.clicked.connect(lambda: self.removed.emit(self))
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def boxes(self):
        return self.start_box, self.end_box

    def description(self) -> Optional[str]:
        # the added interval's text as typed (stripped); None for the first interval
        return self.description_edit.text().strip() if self.description_edit is not None else None

    def set_description(self, text: str) -> None:
        self.description_edit.setText(text)
        self.description_edit.setCursorPosition(0)  # show the start of a long text
        self.description_edit.setToolTip(f'{text}\n\nThis interval\'s description: Enter or click '
                                         'elsewhere to save; a change needs a new review.')

    def frames_text(self) -> str:
        # the interval as typed, … for a blank box: '95-140', '150-…'
        start, end = (('…' if box.is_blank() else str(box.value())) for box in self.boxes())
        return f'{start}-{end}'

    def entry(self) -> Optional[Tuple[int, int]]:
        # (start, end), or None while a box is blank (an added interval not filled in yet)
        if self.start_box.is_blank() or self.end_box.is_blank():
            return None
        return self.start_box.value(), self.end_box.value()

    def update_count(self) -> None:
        # empty while a box is blank, or while the start typed is past the end (order() fixes it
        # when the box is confirmed)
        entry = self.entry()
        if entry is None or entry[0] > entry[1]:
            self.count_label.setText('')
            return
        count = entry[1] - entry[0] + 1
        self.count_label.setText(f'{count} frame{"" if count == 1 else "s"}')

    def order(self, edited_box: str) -> None:
        # a start after the end moves the other box, as the single interval always did
        entry = self.entry()
        if entry is None or entry[0] <= entry[1]:
            return
        if edited_box == 'start':
            self.end_box.setValue(entry[0])
        else:
            self.start_box.setValue(entry[1])

    def set_invalid(self, invalid: bool) -> None:
        # a red frame around a pair that overlaps another (gui/style.py); it is not saved until
        # fixed
        for box in self.boxes():
            if box.property('invalid') != invalid:
                box.setProperty('invalid', invalid)
                restyle(box)
        self.invalid = invalid


class ElidedLabel(QLabel):
    """A one-line label that cuts its text with '…' to the width it is given; the whole text is
    its tooltip."""

    def __init__(self, text: str = '') -> None:
        super().__init__()
        self.full_text = ''
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self.full_text = text
        self.setToolTip(text)
        self.elide()

    def elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideRight,
                                                   max(self.width(), 40)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.elide()


class IntervalSetRow(QFrame):
    """A row of the labeling window's set list: the set's number, its category pill, its
    description (cut to fit) and its frames; tinted when it is the selected set. A click selects
    it."""

    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setProperty('intervalSet', True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.number = QLabel()
        self.number.setProperty('intervalCount', True)  # muted
        self.number.setFixedWidth(22)
        self.number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.category = QLabel()
        self.description = ElidedLabel()
        self.frames = QLabel()  # 'Interval: 95-140', the word in bold
        self.frames.setTextFormat(Qt.TextFormat.RichText)
        self.frames.setMinimumWidth(
            self.frames.fontMetrics().horizontalAdvance('Interval: 00000-00000') + 8)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(10)
        layout.addWidget(self.number)
        layout.addWidget(self.category)
        layout.addWidget(self.description, 1)
        layout.addWidget(self.frames)

    def show_set(self, number: int, category: str, color: Optional[str], description: str,
                 frames: str, selected: bool) -> None:
        self.number.setText(str(number))
        self.category.setText(category)
        category_pill(self.category, color)
        self.description.set_full_text(description)
        self.frames.setText(f'<b>Interval:</b> {html.escape(frames)}')
        if self.property('selected') != selected:
            self.setProperty('selected', selected)
            restyle(self)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


def create_parameter_box(min_val: float, max_val: float, text: str, step: float=1, callback=None):
    layout = QHBoxLayout()

    dial = QSpinBox()
    dial.setMaximumHeight(28)
    dial.setMaximumWidth(150)
    dial.setMinimum(min_val)
    dial.setMaximum(max_val)
    dial.setAlignment(Qt.AlignmentFlag.AlignRight)
    dial.setSingleStep(step)
    dial.valueChanged.connect(callback)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(dial)

    return dial, layout


def create_gauge(text: str):
    layout = QHBoxLayout()

    gauge = QProgressBar()
    gauge.setMaximumHeight(28)
    gauge.setMaximumWidth(200)
    gauge.setAlignment(Qt.AlignmentFlag.AlignCenter)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(gauge)

    return gauge, layout


def apply_to_all_children_widget(layout, func):
    # deliberately non-recursive
    for i in range(layout.count()):
        func(layout.itemAt(i).widget())


def layout_widget(layout: QLayout) -> QWidget:
    # a layout as one widget, e.g. to wrap it as a unit or put it in a scroll area
    widget = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(layout)
    return widget


def fit_window(window: QWidget, preferred: QSize, fill: bool = False) -> None:
    """Shows a top-level window at its preferred size, centred on the screen, or maximized when
    that does not fit. A window that stops fitting later, e.g. when the screen's resolution
    changes, is maximized then. fill: maximized from the start; un-maximized from its title bar,
    it has the preferred size, centred."""
    screen = window.screen()
    available = screen.availableGeometry()
    outer = preferred + WINDOW_FRAME
    fits = outer.width() <= available.width() and outer.height() <= available.height()
    if fits:
        window.resize(preferred)
        frame = QRect(QPoint(0, 0), outer)
        frame.moveCenter(available.center())
        window.move(frame.topLeft())
    if fill or not fits:
        window.showMaximized()
    else:
        window.show()
    ScreenFollower(window, screen)


class ScreenFollower(QObject):
    """Keeps a window on a screen whose size changes. A child of the window, so the connection
    ends with the window."""

    def __init__(self, window: QWidget, screen) -> None:
        super().__init__(window)
        screen.availableGeometryChanged.connect(self.on_available_changed)

    def on_available_changed(self, available: QRect) -> None:
        window = self.parent()
        if not window.isVisible() or window.isMaximized() or window.isFullScreen():
            return  # the window manager fits a maximized window to the new size
        frame = window.frameGeometry()
        if frame.width() > available.width() or frame.height() > available.height():
            window.showMaximized()
        elif not available.contains(frame):
            # still fits, but partly off the screen
            window.move(min(max(frame.x(), available.left()), available.right() + 1 - frame.width()),
                        min(max(frame.y(), available.top()), available.bottom() + 1 - frame.height()))