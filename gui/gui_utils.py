import html
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPalette, QValidator
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLayout, QPlainTextEdit, QSlider, QSpinBox,
                               QProgressBar, QWidget)

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
    'prefill': ('#8d7b00', '#e3cf4a'),
    'no mask': ('#757575', '#b0b0b0'),
}
STATUS_MEANING = {
    'anchor': 'a human decision, never overwritten',
    'empty anchor': 'target absent: propagation stops here',
    'propagated': 'written by Cutie',
    'prefill': 'model mask, not reviewed yet',
    'no mask': 'nothing yet',
}


def is_dark(widget: QWidget) -> bool:
    return widget.palette().color(QPalette.ColorRole.Window).lightness() < 128


def status_kind(source: str) -> str:
    return source.split(':')[0].strip()


def frame_label_html(file_name: str, modality: Optional[str], source: Optional[str],
                     dark: bool) -> str:
    """'000001.jpg (RGB) [anchor: ls]' with the modality and the status colored: visual is shown
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
    would otherwise step on them); Home / End and PageUp / PageDown still move it."""

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            event.ignore()
            return
        super().keyPressEvent(event)


def frame_key_tip(key: str, plain: str, shift: str, alt: str, ctrl: str) -> str:
    """The hover tip of a frame key box: each key combination and what it does."""
    rows = ((key, plain), (f'Shift+{key}', shift), (f'Alt+{key}', alt), (f'Ctrl+{key}', ctrl))
    table = ''.join(f'<tr><td><b>{keys}</b>&nbsp;&nbsp;</td><td>{what}</td></tr>'
                    for keys, what in rows)
    return (f'<table>{table}</table>'
            'The slider also takes Home / End (first / last frame) and PageUp / PageDown '
            '(10 frames).')


def key_box(text: str, tooltip: str) -> QLabel:
    # a small bordered label that names a key; the border follows the theme
    box = QLabel(text)
    box.setToolTip(tooltip)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.setStyleSheet('QLabel { border: 1px solid palette(mid); border-radius: 4px; '
                      'padding: 1px 8px; }')
    return box


class IntervalBox(QSpinBox):
    """A frame-number box that also takes 'm' for its limit: the first frame in the start box, the
    last frame in the end box. Confirming (Enter, or leaving the box) turns the 'm' into that
    number."""

    def __init__(self, which: str) -> None:
        super().__init__()
        self.which = which  # 'start' or 'end'

    @staticmethod
    def is_limit_letter(text: str) -> bool:
        return text.strip().lower() == 'm'

    def validate(self, text: str, pos: int):
        if self.is_limit_letter(text):
            return QValidator.State.Intermediate, text, pos  # typed, but not a number yet
        return super().validate(text, pos)

    def fixup(self, text: str) -> str:
        # called on confirming a text that is not a number yet
        if self.is_limit_letter(text):
            return str(self.minimum() if self.which == 'start' else self.maximum())
        super().fixup(text)
        return text


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


def fit_window(window: QWidget, preferred: QSize) -> None:
    """Shows a top-level window at its preferred size, centred on the screen, or maximized when
    that does not fit. A window that stops fitting later, e.g. when a VNC viewer shrinks the
    remote desktop to its own window, is maximized then."""
    screen = window.screen()
    available = screen.availableGeometry()
    outer = preferred + WINDOW_FRAME
    if outer.width() <= available.width() and outer.height() <= available.height():
        window.resize(preferred)
        frame = QRect(QPoint(0, 0), outer)
        frame.moveCenter(available.center())
        window.move(frame.topLeft())
        window.show()
    else:
        window.showMaximized()
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