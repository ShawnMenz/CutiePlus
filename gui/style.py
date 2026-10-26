"""
The look of every window ("soft cards"): one green accent, rounded controls, and controls grouped
into rounded cards. It stays flat (no shadows, gradients or animations), which keeps a VNC screen
quick. The theme package (pyqtdarktheme, see gui/theme.py) draws the widgets in light or dark; this
module gives it the accent and adds a style sheet on top, built for the light or the dark tones.
The eye-care theme is the light one with every white and light-gray surface (the window, the cards,
the boxes, the buttons, every table row) in EYE_CARE_WHITE.

Widgets opt in with dynamic properties, which the style sheet selects on:
  QPushButton  variant = primary (filled green), tonal (green tint), danger (red outline),
               attention (orange outline, filled while checked), toggle (plain, filled green
               while checked)
  QFrame       card = true (see Card)
  QLabel       pill = accent (green tint), key (a key name); cardTitle = true
"""
from typing import Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLayout, QToolButton,
                               QVBoxLayout, QWidget)

PRIMARY = '#00853E'  # the main color: filled buttons, the progress, the slider handle
SECONDARY = '#509E2F'  # hover, interval bands, and the accent of the dark theme
PRESSED = '#006B32'
WHITE = '#ffffff'
EYE_CARE_WHITE = '#FAF9DE'  # the eye-care theme's surfaces: a soft cream, easier on the eyes
EYE_CARE_HOVER = '#F0EFD5'  # the cream 4 % darker: a control under the pointer
ATTENTION = '#ef6c00'  # Need Attention (the task browser's Attention status is red)
DANGER = ('#d32f2f', '#ef5350')  # light, dark

RADIUS = 8  # px: buttons, boxes
CARD_RADIUS = 12
CARD_MARGINS = (12, 10, 12, 12)  # left, top, right, bottom
GAP = 8

FONT_FAMILIES = ['Noto Sans', 'Ubuntu Sans', 'DejaVu Sans']
FONT_POINTS = 10

# the theme package's own colors; it derives its hover, selection and focus shades from them
CUSTOM_COLORS = {
    '[light]': {'primary': PRIMARY, 'background': '#f1f3f2', 'foreground': '#24292f',
                'border': '#d5dbd8', 'input.background': '#ffffff'},
    '[dark]': {'primary': SECONDARY, 'background': '#17181a', 'foreground': '#e6e8eb',
               'border': '#3a3d40', 'input.background': '#2a2c2f'},
}

# the eye-care theme in the theme package: its background (the window, and the palette) and its
# inputs in cream, and the surfaces it derives from the background by lightening (tooltips, menus,
# lists, tables, text areas), which would be white otherwise
EYE_CARE_COLORS = {'background': EYE_CARE_WHITE, 'input.background': EYE_CARE_WHITE,
                   **{f'background>{surface}': EYE_CARE_WHITE
                      for surface in ('panel', 'popup', 'table', 'textarea')}}


def custom_colors(eye_care: bool = False) -> dict:
    """The theme package's colors; with eye_care, its light surfaces in the light theme are
    EYE_CARE_WHITE."""
    if not eye_care:
        return CUSTOM_COLORS
    return dict(CUSTOM_COLORS, **{'[light]': dict(CUSTOM_COLORS['[light]'], **EYE_CARE_COLORS)})


# the tones of the style sheet, for the light and the dark theme
TONES = {
    False: {
        'window': '#f1f3f2', 'card': '#ffffff', 'border': '#dfe3e1', 'input': '#ffffff',
        'input_border': '#c9d0cc', 'button': '#ffffff', 'button_hover': '#eef2f0',
        'text': '#24292f', 'muted': '#66707a', 'accent': PRIMARY,
        'tonal': 'rgba(0, 133, 62, 0.10)', 'tonal_hover': 'rgba(0, 133, 62, 0.18)',
        'tonal_text': PRESSED, 'danger': DANGER[0], 'danger_hover': 'rgba(211, 47, 47, 0.08)',
        'groove': '#dce1de', 'alternate': '#f7f9f8', 'scroll': 'rgba(0, 0, 0, 0.22)',
    },
    True: {
        'window': '#17181a', 'card': '#212225', 'border': '#323538', 'input': '#2a2c2f',
        'input_border': '#3d4044', 'button': '#2a2c2f', 'button_hover': '#33363a',
        'text': '#e6e8eb', 'muted': '#9aa1a9', 'accent': SECONDARY,
        'tonal': 'rgba(80, 158, 47, 0.22)', 'tonal_hover': 'rgba(80, 158, 47, 0.32)',
        'tonal_text': '#e6e8eb', 'danger': DANGER[1], 'danger_hover': 'rgba(239, 83, 80, 0.14)',
        'groove': '#3a3d40', 'alternate': '#25272a', 'scroll': 'rgba(255, 255, 255, 0.22)',
    },
}

STYLE_SHEET = '''
QWidget {{ font-size: {points}pt; }}
QToolTip {{ background: {card}; color: {text}; border: 1px solid {border}; border-radius: 6px;
           padding: 6px 8px; }}

QPushButton {{ background: {button}; color: {text}; border: 1px solid {input_border};
              border-radius: {radius}px; padding: 5px 14px; min-height: 20px; }}
QPushButton:hover {{ background: {button_hover}; }}
QPushButton:pressed {{ background: {tonal_hover}; }}
QPushButton:disabled {{ color: {muted}; border-color: {border}; background: {card}; }}
QPushButton[variant="primary"] {{ background: {primary}; color: {white}; border: 1px solid {primary};
                                 font-weight: 600; }}
QPushButton[variant="primary"]:hover {{ background: {secondary}; border-color: {secondary}; }}
QPushButton[variant="primary"]:pressed {{ background: {pressed}; border-color: {pressed}; }}
QPushButton[variant="primary"]:disabled {{ background: rgba(0, 133, 62, 0.35);
                                          border-color: transparent; color: {white_text_disabled}; }}
QPushButton[variant="tonal"] {{ background: {tonal}; color: {tonal_text}; border: 1px solid transparent;
                               font-weight: 600; }}
QPushButton[variant="tonal"]:hover {{ background: {tonal_hover}; }}
QPushButton[variant="danger"] {{ background: transparent; color: {danger}; border: 1px solid {danger};
                                font-weight: 600; }}
QPushButton[variant="danger"]:hover {{ background: {danger_hover}; }}
QPushButton[variant="attention"] {{ background: transparent; color: {attention};
                                   border: 1px solid {attention}; font-weight: 600; }}
QPushButton[variant="attention"]:hover {{ background: rgba(239, 108, 0, 0.10); }}
QPushButton[variant="attention"]:checked {{ background: {attention}; color: {white}; }}
QPushButton[variant="toggle"]:checked {{ background: {primary}; color: {white}; border-color: {primary}; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ background: {input}; color: {text};
    border: 1px solid {input_border}; border-radius: {radius}px; padding: 4px 8px; min-height: 20px; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {{ border: 1px solid {accent}; }}
QAbstractSpinBox[invalid="true"] {{ border: 1px solid {danger}; }}
QComboBox QAbstractItemView {{ border-radius: {radius}px; }}

QFrame[card="true"] {{ background: {card}; border: 1px solid {border}; border-radius: {card_radius}px;
                      padding: 0; }}
QWidget#cardBody {{ background: transparent; }}
QLabel[cardTitle="true"] {{ color: {muted}; font-weight: 600; background: transparent; border: none; }}
QToolButton[cardToggle="true"] {{ color: {muted}; font-weight: 600; border: none; background: transparent;
                                 padding: 0; }}
QToolButton[cardToggle="true"]:hover {{ color: {text}; }}
QToolButton[cardToggle="true"]:checked, QToolButton[cardToggle="true"]:pressed {{ background: transparent; }}
QLabel[section="true"] {{ color: {muted}; font-weight: 600; background: transparent; padding-top: 4px; }}
QLabel[intervalCount="true"] {{ color: {muted}; background: transparent; }}
QTextEdit#frameCounter {{ background: {window}; border: 1px solid {border}; border-radius: {radius}px;
                         padding: 0 4px; }}
QLabel[pill="accent"] {{ background: {tonal}; color: {tonal_text}; border-radius: 11px; padding: 2px 10px;
                        font-weight: 600; }}
QLabel[pill="key"] {{ background: {card}; color: {text}; border: 1px solid {input_border};
                     border-radius: 12px; padding: 3px 12px; }}
QLabel[pill="frame"] {{ background: {window}; border: 1px solid {border}; border-radius: 12px;
                       padding: 2px 10px; }}

QProgressBar {{ background: {groove}; border: none; border-radius: 7px; text-align: center;
               color: {text}; min-height: 14px; }}
QProgressBar::chunk {{ background: {primary}; border-radius: 7px; }}

QPlainTextEdit, QTextEdit, QPlainTextEdit:focus, QTextEdit:focus {{ background: {card}; color: {text};
                                                                    border: none; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QSplitter::handle {{ background: transparent; }}

QScrollBar:vertical {{ width: 10px; background: transparent; border: none; margin: 2px; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; border: none; margin: 2px; }}
QScrollBar::handle {{ background: {scroll}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTabWidget::pane {{ border: none; }}
QTabWidget, QTabWidget > QStackedWidget, QTabBar {{ background: transparent; }}
QTabBar::tab, QTabBar::tab:top, QTabBar::tab:bottom {{ background: transparent; color: {muted};
    border: none; border-radius: {radius}px; padding: 6px 16px; margin: 0 4px 6px 0; font-weight: 600; }}
QTabBar::tab:hover, QTabBar::tab:selected:hover:enabled {{ background: {button_hover}; }}
QTabBar::tab:selected:enabled, QTabBar::tab:only-one:selected:enabled {{ background: {tonal};
    color: {tonal_text}; border: none; }}

QTableView {{ background: {card}; alternate-background-color: {alternate}; border: none; margin: 0;
             gridline-color: transparent; selection-background-color: transparent;
             selection-color: {text}; }}
QTableView::item {{ padding: 0 6px; border: none; }}
QTableView::item:alternate {{ background: {alternate}; }}
QTableView::item:selected, QTableView::item:selected:!active {{ background: {tonal}; color: {text}; }}
QHeaderView, QTableView > QHeaderView {{ background: {card}; border: none; }}
QHeaderView::section:horizontal {{ margin-right: 0; }}
QTableView > QHeaderView::section, QTableView > QHeaderView::section:horizontal {{
    background: {card}; color: {muted}; border: none; border-bottom: 1px solid {border};
    padding: 8px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {card}; border: none; }}

QSlider#frameSlider {{ min-height: 26px; }}
QSlider#frameSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {groove}; }}
QSlider#frameSlider::sub-page:horizontal, QSlider#frameSlider::add-page:horizontal {{
    background: {groove}; border-radius: 3px; }}
QSlider#frameSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 9px;
    background: {white}; border: 3px solid {primary}; }}
'''


# the eye-care theme's tones over the light ones: every white and light-gray surface in cream (the
# rows no longer alternate); the hover a deeper cream, the lines and tracks as in the light theme
EYE_CARE_TONES = {'window': EYE_CARE_WHITE, 'card': EYE_CARE_WHITE, 'input': EYE_CARE_WHITE,
                  'button': EYE_CARE_WHITE, 'alternate': EYE_CARE_WHITE,
                  'button_hover': EYE_CARE_HOVER}


def tones(dark: bool) -> dict:
    return TONES[dark]


def style_sheet(dark: bool, eye_care: bool = False) -> str:
    """The style sheet added on top of the theme, for the light or the dark tones; eye_care: the
    light tones with every white and light-gray surface in EYE_CARE_WHITE."""
    white, colors = WHITE, TONES[dark]
    white_text_disabled = 'rgba(255, 255, 255, 0.75)'  # on a disabled filled button
    if eye_care:
        white, colors = EYE_CARE_WHITE, dict(TONES[False], **EYE_CARE_TONES)
        white_text_disabled = 'rgba(250, 249, 222, 0.75)'
    return STYLE_SHEET.format(points=FONT_POINTS, radius=RADIUS, card_radius=CARD_RADIUS,
                              primary=PRIMARY, secondary=SECONDARY, pressed=PRESSED, white=white,
                              white_text_disabled=white_text_disabled, attention=ATTENTION,
                              **colors)


def app_font() -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(FONT_POINTS)
    return font


def restyle(widget: QWidget) -> None:
    # a property the style sheet selects on changed after the widget was shown
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_variant(button: QWidget, variant: Optional[str]) -> QWidget:
    """primary, tonal, danger or attention (see the module docstring); None: a plain button."""
    button.setProperty('variant', variant or '')
    restyle(button)
    return button


def set_pill(label: QLabel, kind: str) -> QLabel:
    label.setProperty('pill', kind)
    restyle(label)
    return label


def _as_widget(content: Union[QWidget, QLayout]) -> QWidget:
    if isinstance(content, QWidget):
        return content
    widget = QWidget()
    content.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(content)
    return widget


class Card(QFrame):
    """A rounded panel that groups controls: an optional title row (a title, and widgets on its
    right), then the content. collapsible: the title is a toggle that shows or hides the content
    (toggled(bool) reports it)."""

    toggled = Signal(bool)

    def __init__(self, content: Union[QWidget, QLayout, None] = None, title: Optional[str] = None,
                 trailing=(), collapsible: bool = False, expanded: bool = True,
                 stretch: bool = False) -> None:
        super().__init__()
        self.setProperty('card', True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*CARD_MARGINS)
        layout.setSpacing(GAP)
        self.toggle: Optional[QToolButton] = None
        self.title_label: Optional[QLabel] = None
        if title is not None or trailing:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(GAP)
            if collapsible:
                self.toggle = QToolButton()
                self.toggle.setProperty('cardToggle', True)
                self.toggle.setText(title or '')
                self.toggle.setCheckable(True)
                self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
                self.toggle.toggled.connect(self.set_expanded)
                row.addWidget(self.toggle)
            elif title is not None:
                self.title_label = QLabel(title)
                self.title_label.setProperty('cardTitle', True)
                row.addWidget(self.title_label)
            row.addStretch(1)
            for widget in trailing:
                row.addWidget(widget)
            layout.addLayout(row)
        self.body = _as_widget(content) if content is not None else QWidget()
        if not self.body.objectName():
            self.body.setObjectName('cardBody')  # transparent on the card (the style sheet)
        if stretch:
            layout.addWidget(self.body, 1)
        else:  # a card taller than its content keeps the title and the content at the top
            layout.addWidget(self.body)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        if collapsible:
            self.toggle.setChecked(expanded)
            self.set_expanded(expanded)

    def is_expanded(self) -> bool:
        return self.body.isVisibleTo(self)

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle is not None:
            self.toggle.blockSignals(True)
            self.toggle.setChecked(expanded)
            self.toggle.blockSignals(False)
            self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        changed = self.body.isVisibleTo(self) != expanded
        self.body.setVisible(expanded)
        if changed:
            self.toggled.emit(expanded)


def is_dark_app() -> bool:
    palette = QApplication.palette()
    return palette.color(palette.ColorRole.Window).lightness() < 128
