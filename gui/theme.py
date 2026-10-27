"""
The color theme of the GUI: Auto follows the desktop's light/dark setting, Light and Dark are fixed,
and Eye care is Light with its white and light-gray surfaces (the window, the cards, the boxes, the
buttons, every table row) in cream (#FAF9DE). The task browser's theme button cycles them. The
choice is kept in the user's Qt settings and applied when the program starts.
"""
import importlib.util

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

EYE_CARE = 'eye care'
THEMES = ('auto', 'light', 'dark', EYE_CARE)
DEFAULT_THEME = 'auto'


def settings() -> QSettings:
    return QSettings('CutiePlus', 'Cutie')


def settings_key() -> str:
    return 'theme'


def saved_theme() -> str:
    value = settings().value(settings_key(), DEFAULT_THEME)
    return value if value in THEMES else DEFAULT_THEME


def save_theme(theme: str) -> None:
    stored = settings()
    stored.setValue(settings_key(), theme)
    stored.sync()


def next_theme(theme: str) -> str:
    return THEMES[(THEMES.index(theme) + 1) % len(THEMES)]


def has_theme_package() -> bool:
    return importlib.util.find_spec('qdarktheme') is not None


def effective_theme(theme: str) -> str:
    """light or dark: Auto is the desktop's setting, dark when it cannot be read (as the theme
    package decides); Eye care is light."""
    if theme == EYE_CARE:
        return 'light'
    if theme != 'auto':
        return theme
    try:
        import darkdetect
        detected = darkdetect.theme()
    except Exception:  # no desktop setting to read
        detected = None
    return detected.lower() if detected else 'dark'


def apply_theme(theme: str) -> bool:
    """Apply the theme to the whole application, windows that are already open included: the
    theme package's widgets with the app's accent and the style sheet of gui/style.py, in the
    light or the dark tones (Eye care: the light ones, its surfaces cream). Without the theme package
    Qt's built-in Fusion style is used, which has no themes: returns False."""
    from gui.style import app_font, custom_colors, style_sheet
    QApplication.setFont(app_font())
    try:
        import qdarktheme
        # Auto is resolved here: the theme package's own Auto watches the desktop with a
        # gsettings process that outlives a window killed on restart
        resolved = effective_theme(theme)
        eye_care = theme == EYE_CARE
        qdarktheme.setup_theme(resolved, custom_colors=custom_colors(eye_care),
                               additional_qss=style_sheet(resolved == 'dark', eye_care))
        return True
    except (ImportError, AttributeError):
        QApplication.setStyle('Fusion')
        return False


def setting(name: str, default=None):
    """A choice of the window (e.g. whether the Advanced card is open), kept as the theme is."""
    return settings().value(name, default)


def save_setting(name: str, value) -> None:
    stored = settings()
    stored.setValue(name, value)
    stored.sync()
