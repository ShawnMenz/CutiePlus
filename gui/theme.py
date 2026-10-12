"""
The color theme of the GUI: Auto follows the desktop's light/dark setting, Light and Dark are fixed.
The task browser's theme button cycles them. The choice is kept per screen (DISPLAY) in the user's
Qt settings, so every VNC screen of one account keeps its own, and is applied when the program starts.
"""
import importlib.util
import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

THEMES = ('auto', 'light', 'dark')
DEFAULT_THEME = 'auto'


def settings() -> QSettings:
    return QSettings('CutiePlus', 'Cutie')


def settings_key() -> str:
    return f'theme/{os.environ.get("DISPLAY") or "default"}'


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
    package decides)."""
    if theme != 'auto':
        return theme
    try:
        import darkdetect
        detected = darkdetect.theme()
    except Exception:  # no desktop setting to read (a VNC screen)
        detected = None
    return detected.lower() if detected else 'dark'


def apply_theme(theme: str) -> bool:
    """Apply the theme to the whole application, windows that are already open included: the
    theme package's widgets with the app's accent and the style sheet of gui/style.py, in the
    light or the dark tones. Without the theme package Qt's built-in Fusion style is used, which
    has no themes: returns False."""
    from gui.style import CUSTOM_COLORS, app_font, style_sheet
    QApplication.setFont(app_font())
    try:
        import qdarktheme
        # Auto is resolved here: the theme package's own Auto watches the desktop with a
        # gsettings process that outlives a window killed on restart, and a VNC screen's desktop
        # never switches themes anyway
        resolved = effective_theme(theme)
        qdarktheme.setup_theme(resolved, custom_colors=CUSTOM_COLORS,
                               additional_qss=style_sheet(resolved == 'dark'))
        return True
    except (ImportError, AttributeError):
        QApplication.setStyle('Fusion')
        return False


def setting(name: str, default=None):
    """A per-screen choice of the window (e.g. whether the Advanced card is open), kept as the
    theme is."""
    return settings().value(f'{name}/{os.environ.get("DISPLAY") or "default"}', default)


def save_setting(name: str, value) -> None:
    stored = settings()
    stored.setValue(f'{name}/{os.environ.get("DISPLAY") or "default"}', value)
    stored.sync()
