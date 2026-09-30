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


def apply_theme(theme: str) -> bool:
    """Apply the theme to the whole application, windows that are already open included. Without
    the theme package Qt's built-in Fusion style is used, which has no themes: returns False."""
    try:
        import qdarktheme
        qdarktheme.setup_theme(theme)
        return True
    except (ImportError, AttributeError):
        QApplication.setStyle('Fusion')
        return False
