"""
The look of every window (gui/style.py): the green accent and the style sheet on top of the theme,
button variants, rounded cards (the Advanced card folds and remembers it per screen), and the
task browser's status pills. The eye-care theme is the light one with every white and light-gray
surface (the window, the cards, the boxes, the buttons, every table row) in #FAF9DE.
"""
import subprocess
import sys

import pytest

from gui import style
from tests.conftest import compose_gui_config
from tests.helpers import make_task

ROOT = __file__.rsplit('/tests/', 1)[0]


def test_the_style_sheet_has_the_accent_in_both_tones():
    assert style.CUSTOM_COLORS['[light]']['primary'] == '#00853E'
    for dark in (False, True):
        sheet = style.style_sheet(dark)
        assert '#00853E' in sheet and '#509E2F' in sheet
        assert 'QPushButton[variant="primary"]' in sheet and 'QFrame[card="true"]' in sheet
        assert style.TONES[dark]['card'] in sheet
    assert style.style_sheet(False) != style.style_sheet(True)


def test_the_eye_care_sheet_has_every_surface_in_cream():
    light, eye_care = style.style_sheet(False), style.style_sheet(False, eye_care=True)
    tones = style.TONES[False]
    surfaces = ('window', 'card', 'input', 'button', 'alternate')  # white and light gray
    assert {tones[name] for name in surfaces} == {'#ffffff', '#f1f3f2', '#f7f9f8'}
    for color in ('#ffffff', '#f1f3f2', '#f7f9f8', '#eef2f0', 'rgba(255, 255, 255, 0.75)'):
        assert color in light and color not in eye_care  # the light sheet keeps them
    assert 'alternate-background-color: #FAF9DE' in eye_care  # the rows no longer alternate
    assert 'QPushButton:hover { background: #F0EFD5; }' in eye_care  # a deeper cream
    for line in (tones['border'], tones['input_border'], tones['groove']):  # lines and tracks
        assert line in eye_care
    light_colors = style.custom_colors()['[light]']
    eye_care_colors = style.custom_colors(eye_care=True)['[light]']
    assert eye_care_colors['background'] == eye_care_colors['input.background'] == '#FAF9DE'
    assert eye_care_colors['primary'] == light_colors['primary']
    assert style.custom_colors(eye_care=True)['[dark]'] == style.CUSTOM_COLORS['[dark]']


def test_the_theme_button_cycles_through_eye_care():
    from gui.theme import THEMES, effective_theme, next_theme
    assert THEMES == ('auto', 'light', 'dark', 'eye care')
    assert [next_theme(theme) for theme in THEMES] == ['light', 'dark', 'eye care', 'auto']
    assert effective_theme('eye care') == 'light'


@pytest.mark.parametrize('theme', ['light', 'dark', 'eye care'])
def test_apply_theme_sets_the_style_and_the_font(theme):
    # in a process of its own: a theme is set for the whole application; eye care leaves no white
    # and no gray window (the theme package writes colors as rgba), and a table is cream in every row
    code = (
        'import re\n'
        'from PySide6.QtWidgets import QApplication, QTableWidget\n'
        'app = QApplication([])\n'
        'from gui.theme import apply_theme\n'
        'from gui.style import style_sheet\n'
        f'assert apply_theme({theme!r})\n'
        f'assert app.styleSheet().endswith(style_sheet({theme == "dark"}, {theme == "eye care"}))\n'
        'assert app.font().family() in ("Noto Sans", "Ubuntu Sans", "DejaVu Sans")\n'
        'white = re.findall(r"#ffffff|rgba\\(255, 255, 255, 1\\.000\\)|rgba\\(241, 243, 242, 1\\.000\\)",'
        ' app.styleSheet(), re.I)\n'
        'cream = app.styleSheet().count("rgba(250, 249, 222, 1.000)")\n'
        f'assert (not white and cream > 0) if {theme == "eye care"} else (white and not cream)\n'
        'table = QTableWidget(2, 2)\n'
        'table.setAlternatingRowColors(True)\n'
        'table.ensurePolished()\n'
        'colors = {table.palette().window().color().name(), table.palette().base().color().name(),'
        ' table.palette().alternateBase().color().name()}\n'
        f'assert (colors == {{"#faf9de"}}) == {theme == "eye care"}, colors\n'
        'print("ok")\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                            env={'QT_QPA_PLATFORM': 'offscreen', 'HOME': '/tmp',
                                 'PATH': '/usr/bin:/bin'}, timeout=120)
    assert result.stdout.strip().endswith('ok'), result.stderr


def test_a_collapsible_card_shows_and_hides_its_content(qapp):
    from PySide6.QtWidgets import QLabel
    card = style.Card(QLabel('inside'), title='Advanced', collapsible=True, expanded=False)
    seen = []
    card.toggled.connect(seen.append)
    assert not card.is_expanded() and card.property('card')
    card.toggle.click()
    assert card.is_expanded() and seen == [True]
    card.set_expanded(False)
    assert not card.is_expanded() and seen == [True, False]
    card.deleteLater()


@pytest.mark.models
@pytest.mark.slow
def test_the_window_uses_the_variants_and_remembers_the_advanced_card(tmp_path, qapp, cpu_networks,
                                                                     monkeypatch):
    from PySide6.QtCore import QSettings
    from gui import theme
    from gui.main_controller import MainController
    stored = str(tmp_path / 'settings.ini')
    monkeypatch.setattr(theme, 'settings', lambda: QSettings(stored, QSettings.Format.IniFormat))
    task = make_task(str(tmp_path), frame_count=12)

    def open_window():
        return MainController(compose_gui_config(task.workspace), networks=cpu_networks)

    gui = open_window().gui
    try:
        variants = {button.text(): button.property('variant') for button in
                    (gui.forward_run_button, gui.backward_run_button, gui.export_binary_button,
                     gui.commit_button, gui.reset_all_masks_button, gui.attention_button)}
        assert variants == {'Propagate forward (F)': 'primary',
                            'Propagate backward (B)': 'primary',
                            'Save binary PNGs (Ctrl+S)': 'primary', 'Mark as anchor (C)': 'tonal',
                            'Reset all frame masks': 'danger', 'Need Attention': 'attention'}
        gui.on_add_interval()
        assert gui.interval_pairs[1].remove_button.property('variant') == 'danger'
        assert gui.task_label.property('pill') == 'accent'
        # the controls of the original Cutie wait folded in the Advanced card
        assert not gui.advanced_card.is_expanded()
        assert gui.advanced_card.body.isAncestorOf(gui.clear_all_mem_button)
        assert gui.advanced_card.body.isAncestorOf(gui.reset_all_masks_button)
        gui.advanced_card.toggle.click()
        assert gui.advanced_card.is_expanded()
    finally:
        gui.close()
    gui = open_window().gui  # open again on this screen: as it was left
    try:
        assert gui.advanced_card.is_expanded()
    finally:
        gui.close()


@pytest.mark.slow
def test_the_browser_draws_the_status_as_pills(tmp_path, qapp):
    from PySide6.QtGui import QFontMetrics
    from gui.task_browser import STATUS_COLUMN, STATUS_COLORS, StatusDelegate, TaskBrowser
    task = make_task(str(tmp_path), frame_count=6)
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    try:
        table = list(browser.tables.values())[0]
        assert isinstance(table.itemDelegateForColumn(STATUS_COLUMN), StatusDelegate)
        widest = max(QFontMetrics(table.font()).horizontalAdvance(status) for status in STATUS_COLORS)
        assert table.columnWidth(STATUS_COLUMN) > widest + 2 * StatusDelegate.PADDING
        browser.resize(1100, 700)
        assert not browser.grab().isNull()  # paints the pills
        assert table.item(0, STATUS_COLUMN).text() == 'not started'  # the item keeps its text
    finally:
        browser.close()
