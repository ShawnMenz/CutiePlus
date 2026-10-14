"""
The box beside Play video sets the playback speed of R and Shift+R in frames per second, also
during playback; every window starts at 24. The original Cutie's layer view is not offered (no
"layer" visualization, no Import layer), its UIUC overlay is not loaded, and nothing about it is
logged.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


@pytest.fixture
def stored_settings(tmp_path, monkeypatch):
    # the user's own settings stay untouched (and a speed set before is not kept: none is saved)
    from PySide6.QtCore import QSettings
    from gui import theme
    stored = str(tmp_path / 'settings.ini')
    monkeypatch.setattr(theme, 'settings', lambda: QSettings(stored, QSettings.Format.IniFormat))


def test_the_fps_box_sets_the_playback_speed(tmp_path, qapp, cpu_networks, stored_settings):
    from gui.gui import PLAYBACK_FPS
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=12)

    def open_window():
        return MainController(compose_gui_config(task.workspace), networks=cpu_networks).gui

    gui = open_window()
    try:
        box = gui.play_fps_box
        assert box.value() == PLAYBACK_FPS == 24 and box.suffix() == ' fps'
        assert box.parentWidget() is gui.play_button.parentWidget()  # beside the button
        box.setValue(10)
        gui.on_play_video()  # R
        assert gui.timer.isActive() and gui.timer.interval() == 100
        box.setValue(50)  # during playback: at once
        assert gui.timer.isActive() and gui.timer.interval() == 20
        gui.on_play_video()  # R again stops it
        gui.on_preview_interval()  # Shift+R at the same speed
        assert gui.timer.isActive() and gui.timer.interval() == 20
        gui.stop_playback()
    finally:
        gui.close()
    gui = open_window()  # the next window starts at the default again
    try:
        assert gui.play_fps_box.value() == 24
        gui.on_play_video()
        assert gui.timer.interval() == round(1000 / 24)
        gui.stop_playback()
    finally:
        gui.close()


def test_the_layer_view_and_the_uiuc_overlay_are_off(tmp_path, qapp, cpu_networks, stored_settings):
    from PySide6.QtWidgets import QPushButton
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        modes = [gui.combo.itemText(index) for index in range(gui.combo.count())]
        assert modes == ['mask', 'davis', 'fade', 'light', 'popup', 'rgba']
        assert 'Import layer' not in [button.text() for button in gui.findChildren(QPushButton)]
        assert gui.advanced_card.body.isAncestorOf(gui.import_mask_button)
        assert controller.overlay_layer is None and controller.overlay_layer_torch is None
        assert 'uiuc' not in controller.gui.console.toPlainText().lower()
        assert 'Layer file' not in controller.gui.console.toPlainText()
    finally:
        controller.gui.close()
