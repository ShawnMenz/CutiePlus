"""
The box beside Play video sets the playback speed of R and Shift+R in frames per second, also
during playback; every window starts at 24, unless the Lock button beside the box keeps a speed for
the following windows (until it is unlocked), on any display. The original Cutie's layer view is not
offered (no "layer" visualization, no Import layer), its UIUC overlay is not loaded, and nothing
about it is logged.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


@pytest.fixture
def stored_settings(tmp_path, monkeypatch):
    # the user's own settings stay untouched (and a speed locked there is not read)
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


def test_lock_keeps_the_speed_for_the_following_windows(tmp_path, qapp, cpu_networks,
                                                        stored_settings, monkeypatch):
    from gui.gui import LOCK_LABEL, LOCKED_LABEL, PLAYBACK_FPS
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=6)

    def open_window():
        return MainController(compose_gui_config(task.workspace), networks=cpu_networks).gui

    gui = open_window()
    try:
        box, lock = gui.play_fps_box, gui.play_fps_lock_button
        assert lock.text() == LOCK_LABEL and lock.isCheckable() and not lock.isChecked()
        assert lock.parentWidget() is box.parentWidget()  # on the right of the box
        assert lock.geometry().left() >= box.geometry().right()
        box.setValue(10)
        lock.click()
        assert lock.isChecked() and lock.text() == LOCKED_LABEL
        assert not box.isEnabled()  # the locked speed stays
        assert 'locked at 10 fps' in gui.console.toPlainText()
    finally:
        gui.close()
    for _ in range(2):  # every following window starts at the locked speed
        gui = open_window()
        try:
            assert gui.play_fps_box.value() == 10 and gui.play_fps_lock_button.isChecked()
            assert not gui.play_fps_box.isEnabled()
            gui.on_play_video()
            assert gui.timer.interval() == 100
            gui.stop_playback()
        finally:
            gui.close()
    monkeypatch.setenv('DISPLAY', ':4')  # not kept per display
    gui = open_window()
    try:
        assert gui.play_fps_box.value() == 10 and gui.play_fps_lock_button.isChecked()
        gui.play_fps_lock_button.click()  # unlock: back to the default, here and after
        assert gui.play_fps_box.value() == PLAYBACK_FPS and gui.play_fps_box.isEnabled()
        assert gui.play_fps_lock_button.text() == LOCK_LABEL
        assert 'unlocked: back to 24 fps' in gui.console.toPlainText()
    finally:
        gui.close()
    gui = open_window()
    try:
        assert gui.play_fps_box.value() == PLAYBACK_FPS
        assert not gui.play_fps_lock_button.isChecked() and gui.play_fps_box.isEnabled()
    finally:
        gui.close()
