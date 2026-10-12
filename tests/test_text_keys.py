"""
The description boxes keep the keys while they have the cursor: the text keys edit the text
(Ctrl+C / X / V, Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y, Ctrl+A, ...) and no window shortcut acts on the
frames, anchors, zoom or masks, except Ctrl+S and F1. A Mac's Command key (Meta) and Option (Alt)
work as in a Mac text field.
"""
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent

from tests.conftest import compose_gui_config
from tests.helpers import make_task

CTRL, SHIFT = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier
META, ALT = Qt.KeyboardModifier.MetaModifier, Qt.KeyboardModifier.AltModifier
NONE = Qt.KeyboardModifier.NoModifier


@pytest.fixture
def box(qapp):
    from gui.gui_utils import TextBox
    widget = TextBox()
    yield widget
    widget.deleteLater()


def kept(box, key, modifiers=NONE) -> bool:
    # whether the box takes a key that is also a window shortcut
    event = QKeyEvent(QEvent.Type.ShortcutOverride, key, modifiers)
    event.ignore()
    QGuiApplication.sendEvent(box, event)
    return event.isAccepted()


def press(box, key, modifiers=NONE):
    from PySide6.QtTest import QTest
    QTest.keyClick(box, key, modifiers)


def test_the_window_keys_stay_in_the_box_but_save_and_tips(box):
    for key, modifiers in ((Qt.Key.Key_D, CTRL), (Qt.Key.Key_A, CTRL), (Qt.Key.Key_Y, CTRL),
                           (Qt.Key.Key_Z, CTRL), (Qt.Key.Key_Z, CTRL | SHIFT),
                           (Qt.Key.Key_Up, NONE), (Qt.Key.Key_Down, NONE),
                           (Qt.Key.Key_0, CTRL), (Qt.Key.Key_1, CTRL), (Qt.Key.Key_Minus, CTRL),
                           (Qt.Key.Key_Equal, CTRL), (Qt.Key.Key_A, ALT), (Qt.Key.Key_D, ALT),
                           (Qt.Key.Key_Delete, NONE), (Qt.Key.Key_E, CTRL),
                           (Qt.Key.Key_C, META), (Qt.Key.Key_V, META)):
        assert kept(box, key, modifiers), (key, modifiers)
    assert not kept(box, Qt.Key.Key_S, CTRL)  # saves, with the text typed
    assert not kept(box, Qt.Key.Key_F1)


def test_ctrl_y_redoes_and_the_mac_keys_edit(box):
    box.setText('')
    for character in 'red car':
        press(box, getattr(Qt.Key, 'Key_Space' if character == ' ' else f'Key_{character.upper()}'))
    assert box.text() == 'red car'
    press(box, Qt.Key.Key_Z, CTRL)
    assert box.text() == ''
    press(box, Qt.Key.Key_Y, CTRL)  # the window's redo key redoes the typing here
    assert box.text() == 'red car'

    # Command (Meta) as on a Mac: select all, copy, cut, paste, undo, redo, line start / end
    press(box, Qt.Key.Key_A, META)
    assert box.selectedText() == 'red car'
    press(box, Qt.Key.Key_C, META)
    assert QGuiApplication.clipboard().text() == 'red car'
    press(box, Qt.Key.Key_Right, META)
    assert box.cursorPosition() == 7 and not box.selectedText()
    press(box, Qt.Key.Key_V, META)
    assert box.text() == 'red carred car'
    press(box, Qt.Key.Key_Z, META)
    assert box.text() == 'red car'
    press(box, Qt.Key.Key_Z, META | SHIFT)
    assert box.text() == 'red carred car'
    press(box, Qt.Key.Key_Left, META | SHIFT)
    assert box.selectedText() == 'red carred car'
    press(box, Qt.Key.Key_X, META)
    assert box.text() == '' and QGuiApplication.clipboard().text() == 'red carred car'
    press(box, Qt.Key.Key_V, CTRL)

    # Option (Alt): word by word; up / down: start / end of the line
    press(box, Qt.Key.Key_Up)
    assert box.cursorPosition() == 0
    press(box, Qt.Key.Key_Right, ALT)
    assert box.cursorPosition() == 4
    press(box, Qt.Key.Key_Down)
    assert box.cursorPosition() == len(box.text())
    press(box, Qt.Key.Key_Backspace, ALT)
    assert box.text() == 'red carred '
    press(box, Qt.Key.Key_Left, ALT)
    press(box, Qt.Key.Key_Backspace, META)  # deletes up to the start
    assert box.text() == 'carred '
    box.setCursorPosition(0)
    press(box, Qt.Key.Key_Backspace, META)  # nothing before the cursor: nothing deleted
    press(box, Qt.Key.Key_Backspace, ALT)
    assert box.text() == 'carred '


@pytest.mark.models
@pytest.mark.slow
def test_the_window_description_boxes_keep_the_keys(tmp_path, qapp, cpu_networks):
    from gui.gui_utils import TextBox
    from gui.main_controller import MainController
    task = make_task(str(tmp_path), frame_count=30)
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        gui.on_add_interval()
        boxes = [gui.description_edit, gui.interval_pairs[1].description_edit]
        assert all(isinstance(box, TextBox) for box in boxes)
        assert all(kept(box, Qt.Key.Key_D, CTRL) and not kept(box, Qt.Key.Key_S, CTRL)
                   for box in boxes)
    finally:
        gui.close()
