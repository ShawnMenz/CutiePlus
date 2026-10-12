"""
Keyboard alternatives to the mouse (for a touchpad): , and . click at the pointer as the left
and right buttons, Ctrl + + / Ctrl + - zoom, Ctrl+0 fits the image, and the arrows scroll a zoomed
view. The mouse controls stay as they are.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


@pytest.fixture
def window(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    task = make_task(str(tmp_path))
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    qapp.processEvents()
    yield controller
    controller.gui.close()


def press(gui, keys):
    from PySide6.QtGui import QKeySequence, QShortcut
    shortcuts = [shortcut for shortcut in gui.findChildren(QShortcut)
                 if shortcut.key() == QKeySequence(keys)]
    assert len(shortcuts) == 1, keys  # two on one key would be ambiguous and neither would fire
    shortcuts[0].activated.emit()


def test_comma_and_period_click_at_the_pointer(window):
    from PySide6.QtCore import QPointF
    gui = window.gui
    scale, offset_x, offset_y = gui.view_transform()

    def point_at(x, y):  # the pointer over image pixel (x, y)
        gui.pointer_on_canvas = lambda: QPointF((x + 0.5) * scale + offset_x,
                                                (y + 0.5) * scale + offset_y)

    point_at(60, 45)
    press(gui, ',')
    point_at(70, 50)
    press(gui, '.')
    assert list(gui.canvas_clicks) == [(60, 45, False), (70, 50, True)]
    assert window.current_anchor() is not None  # as a mouse click, the frame is an anchor

    gui.pointer_on_canvas = lambda: QPointF(-10, -10)  # off the image
    press(gui, ',')
    assert len(gui.canvas_clicks) == 2
    assert 'Move the pointer over the image' in gui.console.toPlainText()


def test_ctrl_plus_minus_zero_zoom_and_the_arrows_scroll(window):
    from PySide6.QtCore import QPointF
    gui = window.gui
    gui.pointer_on_canvas = lambda: QPointF(-1, -1)  # off the canvas: zoom at the centre
    centre = gui.view_center
    press(gui, 'Left')
    assert gui.view_center == centre  # the whole image is shown: nothing to scroll
    press(gui, 'Ctrl+=')
    press(gui, 'Ctrl++')
    assert gui.zoom == pytest.approx(1.25 ** 2)
    press(gui, 'Ctrl+-')
    assert gui.zoom == pytest.approx(1.25)
    press(gui, 'Ctrl++')
    press(gui, 'Ctrl++')  # room to scroll a whole step each way, however tall the canvas is
    zoomed = gui.view_center
    press(gui, 'Right')
    assert gui.view_center[0] > zoomed[0]
    press(gui, 'Left')
    press(gui, 'Down')
    assert gui.view_center[1] > zoomed[1]
    press(gui, 'Up')
    assert gui.view_center == pytest.approx(zoomed)
    press(gui, 'Ctrl+0')
    assert gui.zoom == 1.0 and gui.view_center == pytest.approx(centre)
