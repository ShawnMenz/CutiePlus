"""
A labeling window fills the screen; other windows open at their preferred size when it fits.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task

SMALLEST_DESKTOP = (1379, 796)  # a small screen: the window must still fit it


def test_fill_maximizes_and_keeps_the_preferred_size_for_later(qapp):
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget
    from gui.gui_utils import fit_window
    fitted, filled = QWidget(), QWidget()
    try:
        fit_window(fitted, QSize(300, 200))
        fit_window(filled, QSize(300, 200), fill=True)
        qapp.processEvents()
        assert not fitted.isMaximized() and fitted.size() == QSize(300, 200)
        assert filled.isMaximized()
        assert filled.normalGeometry().size() == QSize(300, 200)  # un-maximized from its title bar
    finally:
        fitted.close()
        filled.close()


@pytest.mark.models
@pytest.mark.slow
def test_a_task_window_fills_the_screen_and_fits_a_small_one(tmp_path, qapp, cpu_networks):
    from gui.main_controller import MainController
    task = make_task(str(tmp_path))
    controller = MainController(compose_gui_config(task.workspace), networks=cpu_networks)
    gui = controller.gui
    try:
        qapp.processEvents()
        assert gui.isMaximized()
        smallest = gui.minimumSizeHint().expandedTo(gui.minimumSize())
        assert smallest.width() <= SMALLEST_DESKTOP[0] and \
            smallest.height() <= SMALLEST_DESKTOP[1], smallest
    finally:
        gui.close()
