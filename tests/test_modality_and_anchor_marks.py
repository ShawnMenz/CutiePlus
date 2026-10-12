"""
The labeling window propagates on the modality it shows (T switches it), and the frame slider
marks the anchors with small hollow rings: black (white in a dark theme) per mask anchor, magenta
per empty anchor. The shared service's side of the modality is in test_service.py.
"""
import pytest

from tests.conftest import compose_gui_config
from tests.helpers import make_task

pytestmark = [pytest.mark.models, pytest.mark.slow]


def open_window(task, cpu_networks):
    from gui.main_controller import MainController
    return MainController(compose_gui_config(task.workspace), networks=cpu_networks)


def test_propagation_reads_the_frames_of_the_shown_modality(tmp_path, qapp, cpu_networks):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, ))  # anchor modality: visual
    controller = open_window(task, cpu_networks)
    read = []
    original = controller.res_man._get_image_unbuffered

    def recording(ti, modality=None):
        read.append(modality)
        return original(ti, modality)

    controller.res_man._get_image_unbuffered = recording
    try:
        controller.on_forward_propagation()  # as opened: RGB
        assert set(read) == {'visual'}
        assert [ti for ti, name in enumerate(task.names) if task.mask(name) is not None] == \
            list(range(8))

        controller.gui.tl_slider.setValue(0)
        controller.on_toggle_modality()  # T: thermal
        read.clear()
        controller.on_forward_propagation()
        assert set(read) == {'thermal'}
        assert 'Propagating on the thermal frames.' in controller.gui.console.toPlainText()
    finally:
        controller.gui.close()


def test_the_slider_marks_the_anchors(tmp_path, qapp, cpu_networks):
    from PySide6.QtGui import QColor
    from gui.gui_utils import STATUS_COLORS, FrameSlider, is_dark
    task = make_task(str(tmp_path), frame_count=12, mask_anchors=(0, ), empty_anchors=(9, ))
    controller = open_window(task, cpu_networks)
    slider = controller.gui.tl_slider
    try:
        controller.gui.resize(1400, 900)
        controller.gui.show()
        qapp.processEvents()
        assert (slider.mask_anchors, slider.empty_anchors) == ((0, ), (9, ))

        controller.on_forward_propagation()  # frames 1-8 get masks; anchors do not change
        assert (slider.mask_anchors, slider.empty_anchors) == ((0, ), (9, ))
        slider.setValue(4)
        controller.on_commit()  # C: frame 4 becomes a mask anchor
        assert slider.mask_anchors == (0, 4)
        controller.on_remove_anchor()  # Delete
        assert slider.mask_anchors == (0, )

        # hollow rings smaller than the handle, in the colors of the theme
        slider.setValue(6)
        qapp.processEvents()
        image = slider.grab().toImage()
        centre, _, _, _, handle = slider.frame_geometry()
        assert FrameSlider.ANCHOR_DIAMETER < min(handle.width(), handle.height())
        dark = int(is_dark(slider))

        def near(pixel, color):  # an antialiased outline: close to the color
            target = QColor(color)
            return max(abs(pixel.red() - target.red()), abs(pixel.green() - target.green()),
                       abs(pixel.blue() - target.blue())) < 60

        for ti, color in ((0, FrameSlider.MASK_ANCHOR_COLORS[dark]),
                          (9, STATUS_COLORS['empty anchor'][dark])):
            ring = slider.anchor_ring(ti, centre)
            x, top = int(ring.center().x()), round(ring.top())
            outline = [image.pixelColor(x, y) for y in (top - 1, top, top + 1)]
            assert any(near(pixel, color) for pixel in outline), (ti, [p.name() for p in outline])
            inside = image.pixelColor(x, int(ring.center().y()))
            assert not near(inside, color), (ti, inside.name())  # hollow
    finally:
        controller.gui.close()
