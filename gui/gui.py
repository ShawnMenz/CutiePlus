import functools
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig

from PySide6.QtWidgets import (QWidget, QComboBox, QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSpinBox, QPlainTextEdit, QVBoxLayout, QSizePolicy,
                               QButtonGroup, QSlider, QRadioButton, QApplication, QFileDialog,
                               QMessageBox, QFrame, QLineEdit, QScrollArea, QSplitter, QStyle)

from PySide6.QtGui import (QKeySequence, QShortcut, QTextCursor, QImage, QPixmap, QIcon, QPainter,
                           QColor)
from PySide6.QtCore import Qt, QEvent, QTimer, QPointF, QRectF, QSize

from gui.flow_layout import FlowLayout
from gui.gui_utils import *
from gui.task_state import describe_attention, interval_index_from
from gui.tips_window import TipsWindow
from gui.interactive_utils import (CLICK_DOT_DIAMETER, NEGATIVE_CLICK_COLOR,
                                   POSITIVE_CLICK_COLOR, color_map_np)

# Need Attention: orange text, filled orange while the sequence is flagged
ATTENTION_COLOR = '#ef6c00'
ATTENTION_STYLE = (f'QPushButton {{ color: {ATTENTION_COLOR}; font-weight: bold; }}'
                   f'QPushButton:checked {{ color: white; background: {ATTENTION_COLOR}; }}')

# canvas view: Ctrl + wheel zooms at the cursor, the wheel scrolls (with Alt: sideways)
ZOOM_STEP = 1.25  # per wheel notch
MAX_ZOOM = 32  # times the size that fits the canvas
SCROLL_STEP = 0.1  # of the canvas per wheel notch
WHEEL_NOTCH = 120  # angleDelta units per wheel notch
CONSOLE_MIN_HEIGHT = 60  # px: a few lines of the log, however short the window

# propagation button labels, with their shortcut keys; they change while a run is going
FORWARD_LABEL = 'Propagate forward (F)'
BACKWARD_LABEL = 'Propagate backward (B)'
PAUSE_FORWARD_LABEL = 'Pause propagation (F)'
PAUSE_BACKWARD_LABEL = 'Pause propagation (B)'
PLAY_LABEL = 'Play video (R)'
STOP_LABEL = 'Stop video (R)'


class GUI(QWidget):
    # what the frame label shows now, to draw it again when the theme changes
    frame_label_args = None

    def __init__(self, controller, cfg: DictConfig) -> None:
        super().__init__()

        # callbacks to be set by the controller
        self.on_mouse_motion_xy = None
        self.click_fn = None

        self.controller = controller
        self.cfg = cfg
        self.h = controller.h
        self.w = controller.w
        self.T = controller.T

        # set up the window
        task = controller.task
        if task is not None:
            self.setWindowTitle(f'Cutie task {task.sequence_id} [{task.split}] '
                                f'anchor modality: {task.anchor_modality} · {task.labeler}')
        else:
            self.setWindowTitle(f'Cutie demo: {cfg["workspace"]}')
        self.setWindowIcon(QIcon('docs/icon.png'))

        # set up some buttons
        self.play_button = QPushButton(PLAY_LABEL)
        self.play_button.clicked.connect(self.on_play_video)
        self.play_button.setToolTip(
            'R: play the video from here to the last frame; again to stop.'
            + ('\nShift+R or Shift+click: preview the intervals, one per press.'
               '\n{ / }: go to the previous / next interval start.'
               if controller.task is not None else ''))
        self.commit_button = QPushButton('Commit to permanent memory (C)')
        self.commit_button.clicked.connect(controller.on_commit)
        self.export_task_button = QPushButton('Export task masks')
        self.export_task_button.clicked.connect(lambda: controller.on_export())
        self.export_video_button = QPushButton('Export as video (Ctrl+E)')
        self.export_video_button.clicked.connect(controller.on_export_visualization)
        self.export_binary_button = QPushButton('Export binary masks (Ctrl+S)')
        self.export_binary_button.clicked.connect(controller.on_export_binary)
        if task is not None:
            # these only write files into the task folder; the dataset comes from the task export
            self.export_video_button.setText('Save overlay video (Ctrl+E)')
            self.export_video_button.setToolTip(
                'A video from the saved visualizations. Not the dataset: that comes from Export '
                'task masks / Mark done.')
            self.export_binary_button.setText('Save binary PNGs (Ctrl+S)')
            self.export_binary_button.setToolTip(
                'Writes binary PNGs of the task masks into binary_masks/ and submits the task '
                'for review; the task browser\'s Mark done then accepts it. Not the dataset: that '
                'comes from Export task masks / Mark done.')
            # a toggle: the sequence needs further attention (orange; filled while on)
            self.attention_button = QPushButton('Need Attention')
            self.attention_button.setCheckable(True)
            self.attention_button.setStyleSheet(ATTENTION_STYLE)
            self.attention_button.clicked.connect(controller.on_toggle_attention)
            self.refresh_attention_button()

        self.forward_run_button = QPushButton(FORWARD_LABEL)
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)
        self.fit_to_labels(self.forward_run_button, (FORWARD_LABEL, PAUSE_FORWARD_LABEL))

        self.backward_run_button = QPushButton(BACKWARD_LABEL)
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)
        self.fit_to_labels(self.backward_run_button, (BACKWARD_LABEL, PAUSE_BACKWARD_LABEL))

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(200)

        self.reset_frame_button = QPushButton('Reset frame')
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_object_button = QPushButton('Reset object')
        self.reset_object_button.clicked.connect(controller.on_reset_object)
        self.reset_all_masks_button = QPushButton('Reset all frame masks')
        self.reset_all_masks_button.setStyleSheet('QPushButton { color: #d32f2f; font-weight: bold; }')
        self.reset_all_masks_button.clicked.connect(controller.on_reset_all_frame_masks)

        # set up the LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(150)
        # 1-based frame numbers, like the frame names and the interval boxes
        self.lcd.setText('{: 5d} / {: 5d}'.format(1, controller.T))

        # current object id
        self.object_dial = QSpinBox()
        self.object_dial.setReadOnly(False)
        self.object_dial.setMinimumSize(50, 30)
        self.object_dial.setMinimum(1)
        self.object_dial.setMaximum(controller.num_objects)
        self.object_dial.editingFinished.connect(controller.on_object_dial_change)

        self.object_color = QLabel()
        self.object_color.setMinimumSize(100, 30)
        self.object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_name = QLabel()
        self.frame_name.setMinimumSize(100, 30)
        # centered vertically, like the boxes beside it in the top row
        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.frame_name.setTextFormat(Qt.TextFormat.RichText)

        # the target of a task: its category, and the referring text as a box to tune it
        self.task_label = QLabel()
        self.description_edit = QLineEdit()
        if task is not None:
            self.task_label.setText(f'<b>{task.category}</b>:')
            self.set_description(task.description)
            self.description_edit.editingFinished.connect(controller.on_description_change)
            self.commit_button.setText('Mark as anchor (C)')

            # the dataset clips, 1-indexed and inclusive, each interval a clip of its own: one pair
            # as preset (the saved or curated interval), more only with '+ Add interval'
            self.interval_pairs: List[IntervalPair] = []
            self.interval_flow = FlowLayout(spacing=10, spread=False)
            intervals_label = QLabel('Intervals')
            intervals_label.setToolTip('{ / }: go to the previous / next interval start; '
                                       'Shift+R: play the intervals, one per press.')
            self.interval_flow.addWidget(intervals_label)
            self.add_interval_button = QPushButton('+ Add interval')
            self.add_interval_button.setToolTip(
                'Keep another stretch of this sequence: each interval becomes a clip of its own. '
                'Intervals may touch but not overlap.')
            self.add_interval_button.clicked.connect(self.on_add_interval)
            self.interval_flow.addWidget(self.add_interval_button)
            intervals, _ = task.intervals(controller.T)
            for start, end in intervals:
                self.add_interval_pair(start, end)
            # the first pair, as the single interval always was
            self.interval_start, self.interval_end = self.interval_pairs[0].boxes()

        # timeline slider
        self.tl_slider = FrameSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(controller.on_slider_update)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(controller.T - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)
        # playback stops on this frame: the last one, or the end of the interval being previewed
        self.play_until = controller.T - 1
        self.preview_index = 0  # the interval Shift+R plays next
        if task is not None:
            self.tl_slider.set_intervals(task.intervals(controller.T)[0])

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("mask")
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("layer")
        self.combo.addItem("rgba")
        self.combo.setCurrentText('davis')
        self.combo.currentTextChanged.connect(controller.set_vis_mode)

        self.save_visualization_combo = QComboBox(self)
        self.save_visualization_combo.addItem("None")
        self.save_visualization_combo.addItem("Always")
        self.save_visualization_combo.addItem("Propagation only (higher quality)")
        self.combo.setCurrentText('None')
        self.save_visualization_combo.currentTextChanged.connect(
            controller.on_set_save_visualization_mode)

        self.save_soft_mask_checkbox = QCheckBox(self)
        self.save_soft_mask_checkbox.toggled.connect(controller.on_save_soft_mask_toggle)
        self.save_soft_mask_checkbox.setChecked(False)

        # controls for output FPS and bitrate
        self.fps_dial = QSpinBox()
        self.fps_dial.setReadOnly(False)
        self.fps_dial.setMinimumSize(40, 30)
        self.fps_dial.setMinimum(1)
        self.fps_dial.setMaximum(60)
        self.fps_dial.setValue(cfg['output_fps'])
        self.fps_dial.editingFinished.connect(controller.on_fps_dial_change)

        self.bitrate_dial = QSpinBox()
        self.bitrate_dial.setReadOnly(False)
        self.bitrate_dial.setMinimumSize(40, 30)
        self.bitrate_dial.setMinimum(1)
        self.bitrate_dial.setMaximum(100)
        self.bitrate_dial.setValue(cfg['output_bitrate'])
        self.bitrate_dial.editingFinished.connect(controller.on_bitrate_dial_change)

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release
        self.main_canvas.wheelEvent = self.on_wheel
        self.main_canvas.resizeEvent = self.on_canvas_resize

        # the view: zoom relative to the fitted image, and the image point at the canvas centre
        # (None: the image centre); it persists across frames
        self.zoom = 1.0
        self.view_center = None
        self.canvas_image = None
        self.canvas_clicks = ()

        # clearing memory
        self.clear_all_mem_button = QPushButton('Reset all memory')
        self.clear_all_mem_button.clicked.connect(controller.on_clear_memory)
        self.clear_non_perm_mem_button = QPushButton('Reset non-permanent memory')
        self.clear_non_perm_mem_button.clicked.connect(controller.on_clear_non_permanent_memory)

        # displaying memory usage
        self.perm_mem_gauge, self.perm_mem_gauge_layout = create_gauge('Permanent memory size')
        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            'GPU mem. (all proc, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            'GPU mem. (torch, w/o caching)')

        # Parameters setting
        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(
            1, 100, 'Min. working memory frames', callback=controller.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(
            2, 100, 'Max. working memory frames', callback=controller.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(
            1000,
            100000,
            'Max. long-term memory size',
            step=1000,
            callback=controller.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, 'Memory frame every (r)', callback=controller.update_config)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(controller.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(controller.on_import_layer)

        # Console on the GUI
        self.console = LogView()
        self.console.setReadOnly(True)
        # it takes the height the right panel has left: tall in a tall window, a few lines in a short one
        self.console.setMinimumHeight(CONSOLE_MIN_HEIGHT)
        self.console.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Tips for the users: the guide opens in a window of its own
        self.tips_window = None
        self.tips_button = QPushButton('Tips (F1)')
        self.tips_button.setToolTip('The labeling guide: keys, mouse, and what each control does.')
        self.tips_button.clicked.connect(self.show_tips)

        # navigator: three groups in a row, which wrap onto more rows in a narrow window
        navi = FlowLayout()

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interact_topbox.addWidget(self.lcd)
        interact_topbox.addWidget(self.play_button)
        interact_topbox.addWidget(self.reset_frame_button)
        interact_topbox.addWidget(self.reset_object_button)
        interact_botbox.addWidget(QLabel('Current object ID:'))
        interact_botbox.addWidget(self.object_dial)
        interact_botbox.addWidget(self.object_color)
        if task is None:
            interact_botbox.addWidget(self.frame_name)
        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        interact_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        navi.addWidget(layout_widget(interact_subbox))

        apply_fixed_size_policy = lambda x: x.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.
                                                            Policy.Fixed)
        apply_to_all_children_widget(interact_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(interact_botbox, apply_fixed_size_policy)

        overlay_subbox = QVBoxLayout()
        overlay_topbox = QHBoxLayout()
        overlay_botbox = QHBoxLayout()
        overlay_topbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_topbox.addWidget(QLabel('Visualization mode'))
        overlay_topbox.addWidget(self.combo)
        overlay_topbox.addWidget(QLabel('Save soft mask during propagation'))
        overlay_topbox.addWidget(self.save_soft_mask_checkbox)
        overlay_topbox.addWidget(self.export_binary_button)
        if task is not None:
            overlay_topbox.addWidget(self.attention_button)
        overlay_botbox.addWidget(QLabel('Save visualization'))
        overlay_botbox.addWidget(self.save_visualization_combo)
        overlay_botbox.addWidget(self.export_video_button)
        overlay_botbox.addWidget(QLabel('Output FPS: '))
        overlay_botbox.addWidget(self.fps_dial)
        overlay_botbox.addWidget(QLabel('Output bitrate (Mbps): '))
        overlay_botbox.addWidget(self.bitrate_dial)
        overlay_subbox.addLayout(overlay_topbox)
        overlay_subbox.addLayout(overlay_botbox)
        navi.addWidget(layout_widget(overlay_subbox))
        apply_to_all_children_widget(overlay_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(overlay_botbox, apply_fixed_size_policy)

        control_subbox = QVBoxLayout()
        control_topbox = QHBoxLayout()
        control_botbox = QHBoxLayout()
        control_topbox.addWidget(self.commit_button)
        if task is not None:
            control_topbox.addWidget(self.export_task_button)
        control_topbox.addWidget(self.forward_run_button)
        control_topbox.addWidget(self.backward_run_button)
        control_botbox.addWidget(self.progressbar)
        control_subbox.addLayout(control_topbox)
        control_subbox.addLayout(control_botbox)
        navi.addWidget(layout_widget(control_subbox))

        # right area
        right_area = QVBoxLayout()
        right_area.addWidget(self.tips_button)

        # Parameters
        right_area.addLayout(self.perm_mem_gauge_layout)
        right_area.addLayout(self.work_mem_gauge_layout)
        right_area.addLayout(self.long_mem_gauge_layout)
        right_area.addLayout(self.gpu_mem_gauge_layout)
        right_area.addLayout(self.torch_mem_gauge_layout)
        right_area.addWidget(self.clear_all_mem_button)
        right_area.addWidget(self.clear_non_perm_mem_button)
        right_area.addWidget(self.reset_all_masks_button)
        right_area.addLayout(self.work_mem_min_layout)
        right_area.addLayout(self.work_mem_max_layout)
        right_area.addLayout(self.long_mem_max_layout)
        right_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        right_area.addLayout(import_area)

        # console: all the height that is left in the panel
        right_area.addWidget(self.console, 1)

        # the right panel scrolls when the window is short; drag its edge to resize it, or to the
        # window's edge to hide it
        right_panel = QScrollArea()
        right_panel.setWidget(layout_widget(right_area))
        right_panel.setWidgetResizable(True)
        right_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_panel.setFrameShape(QFrame.Shape.NoFrame)
        # wide enough for the content beside the scroll bar; it scrolls up-down only (polished
        # first: the theme's style sheet sets the frame)
        right_panel.ensurePolished()
        right_panel.setMinimumWidth(
            right_panel.widget().minimumSizeHint().width() +
            right_panel.verticalScrollBar().sizeHint().width() + 2 * right_panel.frameWidth() +
            right_panel.style().pixelMetric(QStyle.PixelMetric.PM_ScrollView_ScrollBarSpacing,
                                            None, right_panel))

        # Drawing area: main canvas and the right panel
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.main_canvas)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([4000, 1000])  # relative: the panel starts at a fifth of the width
        self.splitter.setCollapsible(0, False)

        layout = QVBoxLayout()
        if task is not None:
            task_row = QHBoxLayout()
            task_row.addWidget(self.task_label)
            task_row.addWidget(self.description_edit, 1)
            # the frame label carries the mask source; the bottom row is too narrow for it
            task_row.addWidget(self.frame_name)
            layout.addLayout(task_row)
            # the intervals in a row of their own, which wraps when many are added
            layout.addWidget(layout_widget(self.interval_flow))
        layout.addWidget(self.splitter, 1)
        # the frame slider between two boxes naming the keys that step it
        slider_row = QHBoxLayout()
        slider_row.addWidget(key_box('<b>A</b>  previous frame',
                                     frame_key_tip('A', 'previous frame', '10 frames back',
                                                   'first frame', 'previous anchor')))
        slider_row.addWidget(self.tl_slider, 1)
        slider_row.addWidget(key_box('next frame  <b>D</b>',
                                     frame_key_tip('D', 'next frame', '10 frames forward',
                                                   'last frame', 'next anchor')))
        layout.addLayout(slider_row)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(controller.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer(self)
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(controller.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # Objects shortcuts
        for i in range(1, controller.num_objects + 1):
            QShortcut(QKeySequence(str(i)),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))
            QShortcut(QKeySequence(f"Ctrl+{i}"),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))

        # next/prev frame shortcuts: A / D (the arrow keys do not move frames)
        QShortcut(QKeySequence('A'), self).activated.connect(controller.on_prev_frame)
        QShortcut(QKeySequence('D'), self).activated.connect(controller.on_next_frame)

        # +/- 10 frames shortcuts
        QShortcut(QKeySequence('Shift+A'),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 10))
        QShortcut(QKeySequence('Shift+D'),
                    self).activated.connect(functools.partial(controller.on_next_frame, 10))

        # first/last frame shortcuts
        QShortcut(QKeySequence('Alt+A'),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 999999))
        QShortcut(QKeySequence('Alt+D'),
                    self).activated.connect(functools.partial(controller.on_next_frame, 999999))

        # commit to permanent memory shortcut (task mode: mark the frame as an anchor)
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(controller.on_commit)

        # task mode: remove an anchor, jump between anchors ([ ] and Ctrl+A / D), switch visual/thermal
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self).activated.connect(controller.on_remove_anchor)
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self).activated.connect(controller.on_next_anchor)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self).activated.connect(controller.on_prev_anchor)
        QShortcut(QKeySequence('Ctrl+D'), self).activated.connect(controller.on_next_anchor)
        QShortcut(QKeySequence('Ctrl+A'), self).activated.connect(controller.on_prev_anchor)
        # { / } (Shift + [ / ]): the previous / next interval start
        QShortcut(QKeySequence(Qt.Key.Key_BraceRight),
                  self).activated.connect(functools.partial(self.on_jump_interval, 1))
        QShortcut(QKeySequence(Qt.Key.Key_BraceLeft),
                  self).activated.connect(functools.partial(self.on_jump_interval, -1))
        QShortcut(QKeySequence(Qt.Key.Key_T), self).activated.connect(controller.on_toggle_modality)

        # undo/redo of clicks and frame edits
        QShortcut(QKeySequence('Ctrl+Z'), self).activated.connect(controller.on_undo)
        for keys in ('Ctrl+Shift+Z', 'Ctrl+Y'):
            QShortcut(QKeySequence(keys), self).activated.connect(controller.on_redo)

        # propagate forward/backward/pause shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(controller.on_forward_propagation)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(controller.cycle_visualization_mode)
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(controller.on_backward_propagation)

        # play / stop the video
        QShortcut(QKeySequence(Qt.Key.Key_R), self).activated.connect(self.on_play_video)
        # preview the intervals, one per press
        QShortcut(QKeySequence('Shift+R'), self).activated.connect(self.on_preview_interval)

        # save binary PNGs / overlay video (the keys are on the buttons)
        QShortcut(QKeySequence('Ctrl+S'), self).activated.connect(controller.on_export_binary)
        QShortcut(QKeySequence('Ctrl+E'), self).activated.connect(controller.on_export_visualization)

        # quit shortcut
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key.Key_F1), self).activated.connect(self.show_tips)
        self.refresh_reset_button_state()

        # a text or number box with the focus takes the arrow and letter keys for itself; the
        # frame slider holds the focus instead, so they reach the shortcuts: from the start, when
        # the image is clicked, and after Enter in a box
        # (the interval boxes are connected when their pair is made, as pairs are added later too)
        for box in self.findChildren(QSpinBox) + [self.description_edit]:
            if not isinstance(box, IntervalBox):
                box.editingFinished.connect(functools.partial(self.return_focus_from, box))
        self.tl_slider.setFocus()


    @staticmethod
    def fit_to_labels(button: QPushButton, labels) -> None:
        # wide enough for the longest label the button shows, so the text never clips
        button.ensurePolished()
        shown = button.text()
        width = 0
        for label in labels:
            button.setText(label)
            width = max(width, button.sizeHint().width())
        button.setText(shown)
        button.setMinimumWidth(width)

    def show_tips(self) -> None:
        # one guide window per labeling window: pressing Tips again brings it forward
        if self.tips_window is None:
            self.tips_window = TipsWindow(self)
        self.tips_window.show_raised()

    def set_frame_label(self, file_name: str, modality=None, source=None) -> None:
        # the frame's file, the modality shown and where its mask comes from, in color
        self.frame_label_args = (file_name, modality, source)
        dark = is_dark(self)
        self.frame_name.setText(frame_label_html(file_name, modality, source, dark))
        if modality is not None:
            self.frame_name.setToolTip(frame_label_legend(dark))

    def changeEvent(self, event) -> None:
        # a new theme (the browser's theme button, or Auto following the desktop) changes the
        # palette; the label's colors depend on it
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and self.frame_label_args is not None:
            self.set_frame_label(*self.frame_label_args)

    def show_fitted(self) -> None:
        # the controls in one row and at least the image plus 200 px; maximized instead when that
        # does not fit the screen
        fit_window(self, self.sizeHint().expandedTo(QSize(self.w + 200, self.h + 200)))

    def add_interval_pair(self, start: Optional[int] = None,
                          end: Optional[int] = None) -> IntervalPair:
        # every pair but the first can be removed; a pair without values starts blank
        first = not self.interval_pairs
        pair = IntervalPair(self.controller.T, start, end, removable=not first)
        for box in pair.boxes():
            which = box.which
            box.setToolTip(f'Interval {which} in frame numbers; type m for the '
                           f'{"first" if which == "start" else "last"} frame. Enter or click '
                           'elsewhere to save; each interval is a clip of its own.')
            if first:  # the controller's handler for the single interval, unchanged
                box.editingFinished.connect(
                    functools.partial(self.controller.on_interval_change, which))
            else:
                box.editingFinished.connect(
                    functools.partial(self.controller.on_interval_change, which, pair))
            box.editingFinished.connect(functools.partial(self.return_focus_from, box))
        pair.removed.connect(self.controller.on_remove_interval)
        self.interval_pairs.append(pair)
        # before the '+ Add interval' button, which stays last
        self.interval_flow.insert_widget(self.interval_flow.count() - 1, pair)
        return pair

    def on_add_interval(self) -> None:
        pair = self.add_interval_pair()
        pair.start_box.setFocus()
        self.text('Added an interval: type its start and end (m = first / last frame). '
                  '× cancels it.')

    def remove_interval_pair(self, pair: IntervalPair) -> None:
        self.interval_pairs.remove(pair)
        self.interval_flow.removeWidget(pair)
        pair.hide()
        pair.deleteLater()
        self.tl_slider.setFocus()

    def return_focus_from(self, box: QWidget) -> None:
        # Enter; a box that lost the focus to another widget leaves it there
        if box.hasFocus():
            self.tl_slider.setFocus()

    def set_description(self, text: str) -> None:
        self.description_edit.setText(text)
        self.description_edit.setCursorPosition(0)  # show the start of a long text
        self.description_edit.setToolTip(f'{text}\n\nEnter or click elsewhere to save; a change '
                                         'needs a new review.')

    def on_canvas_resize(self, event) -> None:
        # window resizes and divider drags; the view keeps its zoom and centre
        if self.canvas_image is not None:
            self.render_canvas()

    def closeEvent(self, event):
        if self.controller.export_process is not None:
            # closing destroys the export process, which kills it halfway through the tree
            self.text('Wait for the task export to finish before closing.')
            event.ignore()
            return
        if self.tips_window is not None:
            self.tips_window.close()
        # the timers hold the controller; stop them so a closed task can be freed
        self.timer.stop()
        self.gpu_timer.stop()
        self.controller.on_window_closed()
        event.accept()

    def refresh_reset_button_state(self) -> None:
        reason = self.controller.reset_block_reason()
        self.reset_all_masks_button.setEnabled(reason is None)
        self.reset_all_masks_button.setToolTip(
            reason or 'Delete this sequence\'s local frame masks and generated mask files.')

    def confirm_reset_all_masks(self) -> bool:
        task = self.controller.task
        sequence = task.sequence_id if task is not None else self.cfg['workspace']
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle('Reset all frame masks')
        dialog.setText(f'Delete all frame masks for {sequence}?')
        if task is not None:
            dialog.setInformativeText(
                'This deletes local, binary, and soft masks, clears task anchors and prefills, '
                'and deletes this sequence\'s Cutie exports. Original frames and external '
                'Label Studio/SAM3 masks remain. This cannot be undone.')
        else:
            dialog.setInformativeText(
                'This deletes local, binary, and soft masks in this workspace. '
                'Original frames remain. This cannot be undone.')
        delete_button = dialog.addButton('Delete all masks', QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = dialog.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is delete_button

    def text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text + '\n')

    def set_canvas(self, image, clicks=()):
        """clicks: (x, y, is_negative) in image pixels, painted as dots after scaling."""
        height, width, channel = image.shape
        # if the image is RGBA, convert to RGB first by coloring the background green
        if channel == 4:
            image_rgb = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255
            green_bg = np.array([0, 255, 0])
            # soft blending
            image = (image_rgb * alpha[:, :, np.newaxis] + green_bg[np.newaxis, np.newaxis, :] *
                     (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

        # kept for redrawing on zoom and scroll; QImage borrows this buffer
        self.canvas_image = np.ascontiguousarray(image)
        self.canvas_clicks = clicks
        self.image_size = QSize(width, height)
        self.render_canvas()

    def render_canvas(self) -> None:
        height, width = self.canvas_image.shape[:2]
        image = QImage(self.canvas_image.data, width, height, 3 * width,
                       QImage.Format.Format_RGB888)
        self.clamp_view()
        scale, offset_x, offset_y = self.view_transform()

        pixmap = QPixmap(self.main_canvas.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        # no smooth transform: nearest neighbour, so zoomed pixels stay sharp
        painter.drawImage(QRectF(offset_x, offset_y, width * scale, height * scale), image)
        if self.canvas_clicks:
            self.draw_clicks(painter, self.canvas_clicks, scale, offset_x, offset_y)
        painter.end()
        self.main_canvas.setPixmap(pixmap)

    def draw_clicks(self, painter: QPainter, clicks, scale: float, offset_x: float,
                    offset_y: float) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = CLICK_DOT_DIAMETER / 2
        for x, y, is_negative in clicks:
            color = NEGATIVE_CLICK_COLOR if is_negative else POSITIVE_CLICK_COLOR
            painter.setBrush(QColor(*color))
            # the centre of the clicked image pixel, in canvas pixels
            painter.drawEllipse(QPointF((x + 0.5) * scale + offset_x, (y + 0.5) * scale + offset_y),
                                radius, radius)

    def fit_scale(self) -> float:
        return min(self.main_canvas.width() / self.image_size.width(),
                   self.main_canvas.height() / self.image_size.height())

    def view_transform(self):
        """(scale, offset_x, offset_y) with canvas position = image position * scale + offset."""
        scale = self.fit_scale() * self.zoom
        center_x, center_y = self.view_center
        return (scale, self.main_canvas.width() / 2 - center_x * scale,
                self.main_canvas.height() / 2 - center_y * scale)

    def clamp_view(self) -> None:
        # along each axis: centre an image that fits the canvas, otherwise leave no empty margin
        scale = self.fit_scale() * self.zoom
        image = (self.image_size.width(), self.image_size.height())
        canvas = (self.main_canvas.width(), self.main_canvas.height())
        center = self.view_center or (image[0] / 2, image[1] / 2)
        clamped = []
        for c, image_extent, canvas_extent in zip(center, image, canvas):
            half_view = canvas_extent / 2 / scale  # in image pixels
            if 2 * half_view >= image_extent:
                clamped.append(image_extent / 2)
            else:
                clamped.append(min(max(c, half_view), image_extent - half_view))
        self.view_center = tuple(clamped)

    def on_wheel(self, event):
        if self.canvas_image is None:
            return
        dx, dy = event.angleDelta().x(), event.angleDelta().y()
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.zoom_at(event.position(), ZOOM_STEP**((dy or dx) / WHEEL_NOTCH))
        else:
            if modifiers & Qt.KeyboardModifier.AltModifier and dx == 0:
                # Qt on X11 already sends Alt + wheel as a horizontal wheel; elsewhere turn it
                dx, dy = dy, 0
            self.scroll_view(dx / WHEEL_NOTCH, dy / WHEEL_NOTCH)
        event.accept()

    def zoom_at(self, position: QPointF, factor: float) -> None:
        # the image point under the cursor stays under it
        image_x, image_y = self.pixel_pos_to_image_pos(position.x(), position.y())
        self.zoom = min(max(self.zoom * factor, 1.0), MAX_ZOOM)
        scale = self.fit_scale() * self.zoom
        self.view_center = (image_x - (position.x() - self.main_canvas.width() / 2) / scale,
                            image_y - (position.y() - self.main_canvas.height() / 2) / scale)
        self.render_canvas()

    def scroll_view(self, notches_x: float, notches_y: float) -> None:
        # wheel up / left moves the view up / left
        scale = self.fit_scale() * self.zoom
        center_x, center_y = self.view_center
        self.view_center = (center_x - notches_x * SCROLL_STEP * self.main_canvas.width() / scale,
                            center_y - notches_y * SCROLL_STEP * self.main_canvas.height() / scale)
        self.render_canvas()

    def update_slider(self, value):
        self.lcd.setText('{: 3d} / {: 3d}'.format(value + 1, self.controller.T))
        self.tl_slider.setValue(value)

    def pixel_pos_to_image_pos(self, x, y):
        # canvas coordinates into (continuous) image coordinates, through the zoomed view
        scale, offset_x, offset_y = self.view_transform()
        return (x - offset_x) / scale, (y - offset_y) / scale

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = ((x < 0) or (y < 0) or (x >= self.w) or (y >= self.h))

        return out_of_bound

    def get_scaled_pos(self, x, y):
        # the pixel under the canvas position: clicks are pixel indices
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = int(max(0, min(self.w - 1, x)))
        y = int(max(0, min(self.h - 1, y)))

        return x, y

    def forward_propagation_start(self):
        self.backward_run_button.setEnabled(False)
        self.forward_run_button.setText(PAUSE_FORWARD_LABEL)

    def backward_propagation_start(self):
        self.forward_run_button.setEnabled(False)
        self.backward_run_button.setText(PAUSE_BACKWARD_LABEL)

    def pause_propagation(self):
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_all_mem_button.setEnabled(True)
        self.clear_non_perm_mem_button.setEnabled(True)
        self.forward_run_button.setText(FORWARD_LABEL)
        self.backward_run_button.setText(BACKWARD_LABEL)
        self.tl_slider.setEnabled(True)

    def process_events(self):
        QApplication.processEvents()

    def on_mouse_press(self, event):
        self.tl_slider.setFocus()  # e.g. from the description box: the keys move frames again
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            return

        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        if event.button() == Qt.MouseButton.LeftButton:
            action = 'left'
        else:
            action = 'right'

        self.click_fn(action, ex, ey)

    def on_mouse_motion(self, event):
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        self.on_mouse_motion_xy(ex, ey)

    def on_mouse_release(self, event):
        pass

    def on_play_video(self):
        # the button and the R key; the video plays once and stops on its last frame (Shift+click
        # on the button previews the intervals instead, as Shift+R)
        if (self.sender() is self.play_button
                and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.on_preview_interval()
            return
        if self.timer.isActive():
            self.stop_playback()
        elif not self.controller.propagating:
            if self.tl_slider.value() >= self.tl_slider.maximum():
                self.tl_slider.setValue(0)  # played to the end before: play it again from the start
            self.play_until = self.tl_slider.maximum()
            self.timer.start(1000 // 30)
            self.play_button.setText(STOP_LABEL)

    def on_preview_interval(self) -> None:
        # Shift+R: the next interval only, from its start to its end; each press plays the next one
        # (after the last, the first again), also while one is playing
        task = self.controller.task
        if task is None:
            self.text('Shift+R previews the intervals of a task; R plays the video.')
            return
        if self.controller.propagating:
            return
        intervals, _ = task.intervals(self.controller.T)
        number = self.preview_index % len(intervals)
        start, end = intervals[number]
        self.preview_index = number + 1
        self.timer.stop()
        self.tl_slider.setValue(start - 1)
        self.play_until = end - 1
        self.tl_slider.set_active_interval((start, end))
        self.timer.start(1000 // 30)
        self.play_button.setText(STOP_LABEL)
        self.text(f'Previewing interval {number + 1} of {len(intervals)}: frames {start}-{end} '
                  '(Shift+R: next).')

    def on_jump_interval(self, direction: int) -> None:
        # { / }: the start of the previous / next interval, from the current frame and around the
        # ends; Shift+R then plays the interval jumped to
        task = self.controller.task
        if task is None:
            self.text('{ and } move between the intervals of a task.')
            return
        if self.controller.propagating:
            return
        intervals, _ = task.intervals(self.controller.T)
        number = interval_index_from(intervals, self.tl_slider.value() + 1, direction)
        start, end = intervals[number]
        self.stop_playback()
        self.tl_slider.setValue(start - 1)
        self.preview_index = number
        self.text(f'Interval {number + 1} of {len(intervals)}: frames {start}-{end} '
                  '(Shift+R plays it).')

    def refresh_attention_button(self) -> None:
        # checked while the task needs attention; who flagged it, in the tooltip
        attention = self.controller.task.attention
        self.attention_button.setChecked(attention is not None)
        tip = ('This sequence needs further attention: the task browser shows it as Attention, '
               'and Mark done waits until it is off. Click to toggle.')
        if attention is not None:
            tip += f'\nOn: {describe_attention(attention)}.'
        self.attention_button.setToolTip(tip)

    def stop_playback(self) -> None:
        self.timer.stop()
        self.play_button.setText(PLAY_LABEL)
        self.tl_slider.set_active_interval(None)

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   prompt,
                                                   "",
                                                   "Image files (*)",
                                                   options=options)
        return file_name

    def set_object_color(self, object_id: int):
        r, g, b = color_map_np[object_id]  # the colour the mask is drawn in
        rgb = f'rgb({r},{g},{b})'
        self.object_color.setStyleSheet('QLabel {background: ' + rgb + ';}')
        self.object_color.setText(f'{object_id}')

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()
