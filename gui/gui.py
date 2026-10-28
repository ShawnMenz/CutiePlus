import functools
from typing import List, Optional

import numpy as np
from omegaconf import DictConfig

from PySide6.QtWidgets import (QWidget, QComboBox, QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSpinBox, QVBoxLayout, QSizePolicy,
                               QSlider, QApplication, QFileDialog,
                               QMessageBox, QFrame, QScrollArea, QSplitter, QStyle,
                               QGridLayout, QToolButton)

from PySide6.QtGui import (QKeySequence, QShortcut, QTextCursor, QImage, QPixmap, QIcon, QPainter,
                           QColor, QCursor, QPen)
from PySide6.QtCore import Qt, QEvent, QTimer, QPointF, QRectF, QSize

from gui.flow_layout import FlowLayout
from gui.gui_utils import *
from gui.category_menu import CategoryMenu, CategoryPill
from gui.style import (ATTENTION, CARD_MARGINS, GAP, Card, category_pill, set_pill,
                       set_variant)
from gui.task_state import (category_text, describe_attention, format_intervals,
                            interval_index_from)
from gui.theme import save_setting, setting
from gui.tips_window import TipsWindow
from gui.interactive_utils import CLICK_DOT_DIAMETER, NEGATIVE_CLICK_COLOR, POSITIVE_CLICK_COLOR

# Need Attention: orange text, filled orange while the sequence is flagged
ATTENTION_COLOR = ATTENTION  # Need Attention's orange (gui/style.py)

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
# the playback's speed (R and Shift+R), set in the box beside the button; every window starts at
# PLAYBACK_FPS, or at the speed the Lock button beside the box keeps (as the theme is) until it
# is unlocked
PLAYBACK_FPS = 24
MAX_PLAYBACK_FPS = 60
FPS_LOCK_SETTING = 'play_fps_lock'  # the locked speed; 0 when it is not locked
OUTPUT_FPS_SETTING = 'output_fps'  # the demo videos' frames per second
OUTPUT_VIDEO_LABEL = 'Output video (Ctrl+E)'
LOCK_LABEL = 'Lock'
LOCKED_LABEL = 'Locked'
# the "layer" view, which puts an image (Import layer; the original Cutie's UIUC logo) between the
# target and the background: off, so neither it nor Import layer is offered; True brings both back
SHOW_LAYER_MODE = False


def locked_play_fps() -> Optional[int]:
    """The locked playback speed, None when it is not locked."""
    try:
        fps = int(setting(FPS_LOCK_SETTING, 0))
    except (TypeError, ValueError):
        return None
    return fps if 1 <= fps <= MAX_PLAYBACK_FPS else None


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
        locked_fps = locked_play_fps()
        self.play_fps_box = QSpinBox()
        self.play_fps_box.setRange(1, MAX_PLAYBACK_FPS)
        self.play_fps_box.setSuffix(' fps')
        self.play_fps_box.setValue(locked_fps or PLAYBACK_FPS)
        self.play_fps_box.valueChanged.connect(self.on_play_fps_change)
        # Lock: every window opened from then on starts at the box's speed
        self.play_fps_lock_button = set_variant(QPushButton(LOCK_LABEL), 'toggle')
        self.play_fps_lock_button.setCheckable(True)
        self.fit_to_labels(self.play_fps_lock_button, (LOCK_LABEL, LOCKED_LABEL))
        self.play_fps_lock_button.clicked.connect(self.on_play_fps_lock)
        self.show_play_fps_lock(locked_fps is not None)
        self.commit_button = set_variant(QPushButton('Commit to permanent memory (C)'), 'tonal')
        self.commit_button.clicked.connect(controller.on_commit)
        self.export_video_button = QPushButton('Export as video (Ctrl+E)')
        self.export_video_button.clicked.connect(controller.on_export_visualization)
        # a task's demo videos (gui/demo_video.py); Cancel output while they are written
        self.output_video_button = set_variant(QPushButton(OUTPUT_VIDEO_LABEL), 'tonal')
        self.output_video_button.clicked.connect(controller.on_output_demo)
        self.fit_to_labels(self.output_video_button, (OUTPUT_VIDEO_LABEL, 'Cancel output'))
        self.output_video_button.setToolTip(
            'Ctrl+E: saves the sequence, writes one video per interval to cutie_demo/ beside the '
            'input (the RGB frame left, the thermal one right, the mask in the visualization mode '
            'and the category\'s color, the frame number at the top left and "category: '
            'description" below), then marks the sequence done.')
        self.export_binary_button = set_variant(QPushButton('Export binary masks (Ctrl+S)'), 'primary')
        self.export_binary_button.clicked.connect(controller.on_save)
        if task is not None:
            # a task's masks and metadata.json are written as it is labeled; the dataset is
            # exported from the task browser
            self.export_binary_button.setText('Save (Ctrl+S)')
            self.export_binary_button.setToolTip(
                'Masks are saved as you label; Save also saves the intervals and the description '
                'and submits the sequence for review. The task browser\'s Mark done then accepts '
                'it, and its Export writes the dataset.')
            # a toggle: the sequence needs further attention (orange; filled while on)
            self.attention_button = QPushButton('Need Attention')
            self.attention_button.setCheckable(True)
            set_variant(self.attention_button, 'attention')  # orange; filled while on
            self.attention_button.clicked.connect(controller.on_toggle_attention)
            self.refresh_attention_button()

        # the main actions are filled (gui/style.py)
        self.forward_run_button = set_variant(QPushButton(FORWARD_LABEL), 'primary')
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)
        self.fit_to_labels(self.forward_run_button, (FORWARD_LABEL, PAUSE_FORWARD_LABEL))

        self.backward_run_button = set_variant(QPushButton(BACKWARD_LABEL), 'primary')
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)
        self.fit_to_labels(self.backward_run_button, (BACKWARD_LABEL, PAUSE_BACKWARD_LABEL))

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(200)
        # the shared GPU's queue, while a propagation waits for it (cooperation service)
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.gpu_status_label.hide()

        self.reset_frame_button = QPushButton('Reset frame')
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_all_masks_button = set_variant(QPushButton('Reset all frame masks'), 'danger')
        self.reset_all_masks_button.clicked.connect(controller.on_reset_all_frame_masks)

        # set up the LCD
        self.lcd = QTextEdit()
        self.lcd.setObjectName('frameCounter')
        self.lcd.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.lcd.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(150)
        # 1-based frame numbers, like the frame names and the interval boxes
        self.lcd.setText('{: 5d} / {: 5d}'.format(1, controller.T))

        self.frame_name = QLabel()
        self.frame_name.setMinimumSize(100, 30)
        # centered vertically, like the boxes beside it in the top row
        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.frame_name.setTextFormat(Qt.TextFormat.RichText)

        # the target of a task: its category (a click chooses another), and the referring text
        # as a box to tune it
        self.task_label = CategoryPill()
        self.category_menu: Optional[CategoryMenu] = None
        self.description_edit = TextBox()
        if task is not None:
            self.task_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.task_label.clicked.connect(self.open_category_menu)
            self.set_description(task.description)
            self.description_edit.editingFinished.connect(controller.on_description_change)
            self.commit_button.setText('Mark as anchor (C)')

            # the dataset clips, 1-indexed and inclusive, each a set of its own: a category, a
            # description and an interval. Set 1 is the sequence's own (its category and
            # description; its interval is the saved one, else the whole sequence); every interval
            # added with '+ Add interval' is another set. One set is shown at a time, on the top
            # row: every set's description box, interval boxes and remove button sit in the same
            # cells, only the selected set's visible. The triangle opens the list of every set
            # under it (row 1) to pick one.
            self.interval_pairs: List[IntervalPair] = []
            self.selected_set = 0
            self.sets_button = QToolButton()  # '▶ 1/3', '▼ 1/3' while the list is open
            self.sets_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.sets_button.setAutoRaise(True)
            self.sets_button.setToolTip('Show every interval of this sequence (its category, '
                                        'description and frames) to pick the one to see and '
                                        'edit.')
            self.sets_button.clicked.connect(self.toggle_sets)
            self.set_rows: List[IntervalSetRow] = []
            self.set_list = QWidget()
            self.set_list_layout = QVBoxLayout(self.set_list)
            self.set_list_layout.setContentsMargins(0, 2, 0, 2)
            self.set_list_layout.setSpacing(2)
            self.set_list.hide()
            self.interval_grid = QGridLayout()
            self.interval_grid.setHorizontalSpacing(10)
            self.interval_grid.setVerticalSpacing(4)
            self.interval_grid.setColumnStretch(2, 1)
            self.interval_grid.addWidget(self.sets_button, 0, 0)
            self.interval_grid.addWidget(self.task_label, 0, 1)
            self.interval_grid.addWidget(self.description_edit, 0, 2)
            self.add_interval_button = QPushButton('+ Add interval')
            self.add_interval_button.setToolTip(
                'Keep another stretch of this sequence: each interval becomes a clip of its own, '
                'with a category and a description of its own (they start as the sequence\'s). '
                'Intervals may touch but not overlap.')
            self.add_interval_button.clicked.connect(self.on_add_interval)
            self.interval_grid.addWidget(self.add_interval_button, 0, 4)
            # the frame label carries the mask source; the bottom row is too narrow for it
            set_pill(self.frame_name, 'frame')
            self.interval_grid.addWidget(self.frame_name, 0, 6)
            # the list of sets, under the selected one
            self.interval_grid.addWidget(self.set_list, 1, 1, 1, 6)
            intervals, _ = task.intervals(controller.T)
            first = task.first_interval(controller.T)
            descriptions = task.interval_descriptions(controller.T)
            categories = task.interval_categories(controller.T)
            self.add_interval_pair(*first)
            for interval in intervals:
                if interval != first:
                    self.add_interval_pair(*interval, description=descriptions[interval],
                                           category=categories[interval])
            # the first pair, as the single interval always was
            self.interval_start, self.interval_end = self.interval_pairs[0].boxes()
            self.show_selected_set()

        # timeline slider
        self.tl_slider = FrameSlider(Qt.Orientation.Horizontal)
        self.tl_slider.setObjectName('frameSlider')  # styled in gui/style.py
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
            self.tl_slider.set_selected_interval(self.selected_pair().entry())

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("mask")
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        if SHOW_LAYER_MODE:
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
        self.fps_dial.setValue(int(setting(OUTPUT_FPS_SETTING, cfg['output_fps'])))
        self.fps_dial.valueChanged.connect(controller.on_fps_dial_change)
        if task is not None:
            self.fps_dial.setSuffix(' fps')
            self.fps_dial.setToolTip('Output FPS: the frames per second of the demo videos')

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

        # the layout, as rounded cards (gui/style.py): the target and its intervals on top, the
        # canvas with a sidebar (messages, and the Advanced controls folded away), the frame
        # slider, and three cards of controls below it, which wrap in a narrow window
        def row(*widgets, stretch=True):
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(GAP)
            for widget in widgets:
                line.addWidget(widget)
            if stretch:
                line.addStretch(1)
            return line

        def rows(*lines):
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(GAP)
            for line in lines:
                column.addLayout(line)
            return column

        def section(text):  # a small heading inside the Advanced card
            label = QLabel(text)
            label.setProperty('section', True)
            return label

        # one object: Reset frame empties the frame; a plain workspace shows its frame label here
        # (a task's sits in the top card)
        frame_card = Card(rows(row(self.lcd, self.play_button, self.play_fps_box,
                                   self.play_fps_lock_button),
                               row(self.reset_frame_button,
                                   *([self.frame_name] if task is None else []))),
                          title='Frame')

        propagate_card = Card(rows(row(self.backward_run_button, self.forward_run_button),
                                   row(self.commit_button, self.progressbar, self.gpu_status_label,
                                       stretch=False)),
                              title='Propagate')
        self.progressbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        save_buttons = [self.export_binary_button] + (
            [self.output_video_button, self.fps_dial] if task is not None else [])
        mode_line = ([self.attention_button] if task is not None else []) + \
            [QLabel('Visualization mode'), self.combo]
        save_card = Card(rows(row(*save_buttons), row(*mode_line)), title='Save')

        for card in (frame_card, propagate_card, save_card):
            for child in card.body.findChildren(QPushButton) + card.body.findChildren(QLabel):
                child.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        navi = FlowLayout(spacing=12, fill=True)
        for card in (frame_card, propagate_card, save_card):
            navi.addWidget(card)

        # the sidebar: the messages take its height; the controls of the original Cutie (memory,
        # imports, the overlay video, Reset all frame masks) wait in the Advanced card, open or
        # closed as last left
        self.tips_button.setText('Tips (F1)')
        messages_card = Card(self.console, title='Messages', trailing=(self.tips_button, ),
                             stretch=True)
        advanced = QVBoxLayout()
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setSpacing(GAP)
        advanced.addWidget(section('Memory'))
        for line in (self.perm_mem_gauge_layout, self.work_mem_gauge_layout,
                     self.long_mem_gauge_layout, self.gpu_mem_gauge_layout,
                     self.torch_mem_gauge_layout):
            advanced.addLayout(line)
        advanced.addLayout(row(self.clear_all_mem_button, self.clear_non_perm_mem_button,
                               stretch=False))
        for line in (self.work_mem_min_layout, self.work_mem_max_layout, self.long_mem_max_layout,
                     self.mem_every_box_layout):
            advanced.addLayout(line)
        advanced.addWidget(section('Import'))
        advanced.addLayout(row(self.import_mask_button,
                               *([self.import_layer_button] if SHOW_LAYER_MODE else []),
                               stretch=False))
        if task is None:
            advanced.addWidget(section('Overlay video'))
            advanced.addLayout(row(QLabel('Save visualization'), self.save_visualization_combo))
            advanced.addLayout(row(QLabel('Save soft mask during propagation'),
                                   self.save_soft_mask_checkbox))
            advanced.addLayout(row(QLabel('Output FPS'), self.fps_dial, QLabel('Bitrate (Mbps)'),
                                   self.bitrate_dial))
            advanced.addLayout(row(self.export_video_button))
        else:
            # a task's work folder holds masks/ and metadata.json only: no overlay video, no
            # visualizations or soft masks saved into it
            self.save_visualization_combo.hide()
            self.save_soft_mask_checkbox.hide()
        advanced.addWidget(section('Reset'))
        advanced.addLayout(row(self.reset_all_masks_button))
        open_advanced = setting('advanced_open', 'false') in (True, 'true')
        self.advanced_card = Card(advanced, title='Advanced', collapsible=True,
                                  expanded=open_advanced)
        self.advanced_card.toggled.connect(lambda shown: save_setting('advanced_open', shown))

        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 0, 0)
        sidebar.setSpacing(12)
        sidebar.addWidget(messages_card, 1)
        sidebar.addWidget(self.advanced_card)

        # the sidebar scrolls when the window is short; drag its edge to resize it, or to the
        # window's edge to hide it
        right_panel = QScrollArea()
        right_panel.setWidget(layout_widget(sidebar))
        right_panel.setWidgetResizable(True)
        right_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_panel.setFrameShape(QFrame.Shape.NoFrame)
        # wide enough for the Advanced card's content, open or not, beside the scroll bar (polished
        # first: the theme's style sheet sets the frame)
        right_panel.ensurePolished()
        content_width = max(right_panel.widget().minimumSizeHint().width(),
                            self.advanced_card.body.minimumSizeHint().width()
                            + CARD_MARGINS[0] + CARD_MARGINS[2] + 2)
        right_panel.setMinimumWidth(
            content_width + right_panel.verticalScrollBar().sizeHint().width()
            + 2 * right_panel.frameWidth()
            + right_panel.style().pixelMetric(QStyle.PixelMetric.PM_ScrollView_ScrollBarSpacing,
                                              None, right_panel))

        # the canvas in a card of its own
        canvas_card = Card(self.main_canvas, stretch=True)
        canvas_card.layout().setContentsMargins(6, 6, 6, 6)

        # Drawing area: main canvas and the right panel
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(canvas_card)
        self.splitter.addWidget(right_panel)
        self.splitter.setHandleWidth(12)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([4000, 1000])  # relative: the panel starts at a fifth of the width
        self.splitter.setCollapsible(0, False)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        if task is not None:
            # the descriptions and their intervals, a row each
            layout.addWidget(Card(self.interval_grid))
        layout.addWidget(self.splitter, 1)
        # the frame slider between two pills naming the keys that step it
        slider_row = QHBoxLayout()
        slider_row.setSpacing(GAP)
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

        # without a mouse (a touchpad): , and . click at the pointer as the left and right buttons
        # do; Ctrl + + / Ctrl + - zoom as Ctrl + wheel, Ctrl+0 fits the image, and the arrows
        # scroll a zoomed view as the wheel. The mouse works as before
        for key, action in ((Qt.Key.Key_Comma, 'left'), (Qt.Key.Key_Period, 'right')):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setAutoRepeat(False)  # a held key adds one click, not a burst
            shortcut.activated.connect(functools.partial(self.on_key_click, action))
        for keys, factor in ((('Ctrl+=', 'Ctrl++'), ZOOM_STEP), (('Ctrl+-', ), 1 / ZOOM_STEP)):
            for keys_text in keys:
                QShortcut(QKeySequence(keys_text),
                          self).activated.connect(functools.partial(self.on_key_zoom, factor))
        QShortcut(QKeySequence('Ctrl+0'), self).activated.connect(self.on_fit_view)
        for key, notches_x, notches_y in ((Qt.Key.Key_Left, 1, 0), (Qt.Key.Key_Right, -1, 0),
                                          (Qt.Key.Key_Up, 0, 1), (Qt.Key.Key_Down, 0, -1)):
            QShortcut(QKeySequence(key), self).activated.connect(
                functools.partial(self.on_key_scroll, notches_x, notches_y))

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

        # save (a task) or export binary masks / the overlay video (the keys are on the buttons)
        QShortcut(QKeySequence('Ctrl+S'), self).activated.connect(controller.on_save)
        QShortcut(QKeySequence('Ctrl+E'), self).activated.connect(
            controller.on_export_visualization if task is None else controller.on_output_demo)

        # quit shortcuts: Q closes the window; Ctrl+Q also marks the task done (it asks first)
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)
        QShortcut(QKeySequence('Ctrl+Q'), self).activated.connect(controller.on_quit_done)
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
        # a task fills the screen; otherwise
        # the controls in one row and at least the image plus 200 px, maximized instead when that
        # does not fit the screen
        fit_window(self, self.sizeHint().expandedTo(QSize(self.w + 200, self.h + 200)),
                   fill=self.controller.task is not None)

    def add_interval_pair(self, start: Optional[int] = None, end: Optional[int] = None,
                          description: Optional[str] = None,
                          category: Optional[str] = None) -> IntervalPair:
        """A set's interval boxes. The first pair is set 1's, the sequence's own (its category and
        description are the sequence's); every later one has a description and a category of its
        own (by default the sequence's) and a button that removes it. Every set's widgets sit in
        the same cells of the top row, only the selected set's shown (show_selected_set). A pair
        without values starts blank."""
        first = not self.interval_pairs
        task = self.controller.task
        if not first and description is None:
            description = self.description_edit.text().strip() or task.description
        if not first and category is None:
            category = task.category
        pair = IntervalPair(self.controller.T, start, end, removable=not first,
                            description=description, category=None if first else category)
        for box in pair.boxes():
            which = box.which
            box.setToolTip(f'Interval {which} in frame numbers; type m for the '
                           f'{"first" if which == "start" else "last"} frame. Enter or click '
                           'elsewhere to save; each interval is a clip of its own.\n'
                           '{ / }: go to the previous / next interval start; Shift+R: play the '
                           'intervals, one per press.')
            if first:  # the controller's handler for the single interval, unchanged
                box.editingFinished.connect(
                    functools.partial(self.controller.on_interval_change, which))
            else:
                box.editingFinished.connect(
                    functools.partial(self.controller.on_interval_change, which, pair))
            box.editingFinished.connect(functools.partial(self.return_focus_from, box))
            box.valueChanged.connect(self.refresh_set_list)
        if first:
            self.description_edit.textChanged.connect(self.refresh_set_list)
        else:
            set_variant(pair.remove_button, 'danger')
            pair.description_edit.editingFinished.connect(
                functools.partial(self.controller.on_interval_description_change, pair))
            pair.description_edit.editingFinished.connect(
                functools.partial(self.return_focus_from, pair.description_edit))
            pair.description_edit.textChanged.connect(self.refresh_set_list)
            self.interval_grid.addWidget(pair.description_edit, 0, 2)
            self.interval_grid.addWidget(pair.remove_button, 0, 5)
        pair.removed.connect(self.controller.on_remove_interval)
        self.interval_pairs.append(pair)
        self.interval_grid.addWidget(pair, 0, 3, Qt.AlignmentFlag.AlignLeft)
        self.show_selected_set()
        return pair

    def selected_pair(self) -> IntervalPair:
        return self.interval_pairs[self.selected_set]

    def selected_category(self) -> Optional[str]:
        # the shown set's category: set 1's is the sequence's
        if self.selected_set == 0:
            return self.controller.task.category
        return self.selected_pair().category

    @staticmethod
    def set_shown(widget: Optional[QWidget], shown: bool) -> None:
        # a widget not placed in the window yet only hides: shown, it would be a window of its own
        if widget is None:
            return
        if not shown:
            widget.hide()
        elif widget.parentWidget() is not None:
            widget.show()

    def show_selected_set(self) -> None:
        # the selected set's category, description and interval on the top row, no other's; the
        # triangle says which set of how many
        for index, pair in enumerate(self.interval_pairs):
            for widget in (pair, pair.description_edit, pair.remove_button):
                self.set_shown(widget, index == self.selected_set)
        self.set_shown(self.description_edit, self.selected_set == 0)
        self.show_sets_button()
        self.show_category()
        self.refresh_set_list()

    def select_set(self, index: int, jump: bool = True) -> None:
        """Shows set index and folds the list: its band is the bold one, and the mask takes its
        category's color. jump: go to the first frame of its interval."""
        self.selected_set = index
        self.show_selected_set()
        self.fold_sets()
        entry = self.selected_pair().entry()
        self.tl_slider.set_selected_interval(entry)
        self.controller.show_category_color()
        self.controller.show_current_frame()
        if jump and entry is not None:
            self.tl_slider.setValue(entry[0] - 1)

    def select_interval(self, interval) -> None:
        # the set of a saved interval ({ / }, Shift+R), the frame left where they put it
        for index, pair in enumerate(self.interval_pairs):
            if pair.entry() == tuple(interval):
                if index != self.selected_set:
                    self.select_set(index, jump=False)
                return

    def show_sets_button(self) -> None:
        # the triangle, down while the list is open, and which set of how many is shown
        arrow = '▶' if self.set_list.isHidden() else '▼'
        self.sets_button.setText(f'{arrow} {self.selected_set + 1}/{len(self.interval_pairs)}')

    def toggle_sets(self) -> None:
        # the triangle: opens the list of every set under the selected one, or folds it
        if self.set_list.isHidden():
            self.refresh_set_list()
            self.set_list.show()
            self.show_sets_button()
        else:
            self.fold_sets()

    def fold_sets(self) -> None:
        self.set_list.hide()
        self.show_sets_button()

    def refresh_set_list(self, *_) -> None:
        # one row per set, as typed: number, category, description, frames; the selected tinted
        task = self.controller.task
        colors = {category.name: category.color for category in task.categories}
        while len(self.set_rows) < len(self.interval_pairs):
            row = IntervalSetRow()
            row.clicked.connect(functools.partial(self.select_set, len(self.set_rows)))
            self.set_list_layout.addWidget(row)
            self.set_rows.append(row)
        while len(self.set_rows) > len(self.interval_pairs):
            row = self.set_rows.pop()
            self.set_list_layout.removeWidget(row)
            row.deleteLater()
        for index, (row, pair) in enumerate(zip(self.set_rows, self.interval_pairs)):
            if index == 0:
                category = task.category
                description = self.description_edit.text().strip() or task.description
            else:
                category = pair.category
                description = pair.description() or pair.kept_description
            row.show_set(index + 1, category_text(category), colors.get(category), description,
                         pair.frames_text(), index == self.selected_set)

    def on_add_interval(self) -> None:
        pair = self.add_interval_pair()
        self.select_set(len(self.interval_pairs) - 1, jump=False)
        pair.start_box.setFocus()
        self.text(f'Added interval {len(self.interval_pairs)}: type its start and end (m = first '
                  '/ last frame). Its category and description start as the sequence\'s: change '
                  f'them here. {REMOVE_INTERVAL_TEXT} cancels it; the triangle on the left lists '
                  'every interval.')

    def remove_interval_pair(self, pair: IntervalPair) -> None:
        # back to set 1, the frame left where it is
        self.interval_pairs.remove(pair)
        for widget in (pair.description_edit, pair, pair.remove_button):
            self.interval_grid.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        self.select_set(0, jump=False)
        self.tl_slider.setFocus()

    def interval_description_text(self, interval) -> str:
        # ' Description: ...' for a message about one of several intervals, else ''
        task = self.controller.task
        intervals, _ = task.intervals(self.controller.T)
        if len(intervals) < 2:
            return ''
        return f' Description: {task.interval_descriptions(self.controller.T)[interval]}'

    def return_focus_from(self, box: QWidget) -> None:
        # Enter; a box that lost the focus to another widget leaves it there
        if box.hasFocus():
            self.tl_slider.setFocus()

    def show_category(self) -> None:
        # the selected set's category, in its color; gray without one (or one missing from the
        # list)
        task = self.controller.task
        category = self.selected_category()
        colors = {item.name: item.color for item in task.categories}
        self.task_label.setText(category_text(category))
        category_pill(self.task_label, colors.get(category))
        self.task_label.setToolTip('Click to choose the category of this interval; a change '
                                   'needs a new review.')

    def open_category_menu(self) -> None:
        task = self.controller.task
        self.category_menu = CategoryMenu(task.categories, self.selected_category(), self)
        if self.selected_set == 0:  # the sequence's own
            self.category_menu.chosen.connect(self.controller.on_category_chosen)
        else:
            self.category_menu.chosen.connect(functools.partial(
                self.controller.on_interval_category_chosen, self.selected_pair()))
        self.category_menu.popup_below(self.task_label)

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

    def confirm_quit_done(self) -> bool:
        # Ctrl+Q: the warning before the task is saved and marked done in one go
        task = self.controller.task
        intervals, _ = task.intervals(self.controller.T)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle('Quit and mark done')
        dialog.setText(f'You are about to quit and save the intervals of {task.sequence_id} as '
                       '\'done\'.')
        dialog.setInformativeText(
            f'Intervals: {format_intervals(intervals)}. They are saved with their descriptions and '
            f'masks and marked done by {task.labeler}, with no separate review in the task '
            'browser. Use this only when you label and review this sequence at the same time, and '
            'check every interval, its description and its masks first.\n\nThe window then '
            'closes. The task browser\'s Export puts it in the dataset.')
        quit_button = dialog.addButton('Quit and mark done', QMessageBox.ButtonRole.AcceptRole)
        cancel_button = dialog.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() is quit_button

    def confirm_held_result(self, job: dict) -> bool:
        """A propagation's result held because the task changed after it was submitted."""
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setWindowTitle('A propagation result is waiting')
        dialog.setText('A propagation of this task ended, but the task changed after it was '
                       'started.')
        dialog.setInformativeText(
            f'It started from frame {job.get("start") or "?"} at {job["submitted_at"][:16]} '
            'UTC. Apply its masks anyway (anchors are kept), or discard them?')
        apply_button = dialog.addButton('Apply', QMessageBox.ButtonRole.AcceptRole)
        discard_button = dialog.addButton('Discard', QMessageBox.ButtonRole.DestructiveRole)
        dialog.setDefaultButton(discard_button)
        dialog.exec()
        return dialog.clickedButton() is apply_button

    def set_status_text(self, text: str) -> None:
        """A line about the shared GPU's queue; empty hides it."""
        self.gpu_status_label.setText(text)
        self.gpu_status_label.setVisible(bool(text))

    def confirm_reset_all_masks(self) -> bool:
        task = self.controller.task
        sequence = task.sequence_id if task is not None else self.cfg['workspace']
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle('Reset all frame masks')
        dialog.setText(f'Delete all frame masks for {sequence}?')
        if task is not None:
            dialog.setInformativeText(
                'This deletes the task\'s masks and clears its anchors. The frames and the '
                'pregenerated masks of the input remain, and show again. This cannot be undone.')
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
        # a thin white ring: a dot stays visible on a mask of its own color (a green car)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
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

    def pointer_on_canvas(self) -> QPointF:
        # where the pointer is, in canvas coordinates (outside the canvas too)
        return QPointF(self.main_canvas.mapFromGlobal(QCursor.pos()))

    def on_key_click(self, action: str) -> None:
        # , and .: a left / right click at the pointer, for a touchpad without buttons
        if self.canvas_image is None:
            return
        pointer = self.pointer_on_canvas()
        if (not self.main_canvas.rect().contains(pointer.toPoint())
                or self.is_pos_out_of_bound(pointer.x(), pointer.y())):
            self.text('Move the pointer over the image, then press , (add) or . (remove).')
            return
        self.click_fn(action, *self.get_scaled_pos(pointer.x(), pointer.y()))

    def on_key_zoom(self, factor: float) -> None:
        # Ctrl + + / Ctrl + -: at the pointer when it is on the canvas (as Ctrl + wheel), else at
        # the centre
        if self.canvas_image is None:
            return
        pointer = self.pointer_on_canvas()
        if not self.main_canvas.rect().contains(pointer.toPoint()):
            pointer = QPointF(self.main_canvas.width() / 2, self.main_canvas.height() / 2)
        self.zoom_at(pointer, factor)

    def on_fit_view(self) -> None:
        # Ctrl+0: the whole image again
        if self.canvas_image is None:
            return
        self.zoom = 1.0
        self.view_center = None
        self.render_canvas()

    def on_key_scroll(self, notches_x: float, notches_y: float) -> None:
        # the arrows, one wheel notch each; an image that fits the canvas does not move
        if self.canvas_image is not None:
            self.scroll_view(notches_x, notches_y)

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
            self.timer.start(self.playback_interval_ms())
            self.play_button.setText(STOP_LABEL)

    def playback_interval_ms(self) -> int:
        # the timer's step at the speed in the box
        return round(1000 / self.play_fps_box.value())

    def on_play_fps_change(self, fps: int) -> None:
        # a playback under way changes speed at once
        if self.timer.isActive():
            self.timer.setInterval(self.playback_interval_ms())

    def on_play_fps_lock(self, locked: bool) -> None:
        # Lock keeps the box's speed for every window opened from now on; Unlock
        # goes back to PLAYBACK_FPS, here and in the next windows
        fps = self.play_fps_box.value()
        save_setting(FPS_LOCK_SETTING, fps if locked else 0)
        if not locked:
            self.play_fps_box.setValue(PLAYBACK_FPS)
        self.show_play_fps_lock(locked)
        if locked:
            self.text(f'Playback speed locked at {fps} fps: every window you open starts at it, '
                      'until you unlock it.')
        else:
            self.text(f'Playback speed unlocked: back to {PLAYBACK_FPS} fps, here and in the '
                      'next windows.')

    def show_play_fps_lock(self, locked: bool) -> None:
        # while locked, the box keeps its speed
        button = self.play_fps_lock_button
        button.setChecked(locked)
        button.setText(LOCKED_LABEL if locked else LOCK_LABEL)
        self.play_fps_box.setEnabled(not locked)
        fps = self.play_fps_box.value()
        if locked:
            button.setToolTip(f'Locked at {fps} fps: every window opened starts at it. Click to unlock: the speed goes back to {PLAYBACK_FPS} fps, '
                              'here and in the next windows.')
            self.play_fps_box.setToolTip(f'Playback speed of R and Shift+R: locked at {fps} fps. '
                                         'Unlock (the button on the right) to change it.')
        else:
            button.setToolTip('Lock the speed in the box: every window opened from now on starts '
                              f'at it, instead of {PLAYBACK_FPS} fps, until you '
                              'unlock it.')
            self.play_fps_box.setToolTip(f'Playback speed of R and Shift+R, in frames per second; '
                                         f'the frames may load slower. Every window starts at '
                                         f'{PLAYBACK_FPS}, unless Lock keeps a speed.')

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
        self.select_interval((start, end))
        self.timer.start(self.playback_interval_ms())
        self.play_button.setText(STOP_LABEL)
        self.text(f'Previewing interval {number + 1} of {len(intervals)}: frames {start}-{end} '
                  '(Shift+R: next).' + self.interval_description_text((start, end)))

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
        self.select_interval((start, end))
        self.preview_index = number
        self.text(f'Interval {number + 1} of {len(intervals)}: frames {start}-{end} '
                  '(Shift+R plays it).' + self.interval_description_text((start, end)))

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

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   prompt,
                                                   "",
                                                   "Image files (*)",
                                                   options=options)
        return file_name

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()

    def show_demo_output(self, running: bool, written: int = 0, total: int = 1) -> None:
        # while the demo videos are written: Cancel output, and the frames written so far
        self.output_video_button.setText('Cancel output' if running else OUTPUT_VIDEO_LABEL)
        self.fps_dial.setEnabled(not running)
        self.progressbar.setValue(int(100 * written / total) if running else 0)
