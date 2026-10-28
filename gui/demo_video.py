"""
The demo videos of a task (the labeling window's Output video): one MP4 for each interval of the
sequence, the dataset's clips, in cutie_demo/<tab>/<sequence>_<start>-<end>_<mode>.mp4 beside
the input root:

    +-------------------------------------+
    | 12/101                              |   the frame of the clip, of its frames
    +------------------+------------------+
    |   visual + mask  |  thermal + mask  |   in the window's visualization mode; the target in
    +------------------+------------------+   the interval's category color
    |    car: the red car turning left    |   the category in that color, then the description
    +-------------------------------------+

The bands are dark: nothing is drawn on the frames but the mask.

A sequence with one modality has one panel; a frame that a modality lacks is black there. Each
frame shows the mask the window shows: the saved one, else the pregenerated one, else none.
Nothing here is a widget: the clips are plain data taken on the GUI thread (task_clips), then
written in a thread of their own (write_demos); QPainter on a QImage is allowed there.
"""
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from os import path
from typing import Callable, List, Optional, Tuple

import av
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

from gui.category_colors import hex_to_rgb
from gui.interactive_utils import colors_with_target, get_visualization
from gui.resource_manager import ResourceManager, read_mask
from gui.task_input import MODALITIES
from gui.task_state import TaskState, category_text

log = logging.getLogger(__name__)

DEMO_DIR = 'cutie_demo'  # beside the input root
Color = Tuple[int, int, int]
BAND_COLOR = (24, 24, 24)  # the counter's band above the frames, the caption's below
TEXT_COLOR = (240, 240, 240)
TEXT_SHARE = 0.05  # the text's height, of the frame's
MARGIN_SHARE = 0.5  # the band's margins, of the text's height


class DemoCancelled(Exception):
    pass


@dataclass
class DemoFrame:
    images: List[Optional[str]]  # one file per panel (DemoClip.modalities); None: it lacks one
    mask: Optional[str] = None  # ResourceManager.saved_mask_source
    pregenerated: bool = False


@dataclass
class DemoClip:
    output: str
    modalities: List[str]
    frames: List[DemoFrame]
    category: str
    description: str
    color: Optional[Color] = None  # the category's; None: the window's cyan
    mode: str = 'davis'
    layer: Optional[np.ndarray] = field(default=None, repr=False)  # the layer mode's RGBA image


def demo_root_of(input_root: str) -> str:
    return path.join(path.dirname(path.abspath(input_root)), DEMO_DIR)


def demo_name(sequence: str, interval: Tuple[int, int], mode: str) -> str:
    return f'{sequence}_{interval[0]}-{interval[1]}_{mode}.mp4'


def task_clips(task: TaskState, res_man: ResourceManager, demo_root: str, mode: str,
               layer: Optional[np.ndarray] = None) -> List[DemoClip]:
    """The clips of the task as saved now, one per interval (the whole sequence without any)."""
    count = res_man.length
    intervals = task.intervals(count)[0]
    categories = task.interval_categories(count)
    descriptions = task.interval_descriptions(count)
    colors = {category.name: category.color for category in task.categories}
    modalities = [modality for modality in MODALITIES if modality in task.frame_dirs]
    clips = []
    for interval in intervals:
        frames = []
        for ti in range(interval[0] - 1, interval[1]):
            mask, pregenerated = res_man.saved_mask_source(ti)
            frames.append(DemoFrame([res_man.modality_path(ti, modality)
                                     for modality in modalities], mask, pregenerated))
        category = categories.get(interval)
        color = colors.get(category)
        clips.append(DemoClip(
            path.join(demo_root, task.split, demo_name(task.sequence_id, interval, mode)),
            modalities, frames, category_text(category), descriptions.get(interval, ''),
            hex_to_rgb(color) if color else None, mode, layer))
    return clips


# drawing

def _fit_height(image: np.ndarray, height: int, resample) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return np.array(Image.fromarray(image).resize((width, height), resample))


def _panel(image: Optional[np.ndarray], size: Tuple[int, int], mask: Optional[np.ndarray],
           mode: str, colors: np.ndarray, layer: Optional[np.ndarray]) -> np.ndarray:
    height, width = size
    if image is None:  # the modality lacks this frame
        return np.zeros((height, width, 3), np.uint8)
    image = _fit_height(image, height, Image.Resampling.BILINEAR)
    if mask is None:
        mask = np.zeros(image.shape[:2], np.uint8)
    elif mask.shape != image.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(image.shape[1::-1], Image.Resampling.NEAREST))
    if layer is not None and layer.shape[:2] != image.shape[:2]:
        layer = np.array(Image.fromarray(layer).resize(image.shape[1::-1],
                                                       Image.Resampling.BILINEAR))
    shown = get_visualization(mode, image, mask, layer, [1], colors)
    if shown.shape[2] == 4:  # rgba: the target on black
        shown = (shown[:, :, :3] * (shown[:, :, 3:] / 255)).astype(np.uint8)
    return shown


def _font(frame_height: int) -> QFont:
    font = QFont()
    font.setPixelSize(max(12, round(frame_height * TEXT_SHARE)))
    return font


def _margin(metrics: QFontMetricsF) -> float:
    return metrics.height() * MARGIN_SHARE


def top_band_height(frame_height: int) -> int:
    """The band above the frames, which holds the counter: one line of text and its margins."""
    metrics = QFontMetricsF(_font(frame_height))
    return int(np.ceil(metrics.lineSpacing() + 2 * _margin(metrics)))


def caption_lines(category: str, description: str, font: QFont, width: float) -> List[str]:
    """'category: description' wrapped by words to the width; the category starts the first."""
    metrics = QFontMetricsF(font)
    words = f'{category}: {description}'.split() if description else [category]
    lines = []
    for word in words:
        if lines and metrics.horizontalAdvance(f'{lines[-1]} {word}') <= width:
            lines[-1] = f'{lines[-1]} {word}'
        else:
            lines.append(word)
    return lines


def compose_frame(images: List[Optional[np.ndarray]], mask: Optional[np.ndarray], mode: str,
                  color: Optional[Color], category: str, description: str, index: int,
                  total: int, layer: Optional[np.ndarray] = None) -> np.ndarray:
    """One frame of a demo video (H*W*3 uint8, both even): the counter (index of total,
    1-based) in a band above, the panels side by side, the caption band below."""
    first = next(image for image in images if image is not None)
    height = first.shape[0]
    colors = colors_with_target(color)
    panels = [_panel(image, (height, (image if image is not None else first).shape[1]), mask,
                     mode, colors, layer) for image in images]
    body = np.concatenate(panels, axis=1)
    font = _font(height)
    metrics = QFontMetricsF(font)
    margin = _margin(metrics)
    lines = caption_lines(category, description, font, body.shape[1] - 2 * margin)
    top = top_band_height(height)
    bottom = int(np.ceil(len(lines) * metrics.lineSpacing() + 2 * margin))
    rows = top + height + bottom
    frame = np.empty((rows + rows % 2, body.shape[1] + body.shape[1] % 2, 3), np.uint8)
    frame[:] = BAND_COLOR
    frame[top:top + height, :body.shape[1]] = body

    image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                   QImage.Format.Format_RGB888)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(*TEXT_COLOR))
    painter.drawText(QPointF(margin, margin + metrics.ascent()), f'{index}/{total}')
    # the caption, each line centered; the category in its color
    y = top + height + margin + metrics.ascent()
    for number, line in enumerate(lines):
        x = (body.shape[1] - metrics.horizontalAdvance(line)) / 2
        if number == 0 and line.startswith(category):
            painter.setPen(QColor(*(int(value) for value in colors[1])))
            painter.drawText(QPointF(x, y), category)
            x += metrics.horizontalAdvance(category)
            line = line[len(category):]
        painter.setPen(QColor(*TEXT_COLOR))
        painter.drawText(QPointF(x, y), line)
        y += metrics.lineSpacing()
    painter.end()
    return frame


# writing

def _read_image(file_path: Optional[str]) -> Optional[np.ndarray]:
    return None if file_path is None else np.array(Image.open(file_path).convert('RGB'))


def write_clip(clip: DemoClip, fps: int, bitrate: int, progress: Callable[[], None],
               cancelled: Callable[[], bool]) -> None:
    """The clip's MP4 (H.264, as gui/exporter.py writes them), written beside it first: a cancel
    or a failure leaves no partial video. progress() after each frame."""
    os.makedirs(path.dirname(clip.output), exist_ok=True)
    partial = clip.output[:-len('.mp4')] + '.partial.mp4'
    output = av.open(partial, mode='w')
    try:
        stream = None
        for number, frame in enumerate(clip.frames, 1):
            if cancelled():
                raise DemoCancelled()
            image = compose_frame([_read_image(file_path) for file_path in frame.images],
                                  read_mask(frame.mask, frame.pregenerated), clip.mode,
                                  clip.color, clip.category, clip.description, number,
                                  len(clip.frames), clip.layer)
            if stream is None:
                stream = output.add_stream('h264', rate=fps)
                stream.width, stream.height = image.shape[1], image.shape[0]
                stream.pix_fmt = 'yuv420p'
                stream.bit_rate = bitrate * (10**7)
            output.mux(stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')))
            progress()
        output.mux(stream.encode(None))
        output.close()
    except BaseException:
        output.close()
        if path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, clip.output)


def write_demos(clips: List[DemoClip], fps: int, bitrate: int,
                progress: Callable[[int, int], None], cancelled: Callable[[], bool]) -> List[str]:
    """Every clip of one sequence; then its older demos of the same mode (intervals since changed)
    are removed. progress(frames written, frames). Returns the videos."""
    total = sum(len(clip.frames) for clip in clips)
    written = [0]

    def advance() -> None:
        written[0] += 1
        progress(written[0], total)

    for clip in clips:
        write_clip(clip, fps, bitrate, advance, cancelled)
    outputs = [clip.output for clip in clips]
    for folder, sequence, mode in {(path.dirname(clip.output),
                                    path.basename(clip.output).rsplit('_', 2)[0], clip.mode)
                                   for clip in clips}:
        pattern = re.compile(rf'{re.escape(sequence)}_\d+-\d+_{re.escape(mode)}\.mp4')
        for name in os.listdir(folder):
            if pattern.fullmatch(name) and path.join(folder, name) not in outputs:
                os.remove(path.join(folder, name))
    return outputs


class DemoWorker(QObject):
    """Writes a task's demo videos in a thread of its own. on_progress(frames written, frames)
    and on_finished(the videos, else None, and the error: '' when cancelled) run on the GUI
    thread: the signals reach this object there, and the thread is joined before on_finished, so
    it never holds the last reference to a Qt object."""
    _progress = Signal(int, int)
    _finished = Signal(object, str)

    def __init__(self, clips: List[DemoClip], fps: int, bitrate: int,
                 on_progress: Callable[[int, int], None],
                 on_finished: Callable[[Optional[List[str]], str], None]):
        super().__init__()
        self.clips, self.fps, self.bitrate = clips, fps, bitrate
        self.on_progress, self.on_finished = on_progress, on_finished
        self.cancel = threading.Event()
        self._progress.connect(self._progressed)
        self._finished.connect(self._done)
        self.thread = threading.Thread(target=self._run, name='demo-video', daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """Cancels and waits for the thread (the window closes); on_finished is not called."""
        self.cancel.set()
        self.on_finished = lambda videos, error: None
        self.thread.join()

    def _run(self) -> None:
        try:
            videos, error = write_demos(self.clips, self.fps, self.bitrate, self._progress.emit,
                                        self.cancel.is_set), ''
        except DemoCancelled:
            videos, error = None, ''
        except Exception as exc:  # reported in the window
            log.exception('writing the demo videos failed')
            videos, error = None, str(exc) or type(exc).__name__
        self._finished.emit(videos, error)

    @Slot(int, int)
    def _progressed(self, written: int, total: int) -> None:
        self.on_progress(written, total)

    @Slot(object, str)
    def _done(self, videos: Optional[List[str]], error: str) -> None:
        self.thread.join()  # it has emitted: done with it
        self.on_finished(videos, error)
