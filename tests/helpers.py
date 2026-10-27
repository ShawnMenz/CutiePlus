"""
Synthetic RGB-T tasks for the tests: an input subset (gui/task_input.py: visual/ and thermal/
frames, categories.json, targets.json, annotation/ pregenerated masks) and the sequence's work
folder (gui/task_state.py: masks/ and metadata.json), with small frames of a textured
ellipse moving across a textured background. Everything lives under a temporary directory; the
real workspace is never touched.
"""
import json
import os
from dataclasses import dataclass, field
from os import path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from cutie.utils.palette import davis_palette

HEIGHT, WIDTH = 96, 128
# radii large enough that the target is not a "small target" (area >= 1500 px at 480 px)
RADII = (20, 27)


def ellipse_mask(frame_index: int, frame_count: int, height: int = HEIGHT,
                 width: int = WIDTH) -> np.ndarray:
    """uint8 index mask (0/1) of the target in a frame: it moves left to right."""
    ry, rx = RADII
    cx = rx + 4 + (width - 2 * rx - 8) * frame_index / max(1, frame_count - 1)
    cy = height / 2 + 6 * np.sin(frame_index / 3)
    yy, xx = np.mgrid[0:height, 0:width]
    return ((((yy - cy) / ry)**2 + ((xx - cx) / rx)**2) <= 1).astype(np.uint8)


def render_frames(frame_count: int, seed: int = 0, height: int = HEIGHT,
                  width: int = WIDTH) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(visual RGB frames, thermal frames as gray RGB), uint8 H*W*3."""
    rng = np.random.default_rng(seed)
    background = rng.integers(40, 120, size=(height, width, 3), dtype=np.uint8)
    texture = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    visual, thermal = [], []
    for index in range(frame_count):
        mask = ellipse_mask(index, frame_count, height, width).astype(bool)
        frame = background.copy()
        object_color = np.array([200, 60, 40], dtype=np.int16)
        frame[mask] = np.clip(object_color + texture[mask], 0, 255).astype(np.uint8)
        visual.append(frame)
        gray = np.full((height, width), 60, dtype=np.int16) + texture[..., 0] // 3
        gray[mask] = 210 + texture[..., 1][mask] // 4
        thermal.append(np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2))
    return visual, thermal


def save_palette_mask(file_path: str, mask: np.ndarray) -> None:
    image = Image.fromarray(mask.astype(np.uint8))
    image.putpalette(davis_palette)
    image.save(file_path)


def read_mask(file_path: str) -> np.ndarray:
    return np.array(Image.open(file_path))


def frame_name(index: int) -> str:
    return f'{index + 1:06d}'


@dataclass
class SyntheticTask:
    root: str  # the input root: <root>/<split>/{visual,thermal}/<sequence>
    split: str  # the subset (tab)
    sequence: str
    workspace: str  # the work folder: <base>/cutie_output/<split>/<sequence>
    frame_dirs: Dict[str, str]
    frame_count: int
    names: List[str] = field(default_factory=list)

    @property
    def masks_dir(self) -> str:
        return path.join(self.workspace, 'masks')

    def mask(self, name: str) -> Optional[np.ndarray]:
        file_path = path.join(self.masks_dir, name + '.png')
        return read_mask(file_path) if path.exists(file_path) else None

    def anchors(self) -> dict:
        from gui.task_state import ANCHORS, read_metadata
        return read_metadata(self.workspace)[ANCHORS]

    def progress(self) -> dict:
        from gui.task_state import PROGRESS, read_section
        return read_section(self.workspace, PROGRESS, {})


CATEGORY, CATEGORY_COLOR = 'car', '#00CF00'
DESCRIPTION = 'The red car moving right.'


def write_json(file_path: str, payload) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def make_task(base: str,
              sequence: str = '0001',
              split: str = 'main',
              frame_count: int = 16,
              mask_anchors: Tuple[int, ...] = (0, ),
              empty_anchors: Tuple[int, ...] = (),
              prefill: Tuple[int, ...] = (),
              height: int = HEIGHT,
              width: int = WIDTH,
              seed: int = 0,
              target: Optional[dict] = None,
              in_targets: bool = True) -> SyntheticTask:
    """One sequence in subset split of the input root base/tasks, and its work folder under
    base/cutie_output. Mask anchors get the true target mask; prefill frames get it as a pregenerated mask (annotation/). target: the sequence's
    entry in targets.json (default the red car); in_targets=False leaves it out of the file."""
    from gui.task_input import read_sequence
    from gui.task_state import ensure_work_folder, write_sections
    root = path.join(base, 'tasks')
    tab = path.join(root, split)
    os.makedirs(tab, exist_ok=True)
    visual, thermal = render_frames(frame_count, seed, height, width)
    frame_dirs = {}
    for modality, frames in (('visual', visual), ('thermal', thermal)):
        directory = path.join(tab, modality, sequence)
        os.makedirs(directory, exist_ok=True)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(path.join(directory, frame_name(index) + '.jpg'),
                                        quality=95)
        frame_dirs[modality] = directory
    names = [frame_name(i) for i in range(frame_count)]
    write_json(path.join(tab, 'categories.json'), [{'name': CATEGORY, 'color': CATEGORY_COLOR}])
    targets_path = path.join(tab, 'targets.json')
    targets = {}
    if path.exists(targets_path):
        with open(targets_path, encoding='utf-8') as f:
            targets = json.load(f)
    if in_targets:
        targets[sequence] = target or {'category': CATEGORY, 'description': DESCRIPTION}
    write_json(targets_path, targets)
    if prefill:
        annotation = path.join(tab, 'annotation', sequence)
        os.makedirs(annotation, exist_ok=True)
        for index in prefill:
            Image.fromarray(ellipse_mask(index, frame_count, height, width) * 255).save(
                path.join(annotation, names[index] + '.png'))

    workspace = ensure_work_folder(root, read_sequence(root, split, sequence))
    os.makedirs(path.join(workspace, 'masks'), exist_ok=True)
    anchors = {}
    for index in mask_anchors:
        save_palette_mask(path.join(workspace, 'masks', names[index] + '.png'),
                          ellipse_mask(index, frame_count, height, width))
        anchors[names[index]] = {'kind': 'mask'}
    for index in empty_anchors:
        anchors[names[index]] = {'kind': 'empty'}
    write_sections(workspace, anchors=dict(sorted(anchors.items())))
    return SyntheticTask(root, split, sequence, workspace, frame_dirs, frame_count, names)


def summary_of(workspace: str) -> dict:
    """The task browser's row of a work folder."""
    from gui.task_state import source_of, summarize_task
    return summarize_task(source_of(workspace), workspace)


def strip_times(anchors: dict) -> dict:
    """The anchors without their timestamps, for comparisons."""
    return {name: {k: v for k, v in entry.items() if k != 'updated_at'}
            for name, entry in anchors.items()}
