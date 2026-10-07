"""
Synthetic RGB-T tasks for the tests: the same files RGBT_MOT's tools/cutie_tasks.py writes
(task_set.json, task.json, anchors.json, prefill.json, masks/), with small frames of a textured
ellipse moving across a textured background. Everything lives under a temporary directory; the
real workspace is never touched.
"""
import json
import os
import sys
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
    root: str  # task root (task_set.json)
    split: str
    sequence: str
    workspace: str  # <root>/<split>/<sequence>
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
        with open(path.join(self.workspace, 'anchors.json'), encoding='utf-8') as f:
            return json.load(f)

    def progress(self) -> dict:
        progress_path = path.join(self.workspace, 'progress.json')
        if not path.exists(progress_path):
            return {}
        with open(progress_path, encoding='utf-8') as f:
            return json.load(f)


FAKE_EXPORT = '''
import json, os, sys
from datetime import datetime, timezone
tree = sys.argv[1]
os.makedirs(os.path.join(tree, "masks"), exist_ok=True)
with open(os.path.join(tree, "metadata.json"), "w") as f:
    json.dump({"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
print("exported", tree)
'''


def make_task(base: str,
              sequence: str = '0001',
              split: str = 'main',
              frame_count: int = 16,
              mask_anchors: Tuple[int, ...] = (0, ),
              empty_anchors: Tuple[int, ...] = (),
              prefill: Tuple[int, ...] = (),
              anchor_modality: str = 'visual',
              height: int = HEIGHT,
              width: int = WIDTH,
              seed: int = 0) -> SyntheticTask:
    """A task root under base with one task; frames under base/frames. Mask anchors get the true
    target mask (origin 'ls', like Label Studio imports)."""
    root = path.join(base, 'tasks')
    workspace = path.join(root, split, sequence)
    frame_root = path.join(base, 'frames')
    os.makedirs(workspace, exist_ok=True)
    visual, thermal = render_frames(frame_count, seed, height, width)
    frame_dirs = {}
    files = {}
    for modality, frames in (('visual', visual), ('thermal', thermal)):
        directory = path.join(frame_root, modality, sequence)
        os.makedirs(directory, exist_ok=True)
        files[modality] = []
        for index, frame in enumerate(frames):
            file_name = frame_name(index) + '.jpg'
            Image.fromarray(frame).save(path.join(directory, file_name), quality=95)
            files[modality].append(file_name)
        frame_dirs[modality] = directory
    names = [frame_name(i) for i in range(frame_count)]

    export_script = path.join(base, 'fake_export.py')
    with open(export_script, 'w', encoding='utf-8') as f:
        f.write(FAKE_EXPORT)
    csv_path = path.join(base, 'curated.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('split,name\n' + ''.join(f'{split},seq\n' for _ in range(int(sequence))))
    task_set = {
        'format': 1,
        'splits': [split],
        'curated_csv': csv_path,
        'export_tree': path.join(base, 'exports', 'cutie_masks_{modality}', '{sequence}'),
        'export_command': [sys.executable, export_script,
                           path.join(base, 'exports', 'cutie_masks_' + anchor_modality, '{sequence}')],
    }
    with open(path.join(root, 'task_set.json'), 'w', encoding='utf-8') as f:
        json.dump(task_set, f, indent=2)

    task = {
        'format': 1,
        'sequence_id': sequence,
        'split': split,
        'anchor_modality': anchor_modality,
        'frame_dirs': frame_dirs,
        'frame_count': frame_count,
        'frame_files': files,
        'width': width,
        'height': height,
        'category': 'car',
        'description': 'The red car moving right.',
        'curated_interval': [1, frame_count],
    }
    with open(path.join(workspace, 'task.json'), 'w', encoding='utf-8') as f:
        json.dump(task, f, indent=2)

    os.makedirs(path.join(workspace, 'masks'), exist_ok=True)
    anchors = {}
    for index in mask_anchors:
        save_palette_mask(path.join(workspace, 'masks', names[index] + '.png'),
                          ellipse_mask(index, frame_count, height, width))
        anchors[names[index]] = {'kind': 'mask', 'origin': 'ls', 'ls_file': f'label_{index}.json'}
    for index in empty_anchors:
        anchors[names[index]] = {'kind': 'empty', 'origin': 'ls'}
    with open(path.join(workspace, 'anchors.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(anchors.items())), f, indent=2)

    prefill_map = {}
    if prefill:
        prefill_dir = path.join(base, 'prefill', sequence)
        os.makedirs(prefill_dir, exist_ok=True)
        for index in prefill:
            file_path = path.join(prefill_dir, names[index] + '.png')
            Image.fromarray(ellipse_mask(index, frame_count, height, width) * 255).save(file_path)
            prefill_map[names[index]] = {'source': 'sam3', 'path': file_path}
    with open(path.join(workspace, 'prefill.json'), 'w', encoding='utf-8') as f:
        json.dump(prefill_map, f, indent=2)
    return SyntheticTask(root, split, sequence, workspace, frame_dirs, frame_count, names)


def strip_times(anchors: dict) -> dict:
    """anchors.json without its timestamps, for comparisons."""
    return {name: {k: v for k, v in entry.items() if k != 'updated_at'}
            for name, entry in anchors.items()}
