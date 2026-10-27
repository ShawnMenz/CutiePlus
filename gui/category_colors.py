"""
The colors of the categories (categories.json, see gui/task_input.py): each category has one of its
own, as far as can be from every other one in CIELAB, at a lightness and saturation that read as a
pill or a mask overlay on the light and the dark theme, none gray (disabled). Colors already given
are kept; only the missing ones are picked. Used by the Import window, which completes a
categories.json that lists names only, and by scripts/extract_targets.py. No Qt.
"""
import json
import os
from os import path
from typing import Dict, List, Tuple

import numpy as np

MIN_LIGHTNESS, MAX_LIGHTNESS = 35.0, 75.0  # CIELAB L*
MIN_CHROMA = 30.0  # CIELAB C*
GRID_LEVELS = 33  # candidate values per RGB channel


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """N * 3 sRGB values in [0, 1] -> N * 3 CIELAB (D65)."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)**2.4)
    xyz = linear @ np.array([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]]).T
    t = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])],
                    axis=1)


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    text = color.lstrip('#')
    if len(text) != 6:
        raise ValueError(f'{color!r} is not #RRGGBB')
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def lab_of(colors: List[str]) -> np.ndarray:
    return srgb_to_lab(np.array([hex_to_rgb(c) for c in colors], dtype=float).reshape(-1, 3) / 255)


def candidate_colors() -> Tuple[np.ndarray, np.ndarray]:
    """(N * 3 RGB 0-255, N * 3 CIELAB): an RGB grid within the lightness and saturation limits."""
    levels = np.linspace(0, 255, GRID_LEVELS).round()
    rgb = np.stack(np.meshgrid(levels, levels, levels, indexing='ij'), axis=-1).reshape(-1, 3)
    lab = srgb_to_lab(rgb / 255)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = (lab[:, 0] >= MIN_LIGHTNESS) & (lab[:, 0] <= MAX_LIGHTNESS) & (chroma >= MIN_CHROMA)
    return rgb[keep], lab[keep]


def assign_colors(names_by_use: List[str], kept: Dict[str, str]) -> Dict[str, str]:
    """Name -> #RRGGBB: the kept colors stay; every other name, in the order given, gets the
    candidate farthest from all the colors already taken (the first one taken without any: the
    most saturated)."""
    colors = dict(kept)
    new = [name for name in names_by_use if name not in colors]
    if not new:
        return colors
    rgb, lab = candidate_colors()
    if colors:
        taken = lab_of(list(colors.values()))
        nearest = np.min(np.linalg.norm(lab[:, None, :] - taken[None, :, :], axis=2), axis=1)
    else:
        nearest = None
    for name in new:
        index = int(np.argmax(np.hypot(lab[:, 1], lab[:, 2]) if nearest is None else nearest))
        colors[name] = '#{:02X}{:02X}{:02X}'.format(*(int(v) for v in rgb[index]))
        distance = np.linalg.norm(lab - lab[index], axis=1)
        nearest = distance if nearest is None else np.minimum(nearest, distance)
    return colors


def categories_text(colors: Dict[str, str]) -> str:
    # categories.json: one category per line, alphabetically
    rows = [json.dumps({'name': name, 'color': colors[name]}, ensure_ascii=False)
            for name in sorted(colors)]
    return '[\n' + ',\n'.join(f'  {row}' for row in rows) + '\n]\n'


def write_text(file_path: str, text: str) -> None:
    """Writes a temporary file and renames it, so a failure leaves the previous file whole."""
    temporary = path.join(path.dirname(file_path), f'.{path.basename(file_path)}.{os.getpid()}.tmp')
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, file_path)
    except BaseException:
        if path.exists(temporary):
            os.remove(temporary)
        raise
