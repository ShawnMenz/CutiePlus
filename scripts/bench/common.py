"""
Shared pieces of the benchmark (scripts/benchmark_service.py): copies of real tasks, click plans,
statistics, and per-process GPU memory from nvidia-smi. Nothing here writes to the task root it
copies from; frames are read in place.
"""
import json
import os
import shutil
import subprocess
import time
from os import path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

FAKE_EXPORT = '''
import json, os, sys
from datetime import datetime, timezone
tree = sys.argv[1]
os.makedirs(os.path.join(tree, "masks"), exist_ok=True)
with open(os.path.join(tree, "metadata.json"), "w") as f:
    json.dump({"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
'''


def copy_task(source_root: str, split: str, sequence: str, target_root: str,
              target_split: str = None) -> str:
    """A copy of one task's files (task.json, anchors, prefill, intervals, description, and the
    masks of mask anchors only, so every run propagates afresh), under target_split when given
    (identical copies of one task). Returns the copy's workspace."""
    source = path.join(source_root, split, sequence)
    target = path.join(target_root, target_split or split, sequence)
    os.makedirs(path.join(target, 'masks'), exist_ok=True)
    for name in ('task.json', 'anchors.json', 'prefill.json', 'interval.json', 'description.json'):
        if path.exists(path.join(source, name)):
            shutil.copy2(path.join(source, name), path.join(target, name))
    if target_split:
        with open(path.join(target, 'task.json'), encoding='utf-8') as f:
            task = json.load(f)
        task['split'] = target_split
        with open(path.join(target, 'task.json'), 'w', encoding='utf-8') as f:
            json.dump(task, f, indent=2)
    with open(path.join(source, 'anchors.json'), encoding='utf-8') as f:
        anchors = json.load(f)
    for name, anchor in anchors.items():
        mask = path.join(source, 'masks', name + '.png')
        if anchor['kind'] == 'mask' and path.exists(mask):
            shutil.copy2(mask, path.join(target, 'masks', name + '.png'))
    return target


def write_task_set(target_root: str, splits: List[str], exports: str) -> None:
    script = path.join(target_root, 'fake_export.py')
    with open(script, 'w', encoding='utf-8') as f:
        f.write(FAKE_EXPORT)
    csv_path = path.join(target_root, 'curated.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('split,name\n')
    task_set = {'format': 1, 'splits': splits, 'curated_csv': csv_path,
                'export_tree': path.join(exports, 'cutie_masks_{modality}', '{sequence}'),
                'export_command': ['python3', script,
                                   path.join(exports, 'cutie_masks_visual', '{sequence}')]}
    with open(path.join(target_root, 'task_set.json'), 'w', encoding='utf-8') as f:
        json.dump(task_set, f, indent=2)


def prepare(source_root: str, tasks: List[Tuple[str, str, str]], target_root: str) -> Dict[str, str]:
    """A fresh task root with copies of the given (split, sequence, target split) tasks;
    {task id in the copy: workspace}."""
    if path.exists(target_root):
        shutil.rmtree(target_root)
    os.makedirs(target_root)
    workspaces = {f'{target}/{sequence}': copy_task(source_root, split, sequence, target_root, target)
                  for split, sequence, target in tasks}
    write_task_set(target_root, sorted({target for _, _, target in tasks}),
                   path.join(path.dirname(target_root), 'exports'))
    return workspaces


def task_summary(workspace: str) -> dict:
    with open(path.join(workspace, 'task.json'), encoding='utf-8') as f:
        task = json.load(f)
    with open(path.join(workspace, 'anchors.json'), encoding='utf-8') as f:
        anchors = json.load(f)
    names = [path.splitext(name)[0] for name in task['frame_files'][task['anchor_modality']]]
    mask_anchors = sorted(names.index(n) for n, a in anchors.items() if a['kind'] == 'mask'
                          and n in names)
    return {'frames': len(names), 'width': task['width'], 'height': task['height'],
            'anchor_modality': task['anchor_modality'], 'mask_anchors': len(mask_anchors),
            'empty_anchors': sum(1 for a in anchors.values() if a['kind'] == 'empty'),
            'first_mask_anchor': mask_anchors[0] if mask_anchors else None, 'names': names}


def click_plan(workspace: str, count: int) -> List[Tuple[int, List[Tuple[int, int, bool]]]]:
    """Clicks on the frames next to mask anchors: a positive click at the target's centroid, a
    negative one beside the target, another positive inside. [(frame index, clicks)]."""
    summary = task_summary(workspace)
    with open(path.join(workspace, 'anchors.json'), encoding='utf-8') as f:
        anchors = json.load(f)
    names = summary['names']
    plan = []
    for name, anchor in sorted(anchors.items()):
        if anchor['kind'] != 'mask' or name not in names:
            continue
        mask = np.array(Image.open(path.join(workspace, 'masks', name + '.png'))) > 0
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        height, width = mask.shape
        x0, x1 = xs.min(), xs.max()
        outside = (int(min(width - 1, x1 + max(4, (x1 - x0) // 4))), cy)
        inside = (int(xs[len(xs) // 3]), int(ys[len(ys) // 3]))
        ti = min(names.index(name) + 1, len(names) - 1)
        plan.append((ti, [(cx, cy, False), (outside[0], outside[1], True),
                          (inside[0], inside[1], False)]))
        if len(plan) * 3 >= count:
            break
    return plan


def stats(values: List[float]) -> dict:
    if not values:
        return {'count': 0}
    array = np.asarray(values, dtype=float)
    return {'count': int(array.size), 'mean': float(array.mean()), 'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)), 'max': float(array.max()), 'min': float(array.min())}


def gpu_processes() -> Dict[int, int]:
    """pid -> MiB used on the GPU (nvidia-smi), every compute process."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    usage = {}
    for line in out.strip().splitlines():
        try:
            pid, used = (int(part.strip()) for part in line.split(','))
            usage[pid] = used
        except ValueError:
            continue
    return usage


def gpu_device() -> dict:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,'
                          'driver_version', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    name, total, used, free, driver = [part.strip() for part in out.split(',')]
    return {'name': name, 'total_mib': int(total), 'used_mib': int(used), 'free_mib': int(free),
            'driver': driver}


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.start
