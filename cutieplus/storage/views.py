"""
The legacy task files, kept by the coordinator as views of the committed state: masks/<frame>.png,
anchors.json, prefill.json, progress.json, interval.json, description.json. The export pipeline and
the task browser read them as before. Every write is durable and atomic, and its size and
modification time are recorded, so a later change by another program (the original labeling
window, while the service did not hold the task) is detected before the service writes again.
"""
import json
import os
from os import path
from typing import Dict, List, Optional

from cutieplus.storage.objects import write_file_durably
from gui.task_state import (ANCHORS_FILE, ATTENTION_FILE, DESCRIPTION_FILE, DISCARD_FILE,
                            INTERVAL_FILE, PREFILL_FILE, PROGRESS_FILE)

JSON_FILES = (ANCHORS_FILE, PREFILL_FILE, PROGRESS_FILE, INTERVAL_FILE, DESCRIPTION_FILE,
              ATTENTION_FILE, DISCARD_FILE)
MASKS_DIR = 'masks'

Stat = Optional[List[int]]  # [size, mtime_ns]; None: no file


def stat_of(file_path: str) -> Stat:
    try:
        info = os.stat(file_path)
    except FileNotFoundError:
        return None
    return [info.st_size, info.st_mtime_ns]


def json_bytes(payload) -> bytes:
    # the format of gui/task_state.py's _write_json
    return (json.dumps(payload, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


class TaskViews:

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.masks_dir = path.join(workspace, MASKS_DIR)

    def mask_path(self, frame: str) -> str:
        return path.join(self.masks_dir, frame + '.png')

    def write_mask(self, frame: str, data: Optional[bytes]) -> Stat:
        """The frame's mask file with these bytes, or no file (None)."""
        file_path = self.mask_path(frame)
        if data is None:
            if path.exists(file_path):
                os.remove(file_path)
            return None
        os.makedirs(self.masks_dir, exist_ok=True)
        write_file_durably(file_path, data, mode=0o644)
        return stat_of(file_path)

    def write_json(self, name: str, payload) -> Stat:
        """A JSON file of the task, or no file (None)."""
        file_path = path.join(self.workspace, name)
        if payload is None:
            if path.exists(file_path):
                os.remove(file_path)
            return None
        write_file_durably(file_path, json_bytes(payload), mode=0o644)
        return stat_of(file_path)

    def read_json(self, name: str, default=None):
        file_path = path.join(self.workspace, name)
        if not path.exists(file_path):
            return default
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)

    def scan_masks(self) -> Dict[str, List[int]]:
        """frame -> [size, mtime_ns] of every mask file."""
        stats = {}
        if not path.isdir(self.masks_dir):
            return stats
        with os.scandir(self.masks_dir) as entries:
            for entry in entries:
                if entry.name.endswith('.png') and not entry.name.startswith('.'):
                    info = entry.stat()
                    stats[entry.name[:-4]] = [info.st_size, info.st_mtime_ns]
        return stats

    def json_stats(self) -> Dict[str, Stat]:
        return {name: stat_of(path.join(self.workspace, name)) for name in JSON_FILES}

    def remove_temporary_files(self) -> int:
        removed = 0
        for directory in (self.workspace, self.masks_dir):
            if not path.isdir(directory):
                continue
            for name in os.listdir(directory):
                if name.startswith('.') and name.endswith('.tmp'):
                    os.remove(path.join(directory, name))
                    removed += 1
        return removed
