"""
What every labeling window's backend does with the task's files, whatever runs its models: the
masks (through the ResourceManager's save threads), anchors, intervals, description, category and
review marks (through TaskState). LocalTaskBackend adds models in the window's process; the
cooperation backends (service.py) send inference to the shared GPU worker instead.
"""
import os
from os import path
from typing import Callable, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from cutieplus.backends.base import BackendError, BinaryExport, FrameUpdate, Interval
from gui.edit_history import FrameState
from gui.exporter import convert_mask_to_binary
from gui.resource_manager import ResourceManager, SaveError, write_atomically
from gui.task_state import (TaskState, attention_of, discard_of, mark_done, submit_for_review,
                            task_status)


class TaskDataBackend:

    def __init__(self, cfg: DictConfig, res_man: ResourceManager, task: Optional[TaskState]):
        self.cfg = cfg
        self.res_man = res_man
        self.task = task
        self.resolution_message: Optional[str] = None

    # official frame data
    def commit_frame(self, ti: int, mask: np.ndarray, anchor: Optional[str]) -> FrameUpdate:
        """Saves the frame's mask; anchor 'edit' records a human edit, 'accept' accepts the shown
        mask as an anchor, None leaves the anchors alone."""
        self.res_man.save_mask(ti, mask)
        name = self.res_man.names[ti]
        if self.task is not None and anchor is not None:
            kind = 'mask' if (mask > 0).any() else 'empty'
            self.task.set_anchor(name, kind, edited=(anchor == 'edit'))
        return FrameUpdate(ti, mask, self._anchor(name))

    def remove_anchor(self, ti: int) -> FrameUpdate:
        name = self.res_man.names[ti]
        self.task.remove_anchor(name)
        self.res_man.delete_mask(ti)
        return FrameUpdate(ti, self.res_man.get_mask(ti), None)

    def restore_frame(self, ti: int, state: FrameState) -> FrameUpdate:
        self.res_man.save_queue.join()
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        if state.mask is None:
            if path.exists(mask_path):
                os.remove(mask_path)
        else:
            def write(temporary: str) -> None:
                with open(temporary, 'wb') as f:
                    f.write(state.mask)

            write_atomically(mask_path, write)
        self.res_man.invalidate(ti)
        if self.task is not None:
            self.task.restore_anchor(name, state.anchor)
        return FrameUpdate(ti, self.res_man.get_mask(ti), self._anchor(name))

    def frame_state(self, ti: int) -> FrameState:
        self.res_man.save_queue.join()
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        mask = None
        if path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                mask = f.read()
        anchor = self._anchor(name)
        return FrameState(mask, dict(anchor) if anchor is not None else None)

    def wait_for_saves(self) -> None:
        self.res_man.save_queue.join()

    def _anchor(self, name: str) -> Optional[dict]:
        return self.task.anchor(name) if self.task is not None else None

    # task data
    def set_intervals(self, intervals: List[Interval],
                      descriptions: Optional[Dict[Interval, str]] = None,
                      categories: Optional[Dict[Interval, str]] = None) -> None:
        self.task.save_intervals(intervals, descriptions, categories)
        self.task.touch()

    def set_description(self, text: str) -> None:
        self.task.save_description(text)
        self.task.touch()

    def set_category(self, name: str) -> None:
        self.task.save_category(name)
        self.task.touch()

    def set_attention(self, flagged: bool) -> None:
        self.task.save_attention(flagged)

    def export_binary(self, target_objects: List[int],
                      progress: Callable[[float], None]) -> Optional[BinaryExport]:
        """A plain workspace's binary PNGs of the masks, in binary_masks/. None when there are no
        masks."""
        self.res_man.save_queue.join()
        mask_folder = self.res_man.mask_dir
        save_folder = path.join(self.res_man.workspace, 'binary_masks')
        if not path.exists(mask_folder):
            return None
        os.makedirs(save_folder, exist_ok=True)
        convert_mask_to_binary(mask_folder, save_folder, target_objects,
                               progress_callback=progress)
        return BinaryExport(save_folder)

    def submit(self) -> List[Interval]:
        """Save (Ctrl+S) of a task: its masks and metadata are written as it is labeled; this
        waits for the last masks, keeps each added interval's description and category as they
        stand (they no longer follow the sequence's), and submits the task for review. Returns
        its intervals."""
        self.res_man.save_queue.join()
        try:
            self.res_man.check_saves()
        except SaveError as exc:
            raise BackendError(f'a mask was not saved ({exc})') from exc
        frame_count = self.res_man.T
        intervals, _ = self.task.intervals(frame_count)
        self.task.save_intervals(intervals, self.task.added_interval_descriptions(frame_count),
                                 self.task.added_interval_categories(frame_count))
        submit_for_review(self.res_man.workspace, self.task.labeler)
        return intervals

    def reset_all_masks(self) -> int:
        """Mask PNGs removed; raises OSError part way."""
        removed = self.res_man.clear_sequence_masks()
        if self.task is not None:
            self.task.clear_mask_references()
        return removed

    def mark_done(self, quit_at: Optional[str] = None) -> None:
        """Right after Ctrl+Q has submitted the task: any change since the submission puts the
        task back in progress, so a task still under review is marked done as it was saved."""
        workspace = self.res_man.workspace
        if discard_of(workspace):
            raise BackendError('the task is discarded: restore it first')
        if attention_of(workspace):
            raise BackendError('the task needs attention: toggle Need Attention off first')
        status = task_status(workspace)
        if status != 'under review':
            raise BackendError(f'the task is {status}, it changed after it was saved')
        mark_done(workspace, self.task.labeler, quit_at)
