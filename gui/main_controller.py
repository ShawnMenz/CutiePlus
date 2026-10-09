import os
import time
from dataclasses import dataclass, field
from os import path
import logging
from typing import Callable, List, Literal, Optional, Tuple

import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import torch
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from cutieplus.backends.base import BackendError, ClickResult, TaskBackend
from cutieplus.backends.local import LocalTaskBackend
from cutieplus.engine import models

from gui.interactive_utils import *
from gui.resource_manager import ResourceManager, SaveError
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.exporter import convert_frames_to_video
from gui.edit_history import Edit, EditHistory, FrameState
from gui.task_export import start_export
from gui.task_state import (DESCRIPTION_TEXT, INTERVAL_TEXT, TASK_SET_FILE, TaskLock, TaskState,
                            check_intervals, export_command, format_intervals, labeler_name,
                            load_task_set, task_root_of)

log = logging.getLogger()


def load_networks(cfg: DictConfig) -> Tuple[torch.nn.Module, ClickController]:
    """(Cutie, the RITM click controller) on cfg's device, for the local mode."""
    return models.load_networks(cfg, cfg['device'])


@dataclass
class ClickState:
    """The window's view of the current click interaction; the backend holds its model state."""
    tar_obj: int
    clicks: List[Tuple[int, int, bool]] = field(default_factory=list)  # drawn as dots


class MainController():

    def __init__(self,
                 cfg: DictConfig,
                 networks: Optional[Tuple[torch.nn.Module, ClickController]] = None,
                 on_task_event: Optional[Callable[['MainController', str], None]] = None,
                 backend_factory: Optional[Callable[..., TaskBackend]] = None) -> None:
        """networks: models loaded by load_networks, to reuse them across tasks (local mode).
        on_task_event: called with 'exported' after an export, 'under review' once the task is
        submitted, 'attention' when Need Attention is toggled, and 'closed' once the window has
        closed and every pending save is written.
        backend_factory: the shared service's backend (cutieplus.backends.shared); without it
        the window runs the models itself (local mode).
        Raises TaskLocked when another window has the task open."""
        super().__init__()

        self.initialized = False
        self.on_task_event = on_task_event
        self.export_process = None
        self.export_id = None  # shared mode: the export the coordinator verifies
        self.browser_exporting = False
        self.close_requested = False
        self.closed = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # a task is open in one window at a time; taken before anything is loaded, so a refusal
        # leaves nothing to undo (shared mode: the service's lease, taken by the backend factory)
        self.task_lock = None
        self.backend: Optional[TaskBackend] = None
        init_only = 'workspace_init_only' in cfg and cfg['workspace_init_only']
        if TaskState.is_task(cfg['workspace']) and not init_only:
            # one name for the lock and the task's records
            labeler = labeler_name(cfg['labeler'] if 'labeler' in cfg else None,
                                   task_root_of(cfg['workspace']))
            with open_dict(cfg):
                cfg['labeler'] = labeler
            if backend_factory is None:
                self.task_lock = TaskLock.acquire(cfg['workspace'], labeler)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s): the local mode runs them in this process
        self.cutie = self.click_ctrl = None
        if backend_factory is None:
            if networks is None:
                self.initialize_networks()
            else:
                self.cutie, self.click_ctrl = networks

        # main components
        self.res_man = ResourceManager(cfg)
        self.task = self.res_man.task
        self.view_modality = self.res_man.default_modality
        self.propagation_modality = self.view_modality  # the frames the last run read
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        try:
            if backend_factory is None:
                self.backend = LocalTaskBackend(cfg, self.networks, self.res_man, self.task)
            else:
                self.backend = backend_factory(cfg, self.res_man, self.task)
        except BaseException:
            self.res_man.close()
            if self.task_lock is not None:
                self.task_lock.release()
            raise
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Optional[ClickState] = None
        self.history = EditHistory()
        self.interaction_before: Optional[FrameState] = None
        self.interaction_edit: Optional[Edit] = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'davis'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        min_mem_frames, max_mem_frames, max_long_tokens, mem_every = self.backend.memory_settings()
        self.gui.work_mem_min.setValue(min_mem_frames)
        self.gui.work_mem_max.setValue(max_mem_frames)
        self.gui.long_mem_max.setValue(max_long_tokens)
        self.gui.mem_every_box.setValue(mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        self.gui.show_fitted()
        self.gui.text('Initialized.')
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_object_color(self.curr_object)
        self.update_config()
        if self.task is not None:
            self.configure_task_resolution()

    def initialize_networks(self) -> None:
        self.cutie, self.click_ctrl = load_networks(self.cfg)

    @property
    def networks(self) -> Tuple[torch.nn.Module, ClickController]:
        return self.cutie, self.click_ctrl

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        # the next click starts an interaction for the new object
        self.reset_this_interaction()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.show_current_frame()

    def click_fn(self, action: Literal['left', 'right'], x: int, y: int):
        if self.propagating:
            return
        if action not in ['left', 'right']:
            raise NotImplementedError

        # left: positive click
        # right: negative click
        self.convert_current_image_mask_torch()
        try:
            if self.interaction is None or self.interaction.tar_obj != self.curr_object:
                # create new interaction is needed
                self.complete_interaction()
                self.backend.close_interaction()
                self.backend.open_interaction(self.curr_ti, self.view_modality, self.curr_image_np,
                                              self.curr_prob, self.curr_object)
                self.interaction = ClickState(self.curr_object)
                # the clicks of one interaction form one undoable edit
                self.interaction_before = self.frame_state(self.curr_ti)

            result = self.backend.add_click(x, y, is_negative=(action == 'right'))
        except BackendError as exc:
            self.gui.text(f'Click not applied: {exc}')
            self.reset_this_interaction()
            return
        self.interaction.clicks.append((x, y, action == 'right'))
        self.update_interacted_mask(result)
        self.update_gpu_gauges()

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti, self.view_modality)
        self.curr_image_torch = None

        if not no_mask:
            # a save still being written would be read half-finished
            self.res_man.save_queue.join()
            self.report_save_errors()
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.vis_target_objects)

    def update_canvas(self):
        # the clicks of the current interaction show as dots on the canvas only, never in files
        clicks = self.interaction.clicks if self.interaction is not None else ()
        self.gui.set_canvas(self.vis_image, clicks)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                 self.curr_prob, self.overlay_layer_torch,
                                                 self.vis_target_objects)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.set_frame_label(*self.res_man.frame_parts(self.curr_ti, self.view_modality))
        if not fast and self.task is not None:
            # every change of an anchor shows its frame again; a propagation (fast) changes none
            self.show_slider_anchors()

    def show_slider_anchors(self) -> None:
        # the anchors as lines on the frame slider
        name_to_ti = self.res_man.name_to_ti
        kinds = [(name_to_ti[name], anchor['kind']) for name, anchor in self.task.anchors.items()
                 if name in name_to_ti]
        self.gui.tl_slider.set_anchors([ti for ti, kind in kinds if kind == 'mask'],
                                       [ti for ti, kind in kinds if kind == 'empty'])

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def cycle_visualization_mode(self) -> None:
        modes = ('davis', 'mask', 'rgba', 'light')
        current = self.gui.combo.currentText()
        next_mode = modes[(modes.index(current) + 1) % len(modes)] if current in modes else modes[0]
        self.gui.combo.setCurrentText(next_mode)

    def save_current_mask(self):
        # save mask to hard disk
        self.backend.commit_frame(self.curr_ti, self.curr_mask, anchor=None)

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        if self.propagating:
            self.curr_ti = self.gui.tl_slider.value()
            return
        # save the frame being left while curr_ti still names it; saving after switching
        # wrote its mask over the new frame's
        if self.curr_frame_dirty:
            self.save_current_mask()
        self.curr_frame_dirty = False

        self.reset_this_interaction()
        self.curr_ti = self.gui.tl_slider.value()
        self.load_current_image_mask()
        self.show_current_frame()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()
        self.gui.refresh_reset_button_state()

    def on_propagate(self):
        # start to propagate
        start_anchor = self.current_anchor()
        if start_anchor is not None and start_anchor['kind'] == 'empty':
            self.gui.text('The target is marked absent here (empty anchor); '
                          'start from a frame that has a mask.')
            self.on_pause()
            return
        self.gui.stop_playback()

        self.convert_current_image_mask_torch()
        # the run reads the frames of the modality shown (T switches it; not during a run): the
        # anchors' masks fit both, which are aligned
        self.propagation_modality = self.view_modality
        try:
            self.backend.start_propagation(self.curr_ti, self.propagate_direction, self.curr_prob,
                                           self.propagation_modality)
        except BackendError as exc:
            self.gui.text(f'Propagation not started: {exc}')
            self.on_pause()
            return
        if self.task is not None and self.task.other_modality is not None:
            shown = 'RGB' if self.propagation_modality == 'visual' else self.propagation_modality
            self.gui.text(f'Propagating on the {shown} frames.')
        # clear
        self.interacted_prob = None
        self.reset_this_interaction()

        self.propagating = True
        self.gui.refresh_reset_button_state()
        self.gui.clear_all_mem_button.setEnabled(False)
        self.gui.clear_non_perm_mem_button.setEnabled(False)
        self.gui.tl_slider.setEnabled(False)

        # propagate till the end, a step at a time; the backend saves the propagated masks and
        # never touches an anchor's
        queued_shown = None
        try:
            while self.propagating:
                progress = self.backend.advance_propagation()
                for message in progress.messages:
                    self.gui.text(message)
                for frame in progress.frames:
                    self.show_propagated_frame(frame)
                if progress.stopped_at is not None:
                    # the run ends on the empty anchor that stopped it
                    self.curr_ti = progress.stopped_at
                    self.gui.tl_slider.setValue(self.curr_ti)
                if progress.queue_position is not None and progress.queue_position != queued_shown:
                    queued_shown = progress.queue_position
                    self.gui.text(f'Waiting for the GPU: {queued_shown} job(s) ahead.')
                if progress.done:
                    if progress.state == 'failed':
                        self.gui.text(f'Propagation failed: {progress.stop_reason}')
                    break
                self.update_memory_gauges()
                self.gui.process_events()
                if not progress.frames and self.backend.kind != 'local':
                    time.sleep(0.015)  # waiting for the service
        except BackendError as exc:
            self.gui.text(f'Propagation stopped: {exc}')
        except Exception as exc:  # e.g. out of GPU memory in local mode
            log.exception('Propagation failed')
            self.gui.text(f'Propagation failed: {exc}')
        finally:
            self.backend.cancel_propagation()

        self.propagating = False
        self.curr_frame_dirty = False
        if self.close_requested:
            # the window was closed mid-propagation
            self.finish_close()
            return
        self.on_pause()
        self.on_slider_update()
        self.gui.process_events()

    def show_propagated_frame(self, frame) -> None:
        if frame.kind == 'seed':
            # the start frame: shown, never saved
            self.curr_prob = frame.prob
            self.curr_mask = frame.mask
            if frame.prob is not None:
                self.convert_current_image_mask_torch(no_mask=True)
                # override this for #41
                self.show_current_frame(fast=True, invalid_soft_mask=True)
            else:
                self.show_current_frame()
            return
        # the frame comes from the backend, never from the slider: a key pressed mid-run moves
        # the slider, and the mask would be saved under that other frame
        self.curr_ti = frame.ti
        if frame.image is not None and self.view_modality == self.propagation_modality:
            self.curr_image_np = frame.image  # the frame the run read
        else:
            self.curr_image_np = self.res_man.get_image(self.curr_ti, self.view_modality)
        self.curr_image_torch = None
        self.gui.tl_slider.setValue(self.curr_ti)
        self.curr_mask = frame.mask
        if frame.prob is not None:
            self.curr_prob = frame.prob
            self.convert_current_image_mask_torch(no_mask=True)
            self.show_current_frame(fast=True)
        else:
            self.res_man.invalidate(self.curr_ti)
            self.curr_prob = None
            self.show_current_frame()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.task is not None:
            self.commit_anchor()
            return

        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()

        if self.backend.kind != 'local':
            self.gui.text('The permanent memory is rebuilt from the anchors of a task.')
            return
        self.convert_current_image_mask_torch()
        self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
        self.curr_prob = self.backend.commit_to_permanent_memory(self.curr_image_torch,
                                                                 self.curr_prob)
        self.update_memory_gauges()
        self.update_gpu_gauges()

    def on_window_closed(self) -> None:
        if self.propagating:
            # on_propagate finishes the close once its loop has stopped
            self.close_requested = True
            self.propagating = False
            return
        self.finish_close()

    def finish_close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.res_man.close()
        self.report_save_errors()
        # the window's signal connections keep this controller alive, so release its GPU and
        # frame memory here rather than waiting for it to be collected (the processor also
        # caches image features, which clear_memory keeps); shared mode: releases the lease
        self.curr_prob = self.curr_image_torch = self.interacted_prob = None
        self.overlay_layer_torch = None
        try:
            self.backend.close()
        except BackendError as exc:
            log.warning('Closing the task: %s', exc)
        # every save is written: another window may take the task now
        if self.task_lock is not None:
            self.task_lock.release()
        self.res_man.get_image.clear()
        self.res_man.get_mask.clear()
        self.gui.deleteLater()
        if self.on_task_event is not None:
            self.on_task_event(self, 'closed')

    def task_set(self) -> Optional[dict]:
        task_root = task_root_of(self.cfg['workspace'])
        if not path.isfile(path.join(task_root, TASK_SET_FILE)):
            self.gui.text(f'No {TASK_SET_FILE} in {task_root}; export with the pipeline tool.')
            return None
        return load_task_set(task_root)

    def on_export(self) -> None:
        # an ad-hoc export; it does not change the review status
        if (self.task is None or self.propagating or self.export_process is not None
                or self.browser_exporting):
            return
        task_set = self.task_set()
        if task_set is None:
            return
        self.backend.wait_for_saves()
        try:
            # shared mode: recorded with the content revision it exports
            self.export_id = self.backend.begin_export()
        except BackendError as exc:
            self.gui.text(f'Export not started: {exc}')
            return
        command = export_command(task_set, self.task.split, self.task.sequence_id)
        self.gui.text(f'Exporting {self.task.sequence_id}...')
        self.export_process = start_export(command, self.gui.text, self.on_export_finished)
        self.gui.refresh_reset_button_state()

    def on_export_finished(self, ok: bool) -> None:
        self.export_process = None
        result = self.backend.finish_export(self.export_id, ok)
        if ok and result['status'] != 'complete':
            self.gui.text(f'The export is not accepted ({result["status"]}): {result["detail"]}')
            ok = False
        self.gui.refresh_reset_button_state()
        self.gui.text('Export finished.' if ok else 'Export failed; see the messages above.')
        if ok and self.on_task_event is not None:
            self.on_task_event(self, 'exported')

    def current_anchor(self) -> dict:
        if self.task is None:
            return None
        return self.task.anchor(self.res_man.names[self.curr_ti])

    def commit_anchor(self) -> None:
        name = self.res_man.names[self.curr_ti]
        if self.interacted_prob is not None:
            # the clicks are saved already; the frame is an anchor since the first one
            self.complete_interaction()
        elif self.current_anchor() is not None:
            self.gui.text(f'{name} is already an anchor.')
            return
        else:
            before = self.frame_state(self.curr_ti)
            if not self.apply_commit(lambda: self.backend.commit_frame(
                    self.curr_ti, self.curr_mask, anchor='accept')):
                return
            self.record_edit(self.curr_ti, before)
        self.show_current_frame()
        self.gui.text(f'{name} is an anchor ({self.current_anchor()["kind"]}).')

    def apply_commit(self, commit: Callable[[], object]) -> bool:
        """Runs a change of the frame's official data; a refusal (shared mode: e.g. the task
        was lost) is shown and the frame reloaded. Returns whether it was applied."""
        try:
            commit()
            return True
        except BackendError as exc:
            self.gui.text(f'Not saved: {exc}')
            self.reset_this_interaction()
            self.res_man.invalidate(self.curr_ti)
            self.load_current_image_mask()
            self.show_current_frame()
            return False

    def configure_task_resolution(self) -> None:
        # stride-16 features lose small targets at the default size; the backend processes those
        # larger (decided when the task is opened)
        if self.backend.resolution_message:
            self.gui.text(self.backend.resolution_message)

    def on_remove_anchor(self) -> None:
        if self.task is None or self.propagating:
            return
        name = self.res_man.names[self.curr_ti]
        if self.current_anchor() is None:
            self.gui.text(f'{name} is not an anchor.')
            return
        before = self.frame_state(self.curr_ti)
        if not self.apply_commit(lambda: self.backend.remove_anchor(self.curr_ti)):
            return
        self.reset_this_interaction()
        self.record_edit(self.curr_ti, before)
        self.load_current_image_mask()
        self.show_current_frame()
        self.gui.text(f'Removed anchor {name}; it now shows {self.res_man.mask_source(self.curr_ti)}.')

    def on_next_anchor(self) -> None:
        self.jump_to_anchor(1)

    def on_prev_anchor(self) -> None:
        self.jump_to_anchor(-1)

    def jump_to_anchor(self, direction: int) -> None:
        if self.task is None or self.propagating:
            return
        anchor_tis = [self.res_man.name_to_ti[name] for name in self.task.anchors
                      if name in self.res_man.name_to_ti]
        ahead = [ti for ti in anchor_tis if (ti - self.curr_ti) * direction > 0]
        if not ahead:
            self.gui.text('No more anchors in that direction.')
            return
        self.gui.tl_slider.setValue(min(ahead) if direction > 0 else max(ahead))

    def on_toggle_modality(self) -> None:
        if self.task is None or self.task.other_modality is None or self.propagating:
            return
        anchor_modality = self.task.anchor_modality
        self.view_modality = (self.task.other_modality
                              if self.view_modality == anchor_modality else anchor_modality)
        # the click model is anchored to the previous image
        self.reset_this_interaction()
        self.curr_image_np = self.res_man.get_image(self.curr_ti, self.view_modality)
        self.curr_image_torch = None
        self.show_current_frame()
        self.gui.text(f'Showing {self.view_modality}.')

    def on_play_video_timer(self):
        # the video plays once and stops on its last frame; an interval preview (Shift+R) on the
        # interval's last frame
        if self.propagating:
            return
        if self.curr_ti >= self.gui.play_until:  # moved past it by hand, or a one-frame interval
            self.gui.stop_playback()
            return
        self.curr_ti += 1
        self.gui.tl_slider.setValue(self.curr_ti)
        if self.curr_ti >= self.gui.play_until:
            self.gui.stop_playback()

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_toggle_attention(self, flagged: bool) -> None:
        # Need Attention: the sequence needs further attention; the task browser shows it as
        # Attention and Mark done waits until it is off. Not a change of the task
        try:
            self.backend.set_attention(flagged)
        except BackendError as exc:
            self.gui.text(f'Need Attention not saved: {exc}')
        else:
            self.gui.text('Marked as needing attention.' if flagged else 'Attention cleared.')
            if self.on_task_event is not None:
                self.on_task_event(self, 'attention')
        self.gui.refresh_attention_button()  # the saved state, also after a failure

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        # (task mode: this also submits the task for review, whatever its status)
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        self.backend.wait_for_saves()
        if self.task is not None:
            # the intervals and the description are saved with every submission; a number (or
            # 'm') typed into an interval box and not confirmed yet counts (the Ctrl+S shortcut,
            # unlike a click on the button, does not move the focus out of the box)
            for pair in self.gui.interval_pairs:
                for box in pair.boxes():
                    box.interpretText()
                pair.order('start')
            self.save_interval_pairs()
            unsaved = [pair for pair in self.gui.interval_pairs if pair.entry() is None or pair.invalid]
            if unsaved:
                self.gui.text(f'{len(unsaved)} interval{"" if len(unsaved) == 1 else "s"} not saved '
                              '(not complete, or overlapping): the saved intervals are used.')
            self.on_description_change()
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        if path.exists(mask_folder):
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            try:
                export = self.backend.export_binary(self.vis_target_objects,
                                                    self.gui.progressbar_update)
            except BackendError as exc:
                self.gui.progressbar_update(0)
                self.gui.text(f'Binary masks not saved: {exc}')
                return
            save_folder = export.folder
            intervals = export.intervals
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
            if self.task is not None:
                if len(intervals) == 1:
                    self.gui.text(f'Interval {format_intervals(intervals)} saved to '
                                  f'{path.join(save_folder, INTERVAL_TEXT)}')
                else:
                    self.gui.text(f'Intervals {format_intervals(intervals)} ({len(intervals)} clips) '
                                  f'saved to {path.join(save_folder, INTERVAL_TEXT)}')
                self.gui.text(f'Description saved to {path.join(save_folder, DESCRIPTION_TEXT)}')
                self.gui.text(f'{self.task.sequence_id} is under review; the task browser\'s '
                              'Mark done accepts it, any further edit reopens it.')
                if self.on_task_event is not None:
                    self.on_task_event(self, 'under review')
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_interval_change(self, edited_box: Literal['start', 'end'], pair=None) -> None:
        # a box of the task window's intervals was confirmed (the first pair when none is named)
        if self.task is None:
            return
        (pair or self.gui.interval_pairs[0]).order(edited_box)
        self.save_interval_pairs()

    def on_remove_interval(self, pair) -> None:
        # the x of an added interval: removes it, or cancels it when it was never saved
        if self.task is None:
            return
        self.gui.remove_interval_pair(pair)
        if not self.save_interval_pairs():
            self.gui.text('Interval cancelled; nothing to save.')

    def save_interval_pairs(self) -> bool:
        """Saves the complete interval pairs when they changed and do not overlap; a pair with a
        blank box is still being typed and is left out. A new list of intervals is a change that
        needs a new review. Returns whether anything was saved."""
        pairs = self.gui.interval_pairs
        complete = [pair for pair in pairs if pair.entry() is not None]
        intervals = sorted(pair.entry() for pair in complete)
        problems = check_intervals(intervals, self.T)
        overlapping = {pair for pair in complete for other in complete if other is not pair
                       and pair.entry()[0] <= other.entry()[1]
                       and other.entry()[0] <= pair.entry()[1]}
        for pair in pairs:
            pair.set_invalid(pair in overlapping)
        if problems:
            self.gui.text('Intervals not saved: ' + ' '.join(problems)
                          + ' Change one, or remove it with ×.')
            return False
        if intervals == self.task.intervals(self.T)[0]:
            return False
        try:
            self.backend.set_intervals(intervals)
        except BackendError as exc:
            self.gui.text(f'Intervals not saved: {exc}')
            return False
        # the slider shows the saved intervals; Shift+R starts again from the first
        self.gui.tl_slider.set_intervals(intervals)
        self.gui.preview_index = 0
        if len(intervals) == 1:
            self.gui.text(f'Interval set to {format_intervals(intervals)}; the task needs a new '
                          'review.')
        else:
            self.gui.text(f'Intervals set to {format_intervals(intervals)}, one clip each; the '
                          'task needs a new review.')
        if self.on_task_event is not None:
            self.on_task_event(self, 'interval')
        return True

    def on_description_change(self) -> None:
        # the task window's description box; a new text is a change that needs a new review
        if self.task is None:
            return
        text = self.gui.description_edit.text().strip()
        if not text:
            self.gui.set_description(self.task.description)
            self.gui.text('The description cannot be empty; kept the previous one.')
            return
        if text == self.task.description:
            self.gui.set_description(text)  # drops stray whitespace
            return
        try:
            self.backend.set_description(text)
        except BackendError as exc:
            self.gui.set_description(self.task.description)
            self.gui.text(f'Description not saved: {exc}')
            return
        self.gui.set_description(text)
        self.gui.text('Description changed; the task needs a new review.')
        if self.on_task_event is not None:
            self.on_task_event(self, 'description')

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self, result: ClickResult):
        # the backend has saved the mask and made the frame an anchor
        self.interacted_prob = result.prob
        self.curr_prob = self.interacted_prob
        self.curr_mask = result.update.mask
        self.record_interaction_edit()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.backend is not None:
            self.backend.close_interaction()

    def on_reset_mask(self):
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        before = self.frame_state(self.curr_ti)
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        if not self.apply_commit(lambda: self.backend.commit_frame(
                self.curr_ti, self.curr_mask, anchor='edit')):
            return
        self.reset_this_interaction()
        self.record_edit(self.curr_ti, before)
        self.show_current_frame()

    def on_reset_object(self):
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        before = self.frame_state(self.curr_ti)
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        if not self.apply_commit(lambda: self.backend.commit_frame(
                self.curr_ti, self.curr_mask, anchor='edit')):
            return
        self.reset_this_interaction()
        self.record_edit(self.curr_ti, before)
        self.show_current_frame()

    def reset_block_reason(self) -> Optional[str]:
        if self.closed:
            return 'The window is closing.'
        if getattr(self, 'propagating', False):
            return 'Pause propagation before resetting all masks.'
        if self.export_process is not None or self.browser_exporting:
            return 'Wait for the task export to finish before resetting all masks.'
        return self.task.reset_block_reason() if self.task is not None else None

    def on_reset_all_frame_masks(self) -> None:
        reason = self.reset_block_reason()
        if reason is not None:
            self.gui.text(reason)
            self.gui.refresh_reset_button_state()
            return
        if not self.gui.confirm_reset_all_masks():
            return
        # The curated CSV or export state can change while the dialog is open.
        reason = self.reset_block_reason()
        if reason is not None:
            self.gui.text(reason)
            self.gui.refresh_reset_button_state()
            return
        try:
            export_dirs = self.task.export_directories() if self.task is not None else []
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.gui.text(f'Cannot locate the Cutie exports; no masks removed: {exc}')
            return

        self.backend.wait_for_saves()
        reason = self.reset_block_reason()
        if reason is not None:
            self.gui.text(reason)
            self.gui.refresh_reset_button_state()
            return
        self.gui.stop_playback()
        self.curr_frame_dirty = False
        self.reset_this_interaction()
        self.interacted_prob = None
        reset_error = None
        try:
            removed, removed_exports = self.backend.reset_all_masks(export_dirs)
        except (OSError, BackendError) as exc:
            reset_error = exc
        finally:
            self.res_man.get_mask.clear()
            self.history = EditHistory()  # a sequence-wide deletion is not a frame-level undo
            self.on_clear_memory()
            self.load_current_image_mask()
            self.show_current_frame()
            if self.on_task_event is not None and self.task is not None:
                self.on_task_event(self, 'reset')
            self.gui.refresh_reset_button_state()
        if reset_error is not None:
            self.gui.text(f'Mask reset stopped after a filesystem error; check the workspace: '
                          f'{reset_error}')
            return
        self.gui.text(f'Reset all frame masks: removed {removed} local PNGs and '
                      f'{removed_exports} Cutie export trees for this sequence.')

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None
        self.interaction_before = None
        self.interaction_edit = None

    # undo/redo of frame edits (see gui/edit_history.py)
    def frame_state(self, ti: int) -> FrameState:
        state = self.backend.frame_state(ti)
        self.report_save_errors()
        return state

    def record_edit(self, ti: int, before: FrameState) -> None:
        self.history.record(Edit(self.res_man.names[ti], before, self.frame_state(ti)))

    def record_interaction_edit(self) -> None:
        if self.interaction_before is None:
            return
        after = self.frame_state(self.curr_ti)
        if self.interaction_edit is None:
            self.interaction_edit = Edit(self.res_man.names[self.curr_ti], self.interaction_before,
                                         after)
            self.history.record(self.interaction_edit)
        else:
            self.interaction_edit.after = after
            self.history.redo_stack.clear()

    def interaction_clicks(self) -> int:
        if self.interaction is None:
            return 0
        return self.backend.interaction_clicks()

    def on_undo(self) -> None:
        if self.propagating:
            return
        if self.interaction_clicks() > 1:
            # take back the last click only
            try:
                result = self.backend.undo_click()
            except BackendError as exc:
                self.gui.text(f'Undo not applied: {exc}')
                self.reset_this_interaction()
                return
            self.interaction.clicks.pop()
            self.update_interacted_mask(result)
            self.gui.text(f'Undid the last click ({self.interaction_clicks()} left).')
            return
        edit = self.history.undo()
        if edit is None:
            self.gui.text('Nothing to undo.')
            return
        self.apply_frame_state(edit.frame, edit.before)
        self.gui.text(f'Undid the edit of {edit.frame}.')

    def on_redo(self) -> None:
        if self.propagating:
            return
        edit = self.history.redo()
        if edit is None:
            self.gui.text('Nothing to redo.')
            return
        self.apply_frame_state(edit.frame, edit.after)
        self.gui.text(f'Redid the edit of {edit.frame}.')

    def apply_frame_state(self, name: str, state: FrameState) -> None:
        self.reset_this_interaction()
        ti = self.res_man.name_to_ti[name]
        if not self.apply_commit(lambda: self.backend.restore_frame(ti, state)):
            return
        # the restored file is newer than whatever the window holds; do not save over it
        self.curr_frame_dirty = False
        if ti != self.curr_ti:
            self.gui.tl_slider.setValue(ti)
        else:
            self.load_current_image_mask()
            self.show_current_frame()

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        gauges = self.backend.gpu_gauges()
        if gauges.used is not None and gauges.total:
            self.gui.gpu_mem_gauge.setFormat(f'{gauges.used:.1f} GB / {gauges.total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(gauges.used / gauges.total * 100))
        elif gauges.used is not None:
            self.gui.gpu_mem_gauge.setFormat(f'{gauges.used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
        if gauges.torch_used is not None and gauges.total:
            self.gui.torch_mem_gauge.setFormat(f'{gauges.torch_used:.1f} GB / {gauges.total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(gauges.torch_used / gauges.total * 100))
        else:
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def report_save_errors(self) -> None:
        # a failed background save is shown, never dropped silently
        try:
            self.res_man.check_saves()
        except SaveError as exc:
            log.error('Mask save failed: %s', exc)
            self.gui.text(f'SAVING FAILED, the frame keeps its previous mask file: {exc}')

    def on_gpu_timer(self):
        self.report_save_errors()
        self.update_gpu_gauges()
        self.gui.refresh_reset_button_state()

    def update_memory_gauges(self):
        gauges = self.backend.memory_gauges()
        if gauges is not None:
            curr_perm_tokens, _ = gauges['permanent']
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            curr_work_tokens, max_work_tokens = gauges['working']
            curr_long_tokens, max_long_tokens = gauges['long_term']

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max(1, max_work_tokens) * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max(1, max_long_tokens) * 100))

        else:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.backend.update_config(self.cfg)

    def on_clear_memory(self):
        self.backend.clear_memory(permanent=True)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.backend.clear_memory(permanent=False)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            before = self.frame_state(self.curr_ti)
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            if not self.apply_commit(lambda: self.backend.commit_frame(
                    self.curr_ti, self.curr_mask, anchor='edit')):
                return
            self.record_edit(self.curr_ti, before)
            self.show_current_frame()

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_set_save_visualization_mode(self):
        self.save_visualization_mode = self.gui.save_visualization_combo.currentText()

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x, y):
        self.last_ex = x
        self.last_ey = y

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
