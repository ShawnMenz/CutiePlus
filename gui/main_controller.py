import os
from os import path
import logging
import math
import shutil
from typing import Callable, Literal, Optional, Tuple

import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.edit_history import Edit, EditHistory, FrameState
from gui.task_export import start_export
from gui.task_state import (DESCRIPTION_TEXT, INTERVAL_TEXT, TASK_SET_FILE, TaskLock, TaskState,
                            export_command, labeler_name, load_task_set, submit_for_review,
                            task_root_of, write_description_text, write_interval_text)
from cutie.utils.download_models import download_models_if_needed

log = logging.getLogger()

# Task mode: a target whose area at the default internal size is below SMALL_TARGET_AREA pixels
# is processed at SMALL_TARGET_INTERNAL_SIZE instead (on RGBT sequences: a 78 px car went from
# mean IoU 0.48 to 0.64 against SAM3; targets of 2.5k-12k px were unchanged, at ~2.5x the time).
SMALL_TARGET_AREA = 1500
SMALL_TARGET_INTERNAL_SIZE = 960
TARGET_AREA_SAMPLES = 25
# memory tokens x query tokens one propagation step may compare (~8 GB of GPU memory); it caps
# how many anchors can sit in the permanent memory at once
MEMORY_READ_BUDGET = 4e8


def load_networks(cfg: DictConfig) -> Tuple[CUTIE, ClickController]:
    weight_dir = download_models_if_needed()
    # relative weight paths in the config refer to the downloaded weights,
    # so the GUI works regardless of the working directory
    weights, ritm_weights = cfg.weights, cfg.ritm_weights
    if not path.isabs(weights):
        weights = path.join(weight_dir, path.basename(weights))
    if not path.isabs(ritm_weights):
        ritm_weights = path.join(weight_dir, path.basename(ritm_weights))

    cutie = CUTIE(cfg).eval().to(cfg['device'])
    model_weights = torch.load(weights, map_location=cfg['device'], weights_only=True)
    cutie.load_weights(model_weights)
    return cutie, ClickController(ritm_weights, device=cfg['device'])


class MainController():

    def __init__(self,
                 cfg: DictConfig,
                 networks: Optional[Tuple[CUTIE, ClickController]] = None,
                 on_task_event: Optional[Callable[['MainController', str], None]] = None) -> None:
        """networks: models loaded by load_networks, to reuse them across tasks.
        on_task_event: called with 'exported' after an export, 'under review' once the task is
        submitted, and 'closed' once the window has closed and every pending save is written.
        Raises TaskLocked when another window has the task open."""
        super().__init__()

        self.initialized = False
        self.on_task_event = on_task_event
        self.export_process = None
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
        # leaves nothing to undo
        self.task_lock = None
        init_only = 'workspace_init_only' in cfg and cfg['workspace_init_only']
        if TaskState.is_task(cfg['workspace']) and not init_only:
            # one name for the lock and the task's records
            labeler = labeler_name(cfg['labeler'] if 'labeler' in cfg else None,
                                   task_root_of(cfg['workspace']))
            with open_dict(cfg):
                cfg['labeler'] = labeler
            self.task_lock = TaskLock.acquire(cfg['workspace'], labeler)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s)
        if networks is None:
            self.initialize_networks()
        else:
            self.cutie, self.click_ctrl = networks
            self.click_ctrl.unanchor()

        # main components
        self.res_man = ResourceManager(cfg)
        self.task = self.res_man.task
        self.view_modality = self.res_man.default_modality
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
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
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

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
    def networks(self) -> Tuple[CUTIE, ClickController]:
        return self.cutie, self.click_ctrl

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.show_current_frame()

    def click_fn(self, action: Literal['left', 'right'], x: int, y: int):
        if self.propagating:
            return

        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                # left: positive click
                # right: negative click
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    # create new interaction is needed
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                       self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction
                        # the clicks of one interaction form one undoable edit
                        self.interaction_before = self.frame_state(self.curr_ti)

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

            else:
                raise NotImplementedError

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti, self.view_modality)
        self.curr_image_torch = None

        if not no_mask:
            # a save still being written would be read half-finished
            self.res_man.save_queue.join()
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
        clicks = self.interaction.clicks if isinstance(self.interaction, ClickInteraction) else ()
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
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

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

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if self.task is not None:
                self.rebuild_permanent_memory()
            self.convert_current_image_mask_torch()
            model_image = self.model_image_torch(self.curr_ti)

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(model_image, self.curr_prob[1:], idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.refresh_reset_button_state()
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            start_ti = self.curr_ti
            step = 1 if self.propagate_direction == 'forward' else -1
            dataset = PropagationReader(self.res_man, start_ti, self.propagate_direction,
                                        self.view_modality)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end
            masks_written = 0
            for index, data in enumerate(loader, 1):
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch, model_image = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                model_image = model_image.to(self.device, non_blocking=True)
                # the frame comes from the reader, never from the slider: an arrow key pressed
                # mid-run moves the slider, and the mask would be saved under that other frame
                self.curr_ti = start_ti + step * index
                self.gui.tl_slider.setValue(self.curr_ti)

                anchor = self.current_anchor()
                if anchor is not None and anchor['kind'] == 'empty':
                    self.gui.text(f'Stopped at t={self.curr_ti}: the target is marked absent '
                                  'here (empty anchor).')
                    break
                if anchor is not None:
                    # a human mask: feed it to the model and leave its file untouched
                    self.curr_mask = self.res_man.get_mask(self.curr_ti).copy()
                    anchor_prob = index_numpy_to_one_hot_torch(self.curr_mask,
                                                               self.num_objects + 1).to(self.device)
                    self.curr_prob = self.processor.step(model_image,
                                                         anchor_prob[1:],
                                                         idx_mask=False)
                else:
                    self.curr_prob = self.processor.step(model_image)
                    self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
                    self.save_current_mask()
                    masks_written += 1
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            if self.task is not None and masks_written:
                self.task.touch()
            if self.close_requested:
                # the window was closed mid-propagation
                self.finish_close()
                return
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

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
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 force_permanent=True)
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
        # every save is written: another window may take the task now
        if self.task_lock is not None:
            self.task_lock.release()
        # the window's signal connections keep this controller alive, so release its GPU and
        # frame memory here rather than waiting for it to be collected (the processor also
        # caches image features, which clear_memory keeps)
        self.processor = None
        self.curr_prob = self.curr_image_torch = self.interacted_prob = None
        self.overlay_layer_torch = None
        self.res_man.get_image.clear()
        self.res_man.get_mask.clear()
        self.gui.deleteLater()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
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
        self.res_man.save_queue.join()
        command = export_command(task_set, self.task.split, self.task.sequence_id)
        self.gui.text(f'Exporting {self.task.sequence_id}...')
        self.export_process = start_export(command, self.gui.text, self.on_export_finished)
        self.gui.refresh_reset_button_state()

    def on_export_finished(self, ok: bool) -> None:
        self.export_process = None
        self.gui.refresh_reset_button_state()
        self.gui.text('Export finished.' if ok else 'Export failed; see the messages above.')
        if ok and self.on_task_event is not None:
            self.on_task_event(self, 'exported')

    def model_image_torch(self, ti: int) -> torch.Tensor:
        # the model always sees the default (task: anchor) modality
        if ti == self.curr_ti and self.view_modality == self.res_man.default_modality:
            self.convert_current_image_mask_torch(no_mask=True)
            return self.curr_image_torch
        return to_tensor(self.res_man.get_image(ti)).to(self.device, non_blocking=True)

    def current_anchor(self) -> dict:
        if self.task is None:
            return None
        return self.task.anchor(self.res_man.names[self.curr_ti])

    def mark_anchor(self, edited: bool = True) -> None:
        # every human decision on a task frame makes it an anchor, like a Label Studio submit
        if self.task is None:
            return
        kind = 'mask' if (self.curr_mask > 0).any() else 'empty'
        self.task.set_anchor(self.res_man.names[self.curr_ti], kind, edited)

    def commit_anchor(self) -> None:
        name = self.res_man.names[self.curr_ti]
        if self.interacted_prob is not None:
            self.complete_interaction()
            self.update_interacted_mask()
        elif self.current_anchor() is not None:
            self.gui.text(f'{name} is already an anchor.')
            return
        else:
            before = self.frame_state(self.curr_ti)
            self.save_current_mask()
            self.mark_anchor(edited=False)
            self.record_edit(self.curr_ti, before)
        self.show_current_frame()
        self.gui.text(f'{name} is an anchor ({self.current_anchor()["kind"]}).')

    def median_target_area(self) -> Optional[int]:
        # from the mask anchors, else from a sample of the prefill masks
        names = [name for name, anchor in self.task.anchors.items() if anchor['kind'] == 'mask']
        if not names:
            prefill = sorted(self.task.prefill)
            names = prefill[::max(1, len(prefill) // TARGET_AREA_SAMPLES)][:TARGET_AREA_SAMPLES]
        areas = []
        for name in names:
            mask = self.res_man.get_mask(self.res_man.name_to_ti[name]) if name in self.res_man.name_to_ti else None
            if mask is not None and mask.any():
                areas.append(int((mask > 0).sum()))
        return int(np.median(areas)) if areas else None

    def configure_task_resolution(self) -> None:
        # stride-16 features lose small targets at the default size; process those larger
        area = self.median_target_area()
        default_size = self.processor.max_internal_size
        if area is None or default_size <= 0:
            return
        scale = min(1.0, default_size / min(self.h, self.w))
        if area * scale**2 < SMALL_TARGET_AREA:
            self.processor.max_internal_size = SMALL_TARGET_INTERNAL_SIZE
            self.processor.upscale_to_internal_size = True
            self.gui.text(f'Small target (median {area} px): processing at '
                          f'{SMALL_TARGET_INTERNAL_SIZE} px.')
        else:
            self.gui.text(f'Target median {area} px: processing at {default_size} px.')

    def memory_tokens_per_frame(self) -> int:
        h, w = self.h, self.w
        size = self.processor.max_internal_size
        min_side = min(h, w)
        if size > 0 and (min_side > size or
                         (self.processor.upscale_to_internal_size and min_side < size)):
            h, w = int(h / min_side * size), int(w / min_side * size)
        return math.ceil(h / 16) * math.ceil(w / 16)

    def permanent_anchor_limit(self) -> int:
        tokens = self.memory_tokens_per_frame()
        memory = self.processor.memory
        other_tokens = (memory.max_mem_frames + 1) * tokens
        if memory.use_long_term:
            other_tokens += memory.max_long_tokens
        return max(1, int((MEMORY_READ_BUDGET / tokens - other_tokens) // tokens))

    def rebuild_permanent_memory(self) -> None:
        # mask anchors enter the permanent memory, so each one guides every frame; beyond the
        # GPU budget only those nearest to the start do, the rest are fed when reached
        self.res_man.save_queue.join()
        self.processor.clear_memory()
        self.processor.update_config(self.cfg)
        all_anchor_tis = sorted(self.res_man.name_to_ti[name]
                                for name, anchor in self.task.anchors.items()
                                if anchor['kind'] == 'mask' and name in self.res_man.name_to_ti)
        nearest = sorted(all_anchor_tis, key=lambda ti: abs(ti - self.curr_ti))
        anchor_tis = sorted(nearest[:self.permanent_anchor_limit()])
        for ti in anchor_tis:
            prob = index_numpy_to_one_hot_torch(self.res_man.get_mask(ti),
                                                self.num_objects + 1).to(self.device)
            self.processor.step(self.model_image_torch(ti),
                                prob[1:],
                                idx_mask=False,
                                force_permanent=True)
        self.processor.clear_sensory_memory()
        self.gui.text(f'Permanent memory rebuilt from {len(anchor_tis)} of '
                      f'{len(all_anchor_tis)} mask anchors.')

    def on_remove_anchor(self) -> None:
        if self.task is None or self.propagating:
            return
        name = self.res_man.names[self.curr_ti]
        if self.current_anchor() is None:
            self.gui.text(f'{name} is not an anchor.')
            return
        before = self.frame_state(self.curr_ti)
        self.task.remove_anchor(name)
        self.res_man.delete_mask(self.curr_ti)
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
        if self.propagating:
            return
        self.curr_ti += 1
        self.gui.tl_slider.setValue(self.curr_ti)
        if self.curr_ti >= self.T - 1:
            # the video plays once: it stops on its last frame
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

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        # (task mode: this also submits the task for review, whatever its status)
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        self.res_man.save_queue.join()
        if self.task is not None:
            # the interval and the description are saved with every submission; a number (or 'm')
            # typed into an interval box and not confirmed yet counts (the Ctrl+S shortcut, unlike
            # a click on the button, does not move the focus out of the box)
            self.gui.interval_start.interpretText()
            self.gui.interval_end.interpretText()
            self.on_interval_change('start')
            self.on_description_change()
            interval_start, interval_end, _ = self.task.interval(self.T)
            self.task.save_interval(interval_start, interval_end)
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
            if self.task is not None:
                write_interval_text(save_folder, interval_start, interval_end)
                self.gui.text(f'Interval {interval_start}-{interval_end} saved to '
                              f'{path.join(save_folder, INTERVAL_TEXT)}')
                write_description_text(save_folder, self.task.description)
                self.gui.text(f'Description saved to {path.join(save_folder, DESCRIPTION_TEXT)}')
                submit_for_review(self.cfg['workspace'], self.task.labeler)
                self.gui.text(f'{self.task.sequence_id} is under review; the task browser\'s '
                              'Mark done accepts it, any further edit reopens it.')
                if self.on_task_event is not None:
                    self.on_task_event(self, 'under review')
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_interval_change(self, edited_box: Literal['start', 'end']) -> None:
        # the task window's interval boxes; a new interval is a change that needs a new review
        if self.task is None:
            return
        start, end = self.gui.interval_start.value(), self.gui.interval_end.value()
        if start > end:
            if edited_box == 'start':
                end = start
            else:
                start = end
            self.gui.interval_start.setValue(start)
            self.gui.interval_end.setValue(end)
        if (start, end) == self.task.interval(self.T)[:2]:
            return
        self.task.save_interval(start, end)
        self.task.touch()
        self.gui.text(f'Interval set to {start}-{end}; the task needs a new review.')
        if self.on_task_event is not None:
            self.on_task_event(self, 'interval')

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
        self.task.save_description(text)
        self.task.touch()
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

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.mark_anchor()
        self.record_interaction_edit()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        if self.propagating:
            self.gui.text('Pause the propagation first.')
            return
        before = self.frame_state(self.curr_ti)
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.save_current_mask()
        self.mark_anchor()
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
        self.save_current_mask()
        self.mark_anchor()
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

        self.res_man.save_queue.join()
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
            removed = self.res_man.clear_sequence_masks()
            removed_exports = 0
            if self.task is not None:
                self.task.clear_mask_references()
                for directory in export_dirs:
                    if path.isdir(directory):
                        shutil.rmtree(directory)
                        removed_exports += 1
        except OSError as exc:
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
        self.res_man.save_queue.join()
        name = self.res_man.names[ti]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        mask = None
        if path.exists(mask_path):
            with open(mask_path, 'rb') as f:
                mask = f.read()
        anchor = self.task.anchor(name) if self.task is not None else None
        return FrameState(mask, dict(anchor) if anchor is not None else None)

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
        if not isinstance(self.interaction, ClickInteraction):
            return 0
        return len(self.click_ctrl.controller.states)

    def on_undo(self) -> None:
        if self.propagating:
            return
        if self.interaction_clicks() > 1:
            # take back the last click only
            self.click_ctrl.controller.undo_click()
            self.interaction.pop_click()
            self.interaction.obj_mask = self.click_ctrl.controller.probs_history[-1][1]
            with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
                self.interacted_prob = self.interaction.predict().to(self.device)
            self.update_interacted_mask()
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
        self.res_man.save_queue.join()
        ti = self.res_man.name_to_ti[name]
        mask_path = path.join(self.res_man.mask_dir, name + '.png')
        if state.mask is None:
            if path.exists(mask_path):
                os.remove(mask_path)
        else:
            with open(mask_path, 'wb') as f:
                f.write(state.mask)
        self.res_man.invalidate(ti)
        if self.task is not None:
            self.task.restore_anchor(name, state.anchor)
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
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100))
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()
        self.gui.refresh_reset_button_state()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError as e:
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

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
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
            self.save_current_mask()
            self.mark_anchor()
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
