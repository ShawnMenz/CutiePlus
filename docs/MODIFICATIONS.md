# Modifications compared with the original Cutie

This document describes what CutiePlus changes relative to the original Cutie repository, [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie), at upstream commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a) (`ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a`, 8 November 2024), the base of this repository. File names are relative to the repository root. The [README](../README.md) gives the overview.

## Environment and packaging

- The distribution is named `cutieplus` (version 0.1.0, `pyproject.toml`); the Python packages are still `cutie` and `gui`, so every `import cutie...` works as before. The project URLs point to CutiePlus, with the original repository listed as "Upstream".
- uv-managed environment: `pyproject.toml` (Python 3.12, PyTorch and torchvision from the CUDA 13.0 index, `opencv-python-headless`, `pyqtdarktheme-fork`), `uv.lock` and `.python-version` are committed. Training and evaluation dependencies moved to the `train` and `eval` extras. The installation steps in the [README](../README.md#installation) changed accordingly.
- `torch.cuda.amp.autocast` (deprecated) replaced by `torch.amp.autocast('cuda', ...)` in `cutie/eval_vos.py`, `cutie/model/big_modules.py`, `cutie/model/cutie.py`, `cutie/model/modules.py`, `cutie/model/train_wrapper.py`, `cutie/model/transformer/object_summarizer.py`, `cutie/utils/tensor_utils.py` and `scripting_demo.py`.
- GUI start-up: relative weight paths are resolved against the download directory (`gui/main_controller.py`), a missing `QT_QPA_PLATFORM_PLUGIN_PATH` is tolerated, and Qt's Fusion style is used when `qdarktheme` is unavailable (`interactive_demo.py`).
- Fixed the PyTorch memory gauge (a stray `/ 1024`, `gui/main_controller.py`); binary masks are written as `uint8` (`gui/exporter.py`).

## Inference and the click model

- `cutie/inference/inference_core.py`: new `upscale_to_internal_size` (off by default, so the original behaviour is unchanged). When on, frames whose shorter side is below `max_internal_size` are enlarged to it instead of only ever being shrunk.
- Task mode uses it for small targets: when the target's median area at the default internal size is below 1,500 px, the sequence is processed at 960 px instead of 480 px, and the console says which (`gui/main_controller.py`).
- `gui/click_controller.py`: the click model (RITM) runs with `NoBRS` and no click limit, instead of `f-BRS-B` with `net_clicks_limit=8`. In our own measurements on RGB-T sequences, clicks after the limit went to the L-BFGS refinement and lowered the mean IoU (0.914 to 0.839 at the ninth click).
- `gui/interaction.py`: clicks are kept as one ordered list (positive and negative), which also drives the click dots on the canvas and the undo of clicks. The old `pos_clicks` / `neg_clicks` remain as properties.
- `gui/interactive_utils.py`: clicks are drawn as green (positive) and red (negative) dots; the target mask is shown in cyan on screen (saved masks keep the DAVIS palette).

## Task mode

Files: `gui/task_state.py`, `gui/resource_manager.py`, `gui/reader.py`, `gui/main_controller.py`, `gui/edit_history.py`, `gui/task_export.py`.

- Tasks are read in place from the frame directories named in `task.json`; nothing is copied.
- **Anchors**: every human edit on a frame makes it an anchor, recorded with its origin and author. Propagation never overwrites an anchor. An *empty* anchor means the target is absent, and propagation stops there. `[` `]` and Ctrl+A / D jump between anchors; Delete removes one.
- **Prefill masks**: an existing model mask is shown for frames nobody has touched, and the frame label says where every mask comes from (anchor, empty anchor, propagated, prefill, no mask).
- **Two modalities** (`visual`, shown as RGB, and `thermal`): `T` switches the displayed modality; clicks and propagation (F / B) use the one shown, with every anchor (the frames of both are aligned, so one mask fits both). The task's anchor modality is the one shown first and names the export folder.
- **Permanent memory** is rebuilt from the anchors before every propagation, within a token budget that caps how many anchors can be used at once.
- **Clip interval and target description** can be edited in the window. The interval boxes accept `m` for the first / last frame. Optionally a sequence keeps several intervals (*+ Add interval*, *- Remove interval* to cancel or remove one; no overlaps), each a clip of its own, shown as bands on the frame slider, reached with { / } (previous / next interval start) and previewed one by one with Shift+R; F / B started inside an interval stop at its last / first frame (`cutieplus/engine/policies.py`, both modes); `interval.json` and `interval.txt` keep their one-interval form and gain a list form (see the README's task folder table). Each interval has its own description: the top box describes the sequence and its first interval (on its right); every added interval is a row of its own with a description box that starts as the top box's text, is edited independently afterwards, and is saved with the interval in `interval.json`; `description.txt` then has one numbered description per interval. Both are saved in the task (`interval.json`, `description.json`), count as a change, and are written next to the binary PNGs by *Save binary PNGs*.
- **Review workflow**: not started, in progress (any change), under review (*Save binary PNGs*), done (*Mark done* in the browser, which exports first). Any change reopens a task. *Need Attention* (an orange toggle in the window, `attention.json`) shows a task as *Attention* in red in the browser and keeps it from being marked done until it is turned off; it is not a change of the task. *Discard* in the browser (confirmed; *Restore* undoes it, `discard.json`) leaves a task out of the dataset: status *discarded* in gray italics, its done mark and Need Attention removed, Export and Mark done refused, and *Hide done / discarded* hides it.
- **Export** runs the command from `task_set.json` in the background, so Cutie does not depend on the pipeline that consumes the masks.
- **Undo / redo** (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y, up to 100 edits) for clicks, *Reset frame*, *Import mask* and anchor changes. Propagation is not undoable.
- **Reset all frame masks** (guarded, cannot be undone) clears a task's masks, anchors and exports.

## Task browser

File: `gui/task_browser.py` (`--tasks`).

- Tabs per split; columns for sequence, frames, anchors, status (colored), annotators, who has it open, modality and target (the category in bold); sortable on every column but *Frames*; the models are loaded once and reused for every task. *Frames* shows the sequence's frame count until the sequence is marked done, then the frames of each of its intervals (the dataset's clips) in order, e.g. `120/85/40` (one number for one interval; the full count when it has none), with the frame ranges on hover.
- A filter box (sequence, annotator or target text) and a *Hide done* checkbox; counts of each status for the current tab, the number of sequences each annotator has worked on, and *Total Intervals*, the intervals of the sequences marked done in every tab (the dataset's clips), in a *Statistics* card beside the messages.
- *Open*, *Export*, *Mark done* (key D), *Discard* / *Restore*, *Refresh* and *Manage annotators*.
- A *Theme* button at the top right cycles Auto (follows the desktop), Light, Dark and Eye care for every window of the program. Eye care is Light with every white and light-gray surface in a cream (#FAF9DE): the window background, the cards, the boxes, the buttons, every table row (they no longer alternate) and the surfaces the theme package draws white (tooltips, menus, lists, tables); a control under the pointer is a deeper cream, and the lines and tracks stay as in Light. The choice is kept for the screen it was made on (in the user's Qt settings, per `DISPLAY`) and applied at the next start; `gui/theme.py`. Auto is resolved when the theme is applied, from the desktop setting (light or dark): the theme package's own Auto mode watched the desktop with a `gsettings monitor` process that was left running each time a window was restarted.

## Several annotators

Files: `gui/task_state.py`, `gui/annotators.py`, `gui/annotator_manager.py`.

- Per-task lock, *Open by* column, recorded annotator names, and the *Manage annotators* page with `annotators.json`, as described in the [README](../README.md#several-annotators).

## GUI

Files: `gui/gui.py`, `gui/gui_utils.py`, `gui/style.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/TIPS.md`.

- **Look** (`gui/style.py`, every window): flat "soft cards" in the light and dark themes. Controls are grouped into rounded cards with 8 px controls and 12 px cards; one green accent (#00853E, with #509E2F for hover, the interval bands and the dark theme's accent); the main actions are filled green buttons (*Propagate*, *Save binary PNGs*, *Open*), *Mark as anchor* / *Mark done* are tinted, destructive ones are outlined in red; the category and the key names are pills; the task browser draws each status as a colored pill. No shadows or gradients, which would slow a VNC screen. In the labeling window the controls of the original Cutie (memory gauges and settings, imports, the overlay video, *Reset all frame masks*) are folded into a collapsible *Advanced* card in the sidebar, open or closed as last left on that screen; the messages take the sidebar's height, with *Tips (F1)* above them; the controls under the slider form three cards: *Frame*, *Propagate* and *Save*.

- **View**: Ctrl + wheel zooms at the cursor (up to 32x, nearest-neighbour so pixels stay sharp); the wheel scrolls up and down, Alt + wheel left and right.
- **Layout that adapts**: the controls below the image wrap onto more rows in a narrow window, the sidebar scrolls and can be resized or hidden, a labeling window opens maximized (it fills the VNC desktop, which follows the viewer's window; un-maximized it has its fitted size), other windows maximize on small screens and follow a screen that is resized (useful through VNC), and the log area takes the free height of the right panel and keeps showing the newest message.
- **Keys**: A / D previous / next frame (Shift: 10 frames, Alt: first / last), Ctrl+A / D previous / next anchor, { / } previous / next interval start, R play / stop (plays once), Shift+R preview the intervals one by one, C mark an anchor, F / B propagate, Space cycles the visualization (including *light*), T toggles the modality, Ctrl+S / Ctrl+E save the binary PNGs / overlay video. Without a mouse: `,` / `.` click at the pointer (left / right), Ctrl + `+` / `-` zoom, Ctrl+0 fits the image, and the arrow keys scroll a zoomed image (they do not move frames); the mouse works as before. Shortcut keys are written on the buttons and hover tips list the combinations.
- **Tips**: a *Tips* button (F1) opens the guide in a window of its own.
- **Frame label**: the file, the modality (RGB in red, green and blue letters; thermal in orange-red) and the mask source, each status in its own color, with a legend on hover; the frame key boxes beside the slider name the A / D keys; the slider marks each anchor with a small hollow ring (black, or white in a dark theme; magenta for an empty anchor), smaller than its handle.
- **Focus**: the frame slider has the keyboard by default, so the frame keys work at once; typing in the description or the interval boxes never moves frames, and Enter or a click on the image gives the keys back. The description boxes (`TextBox` in `gui/gui_utils.py`) keep every key while they have the cursor, so the text keys edit the text and no window shortcut fires but Ctrl+S and F1; Ctrl+Y redoes the typing, and a Mac's ⌘ (Meta) and ⌥ (Alt) keys work as in a Mac text field.
- **Saving**: *Save binary PNGs* and *Save overlay video* write into the task folder and are not the dataset export (the browser's *Export* is).
- **Playback speed**: a box beside *Play video* sets the frames per second of R and Shift+R (1-60, 30 by default), also during playback, remembered for the screen as the theme is.
- **No UIUC overlay**: the original Cutie loads `docs/uiuc.png` as the *layer* view's overlay at start-up and says so in the messages. It is off (`SHOW_DEFAULT_LAYER` in `gui/main_controller.py`; loaded without a message when turned on); *Import layer* still loads a layer.

## Fixes and hardening

Fixes of behaviour in the original tool:

- Changing frame saved the mask of the frame being left under the *new* frame's number, because the frame index was updated before the save (`on_slider_update` in `gui/main_controller.py`). It is now saved first.
- Propagation could lose its place when a frame key was pressed or the video was playing during a run: the original advances one shared frame counter that those inputs also change. The frame number is now derived from the start frame and the step, and video playback does nothing during a run.

Guards and clean-up for opening many tasks in one session:

- A closed task window now releases its processor, caches and window; before, every opened task kept about 38 MB of GPU memory in our tests.
- *Reset* and *Import* are refused while a propagation runs, playback stops when one starts, a window cannot be closed while its export runs, and the window's timers belong to the window so that a closed one can be freed.

## Shared GPU service

Files: `cutieplus/` (new), `interactive_demo.py` (`--backend`, or the task root's `.cutieplus/backend`), `gui/main_controller.py`,
`gui/task_browser.py`, `gui/click_controller.py`, `gui/ritm/controller.py`,
`gui/interactive_utils.py`, `gui/resource_manager.py`, `gui/task_state.py`,
`cutie/config/gui_config.yaml` (section `service`). See [SHARED_SERVICE.md](SHARED_SERVICE.md).

- The window's inference runs through `cutieplus/engine` behind a backend: `local` keeps the
  original behaviour (a recorded session of the original window still matches it exactly);
  `shared` sends the work to a coordinator and its GPU worker.
- `gui/interactive_utils.py` no longer creates CUDA tensors at import (which also made the CPU
  mode fail on a machine with a GPU).
- The mask save threads survive a failed save (before, `save_queue.join()` then waited forever and
  the window froze); masks are written to a temporary file and renamed; failures are shown.
- The permanent-memory anchor limit is computed by `cutieplus/engine/policies.py` (same numbers);
  in shared mode the worker's measured memory budget can lower it, which is recorded.
- The click controller can give each interaction its own predictor state over one shared network
  (`ClickController.session`).

## Removed from the original

Dead or unused code was deleted: `cutie/inference/utils/frame_utils.py` (its import was broken), `scripting_demo_add_del_objects.py`, the RITM modules that nothing imports (`gui/ritm/inference/evaluation.py`, `gui/ritm/model/is_deeplab_model.py`, `gui/ritm/model/modeling/{basic_blocks,deeplab_v3,resnet}.py`, `gui/ritm/utils/{distributed,exp,vis}.py`, `gui/ritm/utils/exp_imports/default.py`).

## New files

`gui/task_state.py`, `gui/task_browser.py`, `gui/task_export.py`, `gui/edit_history.py`, `gui/annotators.py`, `gui/annotator_manager.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/theme.py`, `gui/style.py`. `gui/TIPS.md` was rewritten as the labeling guide. `gui/gui.py`, `gui/gui_utils.py`, `gui/main_controller.py`, `gui/resource_manager.py`, `gui/reader.py`, `gui/interaction.py`, `gui/interactive_utils.py`, `gui/click_controller.py` and `interactive_demo.py` were changed extensively.

## Repository files

- `README.md` was rewritten for CutiePlus. The original material (demos, training and evaluation links, references) is kept, with credit to the original authors; the citation is the official CVPR 2024 entry of the original Cutie paper.
- `LICENSE`: the original MIT license text and the line `Copyright (c) 2024 Ho Kei Cheng` are unchanged; one line for the modifications was added. `gui/ritm/LICENSE` (Samsung, 2021) is untouched.
- `CITATION.cff` (new): citation metadata for CutiePlus that points to the original Cutie paper.
- `docs/MODIFICATIONS.md` (new): this document.
