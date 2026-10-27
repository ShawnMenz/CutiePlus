# Modifications compared with the original Cutie

This document describes what CutiePlus changes relative to the original Cutie repository, [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie), at upstream commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a) (`ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a`, 8 November 2024), the base of this repository. File names are relative to the repository root. The [README](../README.md) gives the overview.

## Environment and packaging

- The distribution is named `cutieplus` (version 0.1.0, `pyproject.toml`); the Python packages are still `cutie` and `gui`, so every `import cutie...` works as before. The project URLs point to CutiePlus, with the original repository listed as "Upstream".
- uv-managed environment: `pyproject.toml` (Python 3.12, PyTorch and torchvision from the CUDA 13.0 index, `opencv-python-headless`, `pyqtdarktheme-fork`), `uv.lock` and `.python-version` are committed. Training and evaluation dependencies moved to the `train` and `eval` extras. The installation steps in the [README](../README.md#installation) changed accordingly.
- `torch.cuda.amp.autocast` (deprecated) replaced by `torch.amp.autocast('cuda', ...)` in `cutie/eval_vos.py`, `cutie/model/big_modules.py`, `cutie/model/cutie.py`, `cutie/model/modules.py`, `cutie/model/train_wrapper.py`, `cutie/model/transformer/object_summarizer.py`, `cutie/utils/tensor_utils.py` (and in `scripting_demo.py`, since removed).
- GUI start-up: relative weight paths are resolved against the download directory (`gui/main_controller.py`), a missing `QT_QPA_PLATFORM_PLUGIN_PATH` is tolerated, and Qt's Fusion style is used when `qdarktheme` is unavailable (`interactive_demo.py`).
- Fixed the PyTorch memory gauge (a stray `/ 1024`, `gui/main_controller.py`); binary masks are written as `uint8` (`gui/exporter.py`).

## Inference and the click model

- `cutie/inference/inference_core.py`: new `upscale_to_internal_size` (off by default, so the original behaviour is unchanged). When on, frames whose shorter side is below `max_internal_size` are enlarged to it instead of only ever being shrunk.
- Task mode uses it for small targets: when the target's median area at the default internal size is below 1,500 px, the sequence is processed at 960 px instead of 480 px, and the console says which (`gui/main_controller.py`).
- `gui/click_controller.py`: the click model (RITM) runs with `NoBRS` and no click limit, instead of `f-BRS-B` with `net_clicks_limit=8`. In our own measurements on RGB-T sequences, clicks after the limit went to the L-BFGS refinement and lowered the mean IoU (0.914 to 0.839 at the ninth click).
- `gui/interaction.py`: clicks are kept as one ordered list (positive and negative), which also drives the click dots on the canvas and the undo of clicks. The old `pos_clicks` / `neg_clicks` remain as properties.
- `gui/interactive_utils.py`: clicks are drawn as green (positive) and red (negative) dots with a thin white ring; the target mask is shown on screen in its category's color (`set_target_color`, from `categories.json`), cyan without a category (saved masks keep the DAVIS palette).

## Task mode

Files: `gui/task_input.py`, `gui/task_state.py`, `gui/resource_manager.py`, `gui/main_controller.py`, `gui/edit_history.py`.

- **Input** (`gui/task_input.py`): `workspace/cutie_tasks/<subset>/` per subset (a tab of the browser), with `visual/<seq>/`, `thermal/<seq>/` and `categories.json` (needed), `targets.json` (each sequence's category and description), `annotation/<seq>/` (pregenerated masks) and `seqinfo/` (optional). Frames are read in place and matched across modalities by name; nothing is copied. A sequence without a target has the category `null` and an empty description. `scripts/extract_targets.py` writes `targets.json` and `categories.json` (one color per category, the same in every subset) from `seqinfo/`.
- **Work folder** (`gui/task_state.py: work_folder`): each sequence's work is kept in `workspace/cutie_output/<subset>/<seq>/`, made when it is first opened, with two things only: `masks/` and `metadata.json`, whose sections are its source (the input), category, description, intervals, review progress, Need Attention, discard and anchors. Masks and metadata are written as the sequence is labeled. The lock that keeps a task in one window lives in the user's runtime folder, outside the output.
- **Anchors**: every human edit on a frame makes it an anchor, recorded with its author and time; the frame label shows one of five statuses (anchor, empty anchor, propagated, pregenerated, no mask). Propagation never overwrites an anchor. An *empty* anchor means the target is absent, and propagation stops there. `[` `]` and Ctrl+A / D jump between anchors; Delete removes one.
- **One set at a time** (`gui/gui.py`, `gui/gui_utils.py: IntervalSetRow`): each interval is a set with a category, a description and its frames; set 1 is the sequence's own, an added interval has its own category (saved with it in the intervals section, next to its description). The top card shows the selected set only; a triangle (`▶ 1/3`) opens the list of every set inline to pick one, which folds it and goes to the interval's first frame; { / } and Shift+R select the interval they reach. On the frame slider the selected interval's band is bold green (#00853E, a little taller), the others light green (#509E2F).
- **Category** (`gui/category_menu.py`): the labeling window's category pill takes the category's color (`categories.json`), with dark or white text, whichever reads (`gui/style.py: contrast_text`); a click opens a menu of the subset's categories (name, color sample) with a search box that keeps the names starting with the letters typed. The choice is saved in the work folder (`metadata.json`) and needs a new review. The target's mask is drawn in the same color. A `categories.json` may list names only (`["car", "person"]`): the Import window's *Save* picks a distinct color for each (`gui/category_colors.py`, the same as `scripts/extract_targets.py`; a name already colored in another subset keeps that color) and writes them into the file.
- **Pregenerated masks**: the input's `annotation/` mask is shown for frames nobody has touched, and the frame label says where every mask comes from (anchor, empty anchor, propagated, pregenerated, no mask).
- **Two modalities** (`visual`, shown as RGB, and `thermal`): `T` switches the displayed modality; clicks and propagation (F / B) use the one shown, with every anchor (the frames of both are aligned, so one mask fits both). A sequence opens on visual (thermal when it has no visual frames).
- **Permanent memory** is rebuilt from the anchors before every propagation, within a token budget that caps how many anchors can be used at once.
- **Clip interval and target description** can be edited in the window. The interval boxes accept `m` for the first / last frame, and every interval row shows its frame count on the right of its boxes (both ends included: 15 - 650 is 636 frames), updated as the numbers are typed. Optionally a sequence keeps several intervals (*+ Add interval*, *- Remove interval* to cancel or remove one; no overlaps), each a clip of its own, shown as bands on the frame slider, reached with { / } (previous / next interval start) and previewed one by one with Shift+R; F / B started inside an interval stop at its last / first frame (`cutieplus/engine/policies.py`, both modes); the intervals section keeps its one-interval form and gains a list form (see `gui/task_state.py`). Each interval has its own description: the top box describes the sequence and its first interval (on its right); every added interval is a row of its own with a description box that starts as the top box's text, is edited independently afterwards, and is saved with the interval. Both are saved in the task (`metadata.json`) and count as a change.
- **Review workflow**: not started, in progress (any change), under review (*Save*, Ctrl+S), done (*Mark done* in the browser). Any change reopens a task. *Ctrl+Q* in the window (after a warning) does both for a labeler who reviews their own work: it saves, marks the task done (`quit_at` in its progress: when it was pressed) and closes the window. *Need Attention* (an orange toggle in the window) shows a task as *Attention* in red in the browser and keeps it from being marked done until it is turned off; it is not a change of the task. *Discard* in the browser (confirmed; *Restore* undoes it) leaves a task out of the dataset: status *discarded* in gray italics, its done mark and Need Attention removed, Mark done refused, and *Hide done / discarded* hides it.
- **Export** (`gui/dataset_export.py`, the browser's *Export*): the dataset in `workspace/cutie_export/`, one folder for all tabs, laid out as an input subset: `visual/<clip>/`, `thermal/<clip>/` (the interval's frames, copied, numbered from 000001), `annotation/<clip>/` (L 0/255: the saved masks, else the pregenerated ones, else empty) and `seqinfo/<clip>_seqinfo.ini` (the RGBT_VOS format; `originalSource` is `<tab>/<sequence>`, shared by every clip cut from it, `intervalStart` / `intervalEnd` are frames of that sequence, `dataset` comes from the input's seqinfo when there is one). Every interval of every done sequence is a clip, numbered from 1 in the browser's order and zero-padded to the digits of the count (3000 clips: 0001 … 3000). It asks first, locks the exported sequences while it runs (a window holding one stops it), shows its progress (Cancel keeps the previous export) and replaces the previous export only once complete.
- **One object**: a window labels one object, the target. The original multi-object controls (the *Current object ID* row, the number keys that switched objects, *Reset object*) and `--num_objects` are gone; *Reset frame* empties the frame. The model code still takes an object index.
- **Undo / redo** (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y, up to 100 edits) for clicks, *Reset frame*, *Import mask* and anchor changes. Propagation is not undoable.
- **Reset all frame masks** (guarded, cannot be undone) clears a task's masks and anchors; its pregenerated masks show again.

## Task browser

Files: `gui/task_browser.py` (`--tasks`, by default `workspace/cutie_tasks`), `gui/import_dialog.py`, `gui/task_import.py`.

- One tab per subset of the input; columns for sequence, frames, anchors, status (colored), annotators, who has it open and target (the category in bold); sortable on every column but *Frames*; the models are loaded once and reused for every task. *Frames* shows the sequence's frame count until the sequence is marked done, then the frames of each of its intervals (the dataset's clips) in order, e.g. `120/85/40` (one number for one interval; the full count when it has none), with the frame ranges on hover.
- A filter box (sequence, annotator or target text) and a *Hide done* checkbox; counts of each status for the current tab, the number of sequences each annotator has worked on, and *Total Intervals*, the intervals of the sequences marked done in every tab (the dataset's clips), in a *Statistics* card beside the messages.
- *Open*, *Export*, *Mark done* (key D), *Discard* / *Restore*, *Refresh* and *Import*.
- **An empty list says why**, in its middle: no data imported yet (before any subset), every sequence done or discarded (Hide done / discarded ticked), or none matching the filter (`TaskTable`).
- **Import** opens a window that lists the subsets, each checked as you type (✓ counts, ✗ what is missing, ⚠ warnings): the input found directly in `workspace/cutie_tasks` moves into `main/` on *Save*, and *+ Add a subset* adds a tab name and a folder, moved into `workspace/cutie_tasks/<tab>` on *Save* (a rename on the same disk, else a copy and then a delete). *Save* works once every row passes; the browser then shows the new tabs.
- A *Theme* button at the top right cycles Auto (follows the desktop), Light, Dark and Eye care for every window of the program. Eye care is Light with every white and light-gray surface in a cream (#FAF9DE): the window background, the cards, the boxes, the buttons, every table row (they no longer alternate) and the surfaces the theme package draws white (tooltips, menus, lists, tables); a control under the pointer is a deeper cream, and the lines and tracks stay as in Light. The choice is kept in the user's Qt settings and applied at the next start; `gui/theme.py`. Auto is resolved when the theme is applied, from the desktop setting (light or dark): the theme package's own Auto mode watched the desktop with a `gsettings monitor` process that was left running each time a window was restarted.

## Locking and edit records

Files: `gui/task_state.py`, `gui/task_browser.py`.

- A task is open in one window at a time (an exclusive file lock in the user's runtime folder, which the operating system drops when the process dies); the browser's *Open by* column shows which window has it.
- Every change records who made it: `--labeler`, else `$CUTIE_LABELER`, else the account name. The browser's *Annotators* column and *Statistics* card show them.

## GUI

Files: `gui/gui.py`, `gui/gui_utils.py`, `gui/style.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/TIPS.md`.

- **Look** (`gui/style.py`, every window): flat "soft cards" in the light and dark themes. Controls are grouped into rounded cards with 8 px controls and 12 px cards; one green accent (#00853E, with #509E2F for hover, the interval bands and the dark theme's accent); the main actions are filled green buttons (*Propagate*, *Save*, *Open*), *Mark as anchor* / *Mark done* are tinted, destructive ones are outlined in red; the category and the key names are pills; the task browser draws each status as a colored pill. No shadows or gradients. In the labeling window the controls of the original Cutie (memory gauges and settings, imports, the overlay video, *Reset all frame masks*) are folded into a collapsible *Advanced* card in the sidebar, open or closed as last left; the messages take the sidebar's height, with *Tips (F1)* above them; the controls under the slider form three cards: *Frame*, *Propagate* and *Save*.

- **View**: Ctrl + wheel zooms at the cursor (up to 32x, nearest-neighbour so pixels stay sharp); the wheel scrolls up and down, Alt + wheel left and right.
- **Layout that adapts**: the controls below the image wrap onto more rows in a narrow window, the sidebar scrolls and can be resized or hidden, a labeling window opens maximized (un-maximized it has its fitted size), other windows maximize on small screens and follow a screen that is resized, and the log area takes the free height of the right panel and keeps showing the newest message.
- **Keys**: A / D previous / next frame (Shift: 10 frames, Alt: first / last), Ctrl+A / D previous / next anchor, { / } previous / next interval start, R play / stop (plays once), Shift+R preview the intervals one by one, C mark an anchor, F / B propagate, Space cycles the visualization (including *light*), T toggles the modality, Ctrl+S saves (submits for review). Without a mouse: `,` / `.` click at the pointer (left / right), Ctrl + `+` / `-` zoom, Ctrl+0 fits the image, and the arrow keys scroll a zoomed image (they do not move frames); the mouse works as before. Shortcut keys are written on the buttons and hover tips list the combinations.
- **Tips**: a *Tips* button (F1) opens the guide in a window of its own.
- **Frame label**: the file, the modality (RGB in red, green and blue letters; thermal in orange-red) and the mask source, each status in its own color, with a legend on hover; the frame key boxes beside the slider name the A / D keys; the slider marks each anchor with a small hollow ring (black, or white in a dark theme; magenta for an empty anchor), smaller than its handle.
- **Focus**: the frame slider has the keyboard by default, so the frame keys work at once; typing in the description or the interval boxes never moves frames, and Enter or a click on the image gives the keys back. The description boxes (`TextBox` in `gui/gui_utils.py`) keep every key while they have the cursor, so the text keys edit the text and no window shortcut fires but Ctrl+S and F1; Ctrl+Y redoes the typing, and a Mac's ⌘ (Meta) and ⌥ (Alt) keys work as in a Mac text field.
- **Saving**: masks are written as you label; *Save* (Ctrl+S) also saves the typed intervals and description and submits the sequence. A task window writes no binary PNGs, visualizations, soft masks or overlay video (a plain workspace keeps them, Ctrl+S / Ctrl+E); the dataset comes from the browser's *Export*.
- **Playback speed**: a box beside *Play video* sets the frames per second of R and Shift+R (1-60), also during playback; every window starts at 24. *Lock* beside it keeps the speed for every window opened afterwards (Qt settings, as the theme); unlocking goes back to 24.
- **No layer view, no UIUC overlay**: the original Cutie's *layer* view (an image, by default the UIUC logo `docs/uiuc.png`, between the target and the background) is not offered: *Visualization mode* has no *layer* and the Advanced card no *Import layer* (`SHOW_LAYER_MODE` in `gui/gui.py` brings both back). The logo is not loaded at start-up nor mentioned in the messages (`SHOW_DEFAULT_LAYER` in `gui/main_controller.py`; loaded without a message when turned on). The code stays.

## Fixes and hardening

Fixes of behaviour in the original tool:

- Changing frame saved the mask of the frame being left under the *new* frame's number, because the frame index was updated before the save (`on_slider_update` in `gui/main_controller.py`). It is now saved first.
- Propagation could lose its place when a frame key was pressed or the video was playing during a run: the original advances one shared frame counter that those inputs also change. The frame number is now derived from the start frame and the step, and video playback does nothing during a run.

Guards and clean-up for opening many tasks in one session:

- A closed task window now releases its processor, caches and window; before, every opened task kept about 38 MB of GPU memory in our tests.
- *Reset* and *Import* are refused while a propagation runs, playback stops when one starts, and the window's timers belong to the window so that a closed one can be freed.

## Inference engine

Files: `cutieplus/` (new), `gui/main_controller.py`, `gui/click_controller.py`,
`gui/ritm/controller.py`, `gui/interactive_utils.py`, `gui/resource_manager.py`.

- The window's inference runs through `cutieplus/engine` (model loading, RITM interactions,
  step-based Cutie propagation, the task-mode policies) behind its backend,
  `cutieplus/backends/local.py`; a recorded session of the original window still matches it
  exactly.
- `gui/interactive_utils.py` no longer creates CUDA tensors at import (which also made the CPU
  mode fail on a machine with a GPU).
- The mask save threads survive a failed save (before, `save_queue.join()` then waited forever and
  the window froze); masks are written to a temporary file and renamed; failures are shown.
- The permanent-memory anchor limit is computed by `cutieplus/engine/policies.py` (same numbers).
- The click controller can give each interaction its own predictor state over one shared network
  (`ClickController.session`).

## Removed from the original

Dead or unused code was deleted: `cutie/inference/utils/frame_utils.py` (its import was broken), `scripting_demo_add_del_objects.py`, `scripting_demo.py`, `scripts/process_video.py` with `cutie/config/video_config.yaml`, `cutie/utils/get_default_model.py`, `examples/`, `docs/INTERACTIVE.md` and `docs/sources.txt` (the original demos), the RITM modules that nothing imports (`gui/ritm/inference/evaluation.py`, `gui/ritm/model/is_deeplab_model.py`, `gui/ritm/model/modeling/{basic_blocks,deeplab_v3,resnet}.py`, `gui/ritm/utils/{distributed,exp,vis}.py`, `gui/ritm/utils/exp_imports/default.py`).

## Collaboration (removed, October 2026)

CutiePlus once let several annotators share the desktop: each had a TigerVNC screen of their own,
reached through an SSH tunnel, and an optional shared GPU service (a coordinator and one GPU
worker) held the only copy of the models for every window.
All of it was removed: CutiePlus runs for one person at the desktop, with the models in the window's
own process. Collaboration is to be designed again.

## New files

`gui/task_input.py`, `gui/task_state.py`, `gui/task_browser.py`, `gui/task_import.py`, `gui/import_dialog.py`, `gui/edit_history.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/theme.py`, `gui/style.py`. `gui/TIPS.md` was rewritten as the labeling guide. `gui/gui.py`, `gui/gui_utils.py`, `gui/main_controller.py`, `gui/resource_manager.py`, `gui/reader.py`, `gui/interaction.py`, `gui/interactive_utils.py`, `gui/click_controller.py` and `interactive_demo.py` were changed extensively.

## Repository files

- `README.md` was rewritten for CutiePlus. The original material (demos, training and evaluation links, references) is kept, with credit to the original authors; the citation is the official CVPR 2024 entry of the original Cutie paper.
- `LICENSE`: the original MIT license text and the line `Copyright (c) 2024 Ho Kei Cheng` are unchanged; one line for the modifications was added. `gui/ritm/LICENSE` (Samsung, 2021) is untouched.
- `CITATION.cff` (new): citation metadata for CutiePlus that points to the original Cutie paper.
- `docs/MODIFICATIONS.md` (new): this document.
