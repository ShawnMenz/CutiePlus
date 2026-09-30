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
- **Two modalities** (`visual`, shown as RGB, and `thermal`): anchors and propagation use the anchor modality; `T` switches the displayed modality and propagation keeps showing it.
- **Permanent memory** is rebuilt from the anchors before every propagation, within a token budget that caps how many anchors can be used at once.
- **Clip interval and target description** can be edited in the window. The interval boxes accept `m` for the first / last frame. Both are saved in the task (`interval.json`, `description.json`), count as a change, and are written next to the binary PNGs by *Save binary PNGs*.
- **Review workflow**: not started, in progress (any change), under review (*Save binary PNGs*), done (*Mark done* in the browser, which exports first). Any change reopens a task.
- **Export** runs the command from `task_set.json` in the background, so Cutie does not depend on the pipeline that consumes the masks.
- **Undo / redo** (Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y, up to 100 edits) for clicks, *Reset frame*, *Import mask* and anchor changes. Propagation is not undoable.
- **Reset all frame masks** (guarded, cannot be undone) clears a task's masks, anchors and exports.

## Task browser

File: `gui/task_browser.py` (`--tasks`).

- Tabs per split; columns for sequence, frames, anchors, status (colored), annotators, who has it open, modality and target (the category in bold); sortable; the models are loaded once and reused for every task.
- A filter box (sequence, annotator or target text) and a *Hide done* checkbox; counts of each status for the current tab, and the number of sequences each annotator has worked on.
- *Open*, *Export*, *Mark done*, *Refresh* and *Manage annotators*.
- A *Theme* button at the top right cycles Auto (follows the desktop), Light and Dark for every window of the program. The choice is kept for the screen it was made on (in the user's Qt settings, per `DISPLAY`) and applied at the next start; `gui/theme.py`.

## Several annotators

Files: `gui/task_state.py`, `gui/annotators.py`, `gui/annotator_manager.py`.

- Per-task lock, *Open by* column, recorded annotator names, and the *Manage annotators* page with `annotators.json`, as described in the [README](../README.md#several-annotators).

## GUI

Files: `gui/gui.py`, `gui/gui_utils.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/TIPS.md`.

- **View**: Ctrl + wheel zooms at the cursor (up to 32x, nearest-neighbour so pixels stay sharp); the wheel scrolls up and down, Alt + wheel left and right.
- **Layout that adapts**: the controls below the image wrap onto more rows in a narrow window, the right panel scrolls and can be resized or hidden, windows maximize on small screens and follow a screen that is resized (useful through VNC), and the log area takes the free height of the right panel and keeps showing the newest message.
- **Keys**: A / D previous / next frame (Shift: 10 frames, Alt: first / last), Ctrl+A / D previous / next anchor, R play / stop (plays once), C mark an anchor, F / B propagate, Space cycles the visualization (including *light*), T toggles the modality, Ctrl+S / Ctrl+E save the binary PNGs / overlay video. The arrow keys do not move frames. Shortcut keys are written on the buttons and hover tips list the combinations.
- **Tips**: a *Tips* button (F1) opens the guide in a window of its own.
- **Frame label**: the file, the modality (RGB in red, green and blue letters; thermal in orange-red) and the mask source, each status in its own color, with a legend on hover; the frame key boxes beside the slider name the A / D keys.
- **Focus**: the frame slider has the keyboard by default, so the frame keys work at once; typing in the description or the interval boxes never moves frames, and Enter or a click on the image gives the keys back.
- **Saving**: *Save binary PNGs* and *Save overlay video* write into the task folder and are not the dataset export (the browser's *Export* is).

## Fixes and hardening

Fixes of behaviour in the original tool:

- Changing frame saved the mask of the frame being left under the *new* frame's number, because the frame index was updated before the save (`on_slider_update` in `gui/main_controller.py`). It is now saved first.
- Propagation could lose its place when a frame key was pressed or the video was playing during a run: the original advances one shared frame counter that those inputs also change. The frame number is now derived from the start frame and the step, and video playback does nothing during a run.

Guards and clean-up for opening many tasks in one session:

- A closed task window now releases its processor, caches and window; before, every opened task kept about 38 MB of GPU memory in our tests.
- *Reset* and *Import* are refused while a propagation runs, playback stops when one starts, a window cannot be closed while its export runs, and the window's timers belong to the window so that a closed one can be freed.

## Removed from the original

Dead or unused code was deleted: `cutie/inference/utils/frame_utils.py` (its import was broken), `scripting_demo_add_del_objects.py`, the RITM modules that nothing imports (`gui/ritm/inference/evaluation.py`, `gui/ritm/model/is_deeplab_model.py`, `gui/ritm/model/modeling/{basic_blocks,deeplab_v3,resnet}.py`, `gui/ritm/utils/{distributed,exp,vis}.py`, `gui/ritm/utils/exp_imports/default.py`).

## New files

`gui/task_state.py`, `gui/task_browser.py`, `gui/task_export.py`, `gui/edit_history.py`, `gui/annotators.py`, `gui/annotator_manager.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/theme.py`. `gui/TIPS.md` was rewritten as the labeling guide. `gui/gui.py`, `gui/gui_utils.py`, `gui/main_controller.py`, `gui/resource_manager.py`, `gui/reader.py`, `gui/interaction.py`, `gui/interactive_utils.py`, `gui/click_controller.py` and `interactive_demo.py` were changed extensively.

## Repository files

- `README.md` was rewritten for CutiePlus. The original material (demos, training and evaluation links, references) is kept, with credit to the original authors; the citation is the official CVPR 2024 entry of the original Cutie paper.
- `LICENSE`: the original MIT license text and the line `Copyright (c) 2024 Ho Kei Cheng` are unchanged; one line for the modifications was added. `gui/ritm/LICENSE` (Samsung, 2021) is untouched.
- `CITATION.cff` (new): citation metadata for CutiePlus that points to the original Cutie paper.
- `docs/MODIFICATIONS.md` (new): this document.
