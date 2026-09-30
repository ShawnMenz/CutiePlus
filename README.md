# CutiePlus

**CutiePlus is a modified version of [Cutie](https://github.com/hkchengrex/Cutie), the video object segmentation framework from *Putting the Object Back into Video Object Segmentation* (CVPR 2024, Highlight).**

All credit for the model, the original code and the interactive tool belongs to the original authors:

[Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Joon-Young Lee](https://joonyoung-cv.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/) — University of Illinois Urbana-Champaign and Adobe.

[[Original repository]](https://github.com/hkchengrex/Cutie) [[arXiv]](https://arxiv.org/abs/2310.12982) [[PDF]](https://arxiv.org/pdf/2310.12982.pdf) [[Project page]](https://hkchengrex.github.io/Cutie/) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

- CutiePlus is based on the original repository at commit `[ec5cdd4](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a)` (8 November 2024). Modified by Zexuan Meng.
- It is **not affiliated with or endorsed by** the original authors. Problems with CutiePlus are not theirs to fix.
- If you use this work, **please cite the original paper** (see [Citation](#citation)).
- Licensed under the MIT License, like the original ([License](#license)).



## What CutiePlus adds

The original Cutie ships a model and a GUI for interactive video segmentation. CutiePlus keeps both and adds what is needed to **label many sequences with several annotators** and deliver masks into a dataset pipeline:

- **Task mode**: a sequence opens as a *task* with human-decided *anchor* frames that propagation never overwrites, optional model-made *prefill* masks, a visible (RGB) / thermal toggle, an editable clip interval and target description, undo/redo, and a review workflow (not started → in progress → under review → done).
- **Task browser**: one window that lists all tasks with their status, annotators and target text, opens them one at a time, and exports them.
- **Several annotators on one machine**: per-task locks, recorded annotator names, and a *Manage annotators* page that assigns each annotator a port.
- **A reworked GUI**: zoom and scroll, a layout that adapts to any screen size, keyboard shortcuts shown on the buttons, a Tips window, and colored status labels.
- **A modern, reproducible environment** (uv, Python 3.12, PyTorch with CUDA 13) and a number of fixes.

The full list of changes is under [Modifications compared with the original Cutie](#modifications-compared-with-the-original-cutie).

## Installation

Tested on Ubuntu 24.04 with an RTX 4090 NVIDIA GPU. The environment is managed with [uv](https://docs.astral.sh/uv/): Python 3.12, and PyTorch with CUDA 13.0 (needs NVIDIA driver 580 or newer).

**Prerequisites:**

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For the GUI on X11: `sudo apt install libxcb-cursor0` (required by Qt 6.5+)

**Install:**

```bash
cd CutiePlus
uv sync
```

This creates `.venv` from `uv.lock`. Training and benchmark evaluation need extra packages: `uv sync --extra train`, `uv sync --extra eval`, or `uv sync --all-extras`.

**Download the pretrained models** (they come from the original Cutie [release](https://github.com/hkchengrex/Cutie/releases/tag/v1.0) and are not stored in this repository):

```bash
uv run python cutie/utils/download_models.py
```

Run the other commands below with `uv run` as well (e.g. `uv run python interactive_demo.py ...`).

## Quick Start



### Scripting Demo

This is probably the best starting point if you want to use Cutie in your project. Hopefully, the script is self-explanatory (additional comments in `scripting_demo.py`).

```python
@torch.inference_mode()
@torch.autocast('cuda')
def main():

    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    # the processor matches the shorter edge of the input to this size
    # you might want to experiment with different sizes, -1 keeps the original size
    processor.max_internal_size = 480

    image_path = './examples/images/bike'
    images = sorted(os.listdir(image_path))  # ordering is important
    mask = Image.open('./examples/masks/bike/00000.png')
    palette = mask.getpalette()
    objects = np.unique(np.array(mask))
    objects = objects[objects != 0].tolist()  # background "0" does not count as an object
    mask = torch.from_numpy(np.array(mask)).cuda()

    for ti, image_name in enumerate(images):
        image = Image.open(os.path.join(image_path, image_name))
        image = to_tensor(image).cuda().float()

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)
        else:
            output_prob = processor.step(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)

        # visualize prediction
        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        mask.putpalette(palette)
        mask.show()  # or use mask.save(...) to save it somewhere


main()
```



### Interactive Demo

Start the interactive demo with:

```bash
uv run python interactive_demo.py --video ./examples/example.mp4 --num_objects 1
```

[See more instructions here](docs/INTERACTIVE.md). Press **F1** (or the **Tips** button) in the window for the keys and what every control does; the same guide is in `[gui/TIPS.md](gui/TIPS.md)`.

If you are running this on a remote server, X11 forwarding is possible (start with `ssh -X`); additional configuration may be needed.

![demo](https://i.imgur.com/nqlYqTq.jpg)

(For single video evaluation, see the unofficial script `scripts/process_video.py` from [https://github.com/hkchengrex/Cutie/pull/16](https://github.com/hkchengrex/Cutie/pull/16))

## Task mode and the task browser

A folder that contains a `task.json` opens in **task mode** (one target per task). Tasks are organised as `<task root>/<split>/<sequence>/`, and the task root holds a `task_set.json`.

```bash
# one task
uv run python interactive_demo.py --workspace <task root>/<split>/<sequence>

# the task browser: every task of the root, one window at a time
uv run python interactive_demo.py --tasks <task root> --task_split main
```

`--task_split` picks the tab shown first. `--labeler NAME` sets who is recorded as the annotator (see [Several annotators](#several-annotators)).

**Task folder** (`<task root>/<split>/<sequence>/`):


| File                | Written by               | Content                                                                                                                                                                                                                                                                                                                                  |
| ------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task.json`         | the task creator         | `sequence_id`, `split`, `anchor_modality` (`visual` or `thermal`), `frame_dirs` (modality → frame directory, read in place; the modalities are named `visual` and `thermal`), optional `frame_files` / `frame_count`, `category`, `description` (the referring text), optional `curated_interval` (`[start, end]`, 1-indexed, inclusive) |
| `anchors.json`      | task creator, then Cutie | frame name → `{kind: mask or empty, origin, updated_at, by}`; origin `ls` / `ls_refined` marks a label imported from an external tool and later edited, `cutie` a label made here                                                                                                                                                        |
| `masks/<frame>.png` | task creator, then Cutie | palette PNG of each anchor and each propagated frame                                                                                                                                                                                                                                                                                     |
| `prefill.json`      | task creator             | frame name → `{source, path}` of an existing model mask, shown until the frame is touched                                                                                                                                                                                                                                                |
| `interval.json`     | Cutie                    | `{start, end}` of the dataset clip when set in Cutie                                                                                                                                                                                                                                                                                     |
| `description.json`  | Cutie                    | `{text, by, updated_at}` when the target description was tuned                                                                                                                                                                                                                                                                           |
| `progress.json`     | Cutie                    | `last_edited`, `last_edited_by`, `edited_by`, and the review marks `review_at` / `review_by`, `done_at` / `done_by`                                                                                                                                                                                                                      |
| `lock.json`         | Cutie                    | who has the task open (see below)                                                                                                                                                                                                                                                                                                        |
| `binary_masks/`     | Cutie                    | binary PNGs plus `interval.txt` and `description.txt`, from *Save binary PNGs*                                                                                                                                                                                                                                                           |


**Task root:**


| File              | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_set.json`   | `splits` (the browser's tabs), `export_command` (a command as a list, with `{split}` and `{sequence}` placeholders, run by the *Export* button), `export_tree` (where an export lands; a template ending in `cutie_masks_{modality}/{sequence}`), `curated_csv` (a CSV with a `split` column: row *n*, counting from 1, is the sequence `n` padded to four digits, and *Reset all frame masks* is disabled for a sequence whose row says `done`, or when the file cannot be read) |
| `annotators.json` | port → annotator name, edited with *Manage annotators*                                                                                                                                                                                                                                                                                                                                                                                                                            |


Cutie never imports the code that creates tasks or consumes exports: both are plain files and an external command. Task folders are created by a separate labeling pipeline, which is not part of this repository.

### Several annotators

Several Cutie sessions can share one task root, for example one virtual screen per annotator on a shared workstation.

- A task is open in one window at a time (`lock.json`, an exclusive file lock that the operating system drops if the process dies). The browser's *Open by* column shows who has which task and refreshes every 10 seconds. The lock uses `fcntl`, so tasks are not locked on Windows.
- The annotator's name is recorded on anchors, description changes and review steps. It is, in order: `--labeler`, the annotator assigned to this screen's port in `annotators.json`, `$CUTIE_LABELER`, the account name.
- **Manage annotators** (button in the task browser) edits `annotators.json`: rows of `Port: [5902] --> [name]`, where port 59NN is virtual screen `:NN`. Add, edit and remove rows, look one up by port or name, and press **Save** to make everything effective; nothing is written before that. The next task opened on any screen records the new names without a restart.
- `annotators.json` holds people's names, so it is listed in `.gitignore`.



## Modifications compared with the original Cutie

This section lists every change relative to the original repository at commit `ec5cdd4`. File names are relative to the repository root.

### Environment and packaging

- uv-managed environment: `pyproject.toml` (Python 3.12, PyTorch and torchvision from the CUDA 13.0 index, `opencv-python-headless`, `pyqtdarktheme-fork`), `uv.lock` and `.python-version` are committed. Training and evaluation dependencies moved to the `train` and `eval` extras. The installation steps in this README changed accordingly.
- `torch.cuda.amp.autocast` (deprecated) replaced by `torch.amp.autocast('cuda', ...)` in `cutie/eval_vos.py`, `cutie/model/big_modules.py`, `cutie/model/cutie.py`, `cutie/model/modules.py`, `cutie/model/train_wrapper.py`, `cutie/model/transformer/object_summarizer.py`, `cutie/utils/tensor_utils.py` and `scripting_demo.py`.
- GUI start-up: relative weight paths are resolved against the download directory (`gui/main_controller.py`), a missing `QT_QPA_PLATFORM_PLUGIN_PATH` is tolerated, and Qt's Fusion style is used when `qdarktheme` is unavailable (`interactive_demo.py`).
- Fixed the PyTorch memory gauge (a stray `/ 1024`, `gui/main_controller.py`); binary masks are written as `uint8` (`gui/exporter.py`).



### Inference and the click model

- `cutie/inference/inference_core.py`: new `upscale_to_internal_size` (off by default, so the original behaviour is unchanged). When on, frames whose shorter side is below `max_internal_size` are enlarged to it instead of only ever being shrunk.
- Task mode uses it for small targets: when the target's median area at the default internal size is below 1,500 px, the sequence is processed at 960 px instead of 480 px, and the console says which (`gui/main_controller.py`).
- `gui/click_controller.py`: the click model (RITM) runs with `NoBRS` and no click limit, instead of `f-BRS-B` with `net_clicks_limit=8`. In our own measurements on RGB-T sequences, clicks after the limit went to the L-BFGS refinement and lowered the mean IoU (0.914 to 0.839 at the ninth click).
- `gui/interaction.py`: clicks are kept as one ordered list (positive and negative), which also drives the click dots on the canvas and the undo of clicks. The old `pos_clicks` / `neg_clicks` remain as properties.
- `gui/interactive_utils.py`: clicks are drawn as green (positive) and red (negative) dots; the target mask is shown in cyan on screen (saved masks keep the DAVIS palette).



### Task mode

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



### Task browser

File: `gui/task_browser.py` (`--tasks`).

- Tabs per split; columns for sequence, frames, anchors, status (colored), annotators, who has it open, modality and target (the category in bold); sortable; the models are loaded once and reused for every task.
- A filter box (sequence, annotator or target text) and a *Hide done* checkbox; counts of each status for the current tab, and the number of sequences each annotator has worked on.
- *Open*, *Export*, *Mark done*, *Refresh* and *Manage annotators*.



### Several annotators

Files: `gui/task_state.py`, `gui/annotators.py`, `gui/annotator_manager.py`.

- Per-task lock, *Open by* column, recorded annotator names, and the *Manage annotators* page with `annotators.json`, as described above.



### GUI

Files: `gui/gui.py`, `gui/gui_utils.py`, `gui/flow_layout.py`, `gui/tips_window.py`, `gui/TIPS.md`.

- **View**: Ctrl + wheel zooms at the cursor (up to 32x, nearest-neighbour so pixels stay sharp); the wheel scrolls up and down, Alt + wheel left and right.
- **Layout that adapts**: the controls below the image wrap onto more rows in a narrow window, the right panel scrolls and can be resized or hidden, windows maximize on small screens and follow a screen that is resized (useful through VNC), and the log area takes the free height of the right panel and keeps showing the newest message.
- **Keys**: A / D previous / next frame (Shift: 10 frames, Alt: first / last), Ctrl+A / D previous / next anchor, R play / stop (plays once), C mark an anchor, F / B propagate, Space cycles the visualization (including *light*), T toggles the modality, Ctrl+S / Ctrl+E save the binary PNGs / overlay video. The arrow keys do not move frames. Shortcut keys are written on the buttons and hover tips list the combinations.
- **Tips**: a *Tips* button (F1) opens the guide in a window of its own.
- **Frame label**: the file, the modality (RGB in red, green and blue letters; thermal in orange-red) and the mask source, each status in its own color, with a legend on hover; the frame key boxes beside the slider name the A / D keys.
- **Focus**: the frame slider has the keyboard by default, so the frame keys work at once; typing in the description or the interval boxes never moves frames, and Enter or a click on the image gives the keys back.
- **Saving**: *Save binary PNGs* and *Save overlay video* write into the task folder and are not the dataset export (the browser's *Export* is).



### Fixes and hardening

Fixes of behaviour in the original tool:

- Changing frame saved the mask of the frame being left under the *new* frame's number, because the frame index was updated before the save (`on_slider_update` in `gui/main_controller.py`). It is now saved first.
- Propagation could lose its place when a frame key was pressed or the video was playing during a run: the original advances one shared frame counter that those inputs also change. The frame number is now derived from the start frame and the step, and video playback does nothing during a run.

Guards and clean-up for opening many tasks in one session:

- A closed task window now releases its processor, caches and window; before, every opened task kept about 38 MB of GPU memory in our tests.
- *Reset* and *Import* are refused while a propagation runs, playback stops when one starts, a window cannot be closed while its export runs, and the window's timers belong to the window so that a closed one can be freed.



### Removed from the original

Dead or unused code was deleted: `cutie/inference/utils/frame_utils.py` (its import was broken), `scripting_demo_add_del_objects.py`, the RITM modules that nothing imports (`gui/ritm/inference/evaluation.py`, `gui/ritm/model/is_deeplab_model.py`, `gui/ritm/model/modeling/{basic_blocks,deeplab_v3,resnet}.py`, `gui/ritm/utils/{distributed,exp,vis}.py`, `gui/ritm/utils/exp_imports/default.py`).

### New files

`gui/task_state.py`, `gui/task_browser.py`, `gui/task_export.py`, `gui/edit_history.py`, `gui/annotators.py`, `gui/annotator_manager.py`, `gui/flow_layout.py`, `gui/tips_window.py`. `gui/TIPS.md` was rewritten as the labeling guide. `gui/gui.py`, `gui/gui_utils.py`, `gui/main_controller.py`, `gui/resource_manager.py`, `gui/reader.py`, `gui/interaction.py`, `gui/interactive_utils.py`, `gui/click_controller.py` and `interactive_demo.py` were changed extensively.

## Training and Evaluation

1. [Running Cutie on video object segmentation data.](docs/EVALUATION.md)
2. [Training Cutie.](docs/TRAINING.md)

These parts of the original repository are unchanged apart from the `autocast` replacement above.

## About the original Cutie

Cutie is a video object segmentation framework -- a follow-up work of [XMem](https://github.com/hkchengrex/XMem) with better consistency, robustness, and speed.
The original repository contains code for standard video object segmentation and a GUI tool for interactive video segmentation.
The GUI tool additionally contains the "permanent memory" (from [XMem++](https://github.com/max810/XMem2)) option for better controllability.

![overview](https://imgur.com/k84c965.jpg)

Demo video of the original tool:

[https://github.com/hkchengrex/Cutie/assets/7107196/83a8abd5-369e-41a9-bb91-d9cc1289af70](https://github.com/hkchengrex/Cutie/assets/7107196/83a8abd5-369e-41a9-bb91-d9cc1289af70)

Source: [https://raw.githubusercontent.com/hkchengrex/Cutie/main/docs/sources.txt](https://raw.githubusercontent.com/hkchengrex/Cutie/main/docs/sources.txt)

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{cheng2023putting,
  title={Putting the Object Back into Video Object Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
  booktitle={arXiv},
  year={2023}
}
```



## References and acknowledgements

- The GUI tools uses [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) for interactive image segmentation. This repository also contains a redistribution of their code in `gui/ritm`. That part of code follows RITM's license.
- For automatic video segmentation/integration with external detectors, see [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA).
- The interactive demo is developed upon [IVS](https://github.com/seoungwugoh/ivs-demo), [MiVOS](https://github.com/hkchengrex/MiVOS), and [XMem](https://github.com/hkchengrex/XMem).
- The original authors used [ProPainter](https://github.com/sczhou/ProPainter) in their video inpainting demo.
- Thanks to [RTIM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [XMem++](https://github.com/max810/XMem2) for making this possible.
- Thanks to the original authors of Cutie, and to everyone who contributed to it, for releasing it as open source.



## License

CutiePlus is released under the [MIT License](LICENSE), the same license as the original Cutie:

- Copyright (c) 2024 Ho Kei Cheng (the original Cutie)
- Copyright (c) 2026 Zexuan Meng (the modifications in this repository)

The redistributed RITM code in `gui/ritm` is under its own MIT license, Copyright (c) 2021 Samsung Electronics Co., Ltd. (`[gui/ritm/LICENSE](gui/ritm/LICENSE)`). The pretrained weights are the original authors' and are downloaded from their release page.