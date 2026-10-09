# CutiePlus

**CutiePlus is a modified version of [Cutie](https://github.com/hkchengrex/Cutie), the video object segmentation framework from *Putting the Object Back into Video Object Segmentation* (CVPR 2024, Highlight).**

All credit for the model, the original code and the interactive tool belongs to the original authors:

[Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Joon-Young Lee](https://joonyoung-cv.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/) — University of Illinois Urbana-Champaign and Adobe.

[[Original repository]](https://github.com/hkchengrex/Cutie) [[arXiv]](https://arxiv.org/abs/2310.12982) [[PDF]](https://arxiv.org/pdf/2310.12982.pdf) [[Project page]](https://hkchengrex.github.io/Cutie/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yo43XTbjxuWA7XgCUO9qxAi7wBI6HzvP?usp=sharing)

- CutiePlus is based on the original repository at commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a) (8 November 2024) and keeps its git history. Modified by Zexuan Meng.
- It is **not affiliated with or endorsed by** the original authors. Please report problems specific to CutiePlus in the [CutiePlus issue tracker](https://github.com/ShawnMenz/CutiePlus/issues), not to them.
- The pretrained Cutie weights are the original authors' work: they are downloaded from the original Cutie release and are not part of this repository.
- If you use CutiePlus, **please cite the original Cutie paper** (see [Citation](#citation)). CutiePlus itself is not a publication.
- Licensed under the MIT License, like the original ([License](#license)).


## What's different from Cutie?

The original Cutie ships a model and a GUI for interactive video segmentation. CutiePlus keeps both and extends Cutie with:

- **Task-based annotation workflows**: a sequence opens as a *task* with an editable clip interval and target description and optional model-made *prefill* masks; a *task browser* lists all tasks with their status, annotators and target text, opens them one at a time, and exports them.
- **Anchor-aware mask propagation**: every human edit makes a frame an *anchor* that propagation never overwrites; an empty anchor means the target is absent and propagation stops there; propagation started inside a clip interval stays inside it; permanent memory is rebuilt from the anchors.
- **RGB/thermal visualization and annotation support**: a visible (RGB) / thermal toggle, with the anchors and the propagation following the anchor modality.
- **Multi-annotator task management and locking**: per-task locks, recorded annotator names, and a *Manage annotators* page that assigns each annotator a port.
- **One shared GPU for every annotator**: an optional service (a coordinator and one GPU worker) holds the only copy of the models and serves every window, with memory limits measured on the GPU, clicks served between propagation steps, and every change saved before it is shown.
- **Review and export workflows**: not started, in progress, under review, done; masks are exported through an external command. A *Need Attention* toggle in the window flags a sequence for further attention (status *Attention* in the browser; it is not marked done while flagged).
- **Undo/redo and improved interactive editing**: undo/redo of clicks, *Reset frame*, *Import mask* and anchor changes; the click model runs without a click limit.
- **A modern uv-based environment**: Python 3.12 and PyTorch with CUDA 13, with a lockfile.
- **GUI and robustness improvements for repeated annotation work**: zoom and scroll, a layout that adapts to any screen size, keyboard shortcuts shown on the buttons, a Tips window, colored status labels, a light/dark theme button, and fixes.

For a detailed, file-level description of the changes relative to upstream Cutie commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a), see [MODIFICATIONS.md](docs/MODIFICATIONS.md).

## Installation

Tested on Ubuntu 24.04 with an RTX 4090 NVIDIA GPU. The environment is managed with [uv](https://docs.astral.sh/uv/): Python 3.12, and PyTorch with CUDA 13.0 (needs NVIDIA driver 580 or newer).

**Prerequisites:**

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For the GUI on X11: `sudo apt install libxcb-cursor0` (required by Qt 6.5+)

**Install:**

```bash
git clone https://github.com/ShawnMenz/CutiePlus.git
cd CutiePlus
uv sync
```

This creates `.venv` from `uv.lock`. Training and benchmark evaluation need extra packages: `uv sync --extra train`, `uv sync --extra eval`, or `uv sync --all-extras`.

**Tests** (pytest comes with `uv sync`): `uv run pytest` runs the CPU tests; `CUTIEPLUS_TEST_GPU=1 uv run pytest -m gpu` the ones that need the GPU. If a system-wide `PYTHONPATH` (ROS, for example) brings its own pytest plugins, run them with `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

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

[See more instructions here](docs/INTERACTIVE.md). Press **F1** (or the **Tips** button) in the window for the keys and what every control does; the same guide is in [`gui/TIPS.md`](gui/TIPS.md).

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
| `interval.json`     | Cutie                    | `{start, end}` of the dataset clip when set in Cutie; with several intervals (*+ Add interval*, each a clip of its own) `{intervals: [{start, end}, ...]}`, sorted and never overlapping                                                                                                                                                |
| `description.json`  | Cutie                    | `{text, by, updated_at}` when the target description was tuned                                                                                                                                                                                                                                                                           |
| `attention.json`    | Cutie                    | `{at, by}` while *Need Attention* is on: the browser shows the status *Attention* and *Mark done* refuses the task; removed when it is turned off (edits and reviews leave it)                                                                                                                                                             |
| `progress.json`     | Cutie                    | `last_edited`, `last_edited_by`, `edited_by`, and the review marks `review_at` / `review_by`, `done_at` / `done_by`                                                                                                                                                                                                                      |
| `lock.json`         | Cutie                    | who has the task open (see below)                                                                                                                                                                                                                                                                                                        |
| `binary_masks/`     | Cutie                    | binary PNGs plus `interval.txt` (`interval_start=` / `interval_end=`; with several intervals `interval_count=` and `interval_N_start=` / `interval_N_end=`) and `description.txt`, from *Save binary PNGs*                                                                                                                              |


**Task root:**


| File              | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_set.json`   | `splits` (the browser's tabs), `export_command` (a command as a list, with `{split}` and `{sequence}` placeholders, run by the *Export* button), `export_tree` (where an export lands; a template ending in `cutie_masks_{modality}/{sequence}`), `curated_csv` (a CSV with a `split` column: row *n*, counting from 1, is the sequence `n` padded to four digits, and *Reset all frame masks* is disabled for a sequence whose row says `done`, or when the file cannot be read) |
| `annotators.json` | port → annotator name, edited with *Manage annotators*                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `.cutieplus/`     | the shared service's state (see [One shared GPU for every annotator](#one-shared-gpu-for-every-annotator)): `service.db` (SQLite: tasks, revisions, edit history, sessions, jobs, exports), `objects/` (every committed mask, never changed), `logs/`, the socket, and `backend` (the windows' mode) |


Cutie never imports the code that creates tasks or consumes exports: both are plain files and an external command. Task folders are created by a separate labeling pipeline, which is not part of this repository.

### Several annotators

Several Cutie sessions can share one task root, for example one virtual screen per annotator on a shared workstation.

- A task is open in one window at a time (`lock.json`, an exclusive file lock that the operating system drops if the process dies; in shared mode the coordinator holds it for the window that has the task). The browser's *Open by* column shows who has which task and refreshes every 10 seconds. The lock uses `fcntl`, so tasks are not locked on Windows.
- The annotator's name is recorded on anchors, description changes and review steps. It is, in order: `--labeler`, the annotator assigned to this screen's port in `annotators.json`, `$CUTIE_LABELER`, the account name.
- **Manage annotators** (button in the task browser) edits `annotators.json`: rows of `Port: [5902] --> [name]`, where port 59NN is virtual screen `:NN`. Add, edit and remove rows, look one up by port or name, and press **Save** to make everything effective; nothing is written before that. The next task opened on any screen records the new names without a restart.
- `annotators.json` holds people's names, so it is listed in `.gitignore`.

### One shared GPU for every annotator

Each window normally loads its own copy of Cutie and RITM onto the GPU (2.3-19.3 GiB per window on our tasks). With the shared service, one **coordinator** (CPU) and one **GPU worker** that it starts hold the only copies; the windows send clicks and propagations to them and use no GPU memory. Clicks are served between the steps of a running propagation, propagations queue for the GPU, and the coordinator alone writes the tasks' annotation data: every click and propagated frame is saved (SQLite and immutable mask files, with the task files above kept up to date for the export pipeline) before the window shows it. The GPU's memory limits are measured when the worker starts; when memory runs short, a propagation keeps fewer permanent anchors (and says so) or waits, and never runs at lower precision or resolution.

Start the coordinator once per task root, in a terminal that outlives your login (here a `tmux` session):

```bash
tmux new -d -s cutieplus "uv run python -m cutieplus.coordinator start --tasks workspace/cutie_tasks"
uv run python -m cutieplus.coordinator status --tasks workspace/cutie_tasks   # worker, sessions, jobs
uv run python -m cutieplus.coordinator stop --tasks workspace/cutie_tasks
```

Then start windows in shared mode, one way or another:

```bash
uv run python interactive_demo.py --tasks workspace/cutie_tasks --backend shared   # this window
CUTIEPLUS_BACKEND=shared uv run python interactive_demo.py --tasks workspace/cutie_tasks
echo shared > workspace/cutie_tasks/.cutieplus/backend   # every window of the root that starts afterwards
```

The last one moves windows started by a fixed command (such as a VNC `xstartup` loop) without changing it; `echo local > ...` moves them back, and open windows keep their mode until they restart. Without any of these a window runs in the original local mode. Both modes respect each other's task locks and read and write the same task files, so they can be mixed, and work done in one is there in the other.

What to expect, measured on an RTX 4090 with real tasks ([report](docs/benchmarks/2026-10-07-NVIDIA-GeForce-RTX-4090.md)): the same masks as the local mode; a click 3-12 ms slower when labeling alone (about 35-80 ms in all), and 60-90 ms while a propagation runs for someone else; propagation at 84-110 % of the local speed; a crashed GPU worker replaced in about 5 s, with the running propagation resuming where it was. In shared mode the window's permanent-memory controls and *Clear memory* have no effect.

[docs/SHARED_SERVICE.md](docs/SHARED_SERVICE.md) describes the design, settings (`service` in `cutie/config/gui_config.yaml`), recovery, diagnosis and limitations.


## Documentation

- [docs/MODIFICATIONS.md](docs/MODIFICATIONS.md): what CutiePlus changes relative to the original Cutie, file by file.
- [docs/SHARED_SERVICE.md](docs/SHARED_SERVICE.md): the shared GPU service (coordinator and GPU worker): design, running it, recovery, limitations.
- [docs/benchmarks/](docs/benchmarks/): measurements of the shared service against the original local mode (`scripts/benchmark_service.py`).
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the labeling window worked before the shared service (the local mode).
- [gui/TIPS.md](gui/TIPS.md): the labeling guide (keys and controls); also in the window with **F1**.
- [docs/INTERACTIVE.md](docs/INTERACTIVE.md): instructions for the interactive demo (from the original Cutie).

## Training and Evaluation

1. [Running Cutie on video object segmentation data.](docs/EVALUATION.md)
2. [Training Cutie.](docs/TRAINING.md)

These parts of the original repository are unchanged apart from the `autocast` replacement (see [MODIFICATIONS.md](docs/MODIFICATIONS.md#environment-and-packaging)).

## About the original Cutie

Cutie is a video object segmentation framework -- a follow-up work of [XMem](https://github.com/hkchengrex/XMem) with better consistency, robustness, and speed.
The original repository contains code for standard video object segmentation and a GUI tool for interactive video segmentation.
The GUI tool additionally contains the "permanent memory" (from [XMem++](https://github.com/max810/XMem2)) option for better controllability.

![overview](https://imgur.com/k84c965.jpg)

Demo video of the original tool:

[https://github.com/hkchengrex/Cutie/assets/7107196/83a8abd5-369e-41a9-bb91-d9cc1289af70](https://github.com/hkchengrex/Cutie/assets/7107196/83a8abd5-369e-41a9-bb91-d9cc1289af70)

Source: [https://raw.githubusercontent.com/hkchengrex/Cutie/main/docs/sources.txt](https://raw.githubusercontent.com/hkchengrex/Cutie/main/docs/sources.txt)

## Citation

CutiePlus is built upon Cutie. If you use CutiePlus, please cite the original Cutie paper (CVPR 2024):

```bibtex
@InProceedings{Cheng_2024_CVPR,
    author    = {Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
    title     = {Putting the Object Back into Video Object Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {3151-3161}
}
```

CutiePlus has no publication of its own. To also acknowledge the tool, see [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository").

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

The redistributed RITM code in `gui/ritm` is under its own MIT license, Copyright (c) 2021 Samsung Electronics Co., Ltd. ([`gui/ritm/LICENSE`](gui/ritm/LICENSE)). The pretrained weights are the original authors' and are downloaded from their release page.