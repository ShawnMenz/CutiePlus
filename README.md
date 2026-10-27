# CutiePlus

## Declaration

**CutiePlus is a modified version of [Cutie](https://github.com/hkchengrex/Cutie), the video object segmentation framework from *Putting the Object Back into Video Object Segmentation* (CVPR 2024, Highlight).**

All credit for the model and the original code belongs to the original authors:

[Ho Kei Cheng](https://hkchengrex.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Brian Price](https://www.brianpricephd.com/), [Joon-Young Lee](https://joonyoung-cv.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/) — University of Illinois Urbana-Champaign and Adobe.

[[Original repository]](https://github.com/hkchengrex/Cutie) [[arXiv]](https://arxiv.org/abs/2310.12982) [[PDF]](https://arxiv.org/pdf/2310.12982.pdf)

- CutiePlus is based on the original repository at commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a) (8 November 2024) and keeps its git history. Modified by Zexuan Meng.
- It is **not affiliated with or endorsed by** the original authors. Please report problems specific to CutiePlus in the [CutiePlus issue tracker](https://github.com/ShawnMenz/CutiePlus/issues), not to them.
- If you decide to use the pretrained Cutie weights, they are the original authors' work. They are downloaded from the original Cutie release and are not part of this repository.
- If you use CutiePlus, **please cite the original Cutie paper** (see [Citation](#citation)). CutiePlus itself is not a publication.
- Licensed under the MIT License, like the original ([License](#license)).


## What's different from Cutie?

The original Cutie ships a model and a GUI for interactive video segmentation. CutiePlus keeps both and extends Cutie with:

- **Task-based annotation workflows**: a sequence opens as a *task* with an editable clip interval and target description (several intervals optional, each with its own description) and optional pregenerated masks; a *task browser* lists the sequences of every subset of the input (a tab each, added with *Import*) with their status, annotators and target text, and opens them one at a time.
- **Anchor-aware mask propagation**: every human edit makes a frame an *anchor* that propagation never overwrites; an empty anchor means the target is absent and propagation stops there; propagation started inside a clip interval stays inside it; permanent memory is rebuilt from the anchors.
- **RGB/thermal visualization and annotation support**: a visible (RGB) / thermal toggle; clicks and propagation use the modality shown, with the same anchors (the two are aligned), and the frame slider marks every anchor.
- **Task locking and edit records**: a task is open in one window at a time, and every change records who made it (the signed-in user on a cooperation host; else `--labeler`, `$CUTIE_LABELER`, or the account name).
- **Review workflow**: not started, in progress, under review (*Save*, Ctrl+S), done. *Export* in the browser writes the dataset into `workspace/cutie_export/`: one clip per interval of every done sequence, with its frames, masks and a seqinfo naming the sequence it was cut from. A *Need Attention* toggle in the window flags a sequence for further attention (status *Attention* in the browser; it is not marked done while flagged), and *Discard* in the browser leaves a sequence out of the dataset (status *discarded*; *Restore* undoes it).
- **Cooperation on one computer with one GPU** (optional): several people label at once, the owner at the monitor and collaborators on remote desktops reached over SSH and VNC; each signs in with an identity and a role (Admin, Supervisor, Annotator), supervisors and admins manage collaborators in the task browser (*Manage Annotators*), and every model runs in one shared GPU worker with a queue. See [COOPERATION.md](docs/COOPERATION.md).
- **Undo/redo and improved interactive editing**: undo/redo of clicks, *Reset frame*, *Import mask* and anchor changes; the click model runs without a click limit.
- **A modern uv-based environment**: Python 3.12 and PyTorch with CUDA 13, with a lockfile.
- **GUI and robustness improvements for repeated annotation work**: zoom and scroll, a layout that adapts to any screen size, keyboard shortcuts shown on the buttons, a Tips window, colored status labels, a light/dark/eye-care theme button, a flat rounded "soft cards" look with a green accent (`gui/style.py`), and fixes.

For a detailed, file-level description of the changes relative to upstream Cutie commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a), see [MODIFICATIONS.md](docs/MODIFICATIONS.md).

## Installation

Tested on Ubuntu 24.04 with an RTX 4090 NVIDIA GPU. The environment is managed with [uv](https://docs.astral.sh/uv/): Python 3.12, and PyTorch with CUDA 13.0 (needs NVIDIA driver 580 or newer).

**Prerequisites:**

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

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