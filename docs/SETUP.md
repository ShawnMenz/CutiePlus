# Setup and troubleshooting

Start with the [README Quick Start](../README.md#quick-start). This guide provides additional checks for the local desktop installation. For service installation and remote access, use [multi-annotator deployment](COOPERATION.md).

## Environment

The repository requires Python 3.12 and uses `uv.lock` to pin its Linux environment. PyTorch and torchvision are selected from the CUDA 13 wheel index in [`pyproject.toml`](../pyproject.toml). Use the project environment from the repository directory:

```bash
uv sync --locked
uv run python --version
uv run python interactive_demo.py --help
```

`uv run` selects the project environment without requiring shell activation. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for environment management. Run the GUI as your desktop user; use sudo only for the system-package and deployment commands that require it.

The local GUI selects CUDA when available, then MPS, then CPU. The committed dependency environment targets Linux; the device-selection branches do not establish Windows or macOS installation support. In cooperation mode, the GUI runs without CUDA access and uses the shared GPU worker.

### Check GPU access

```bash
nvidia-smi
uv run python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Architectures in this build:", torch.cuda.get_arch_list())
    x = torch.ones(1, device="cuda")
    print("CUDA tensor check:", (x + 1).item())
PY
```

For GPU inference, the availability check should be `True` and the tensor check should print `2.0`. CUDA 13 requires the NVIDIA 580 driver family or newer; see [NVIDIA's compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The GPU must also be supported by the installed PyTorch build. Successful driver detection alone does not establish that a model can run on the device.

### Check Qt and the click-model extension

The GUI uses PySide6. This check initializes Qt without opening a window:

```bash
QT_QPA_PLATFORM=offscreen uv run python - <<'PY'
from PySide6.QtWidgets import QApplication

app = QApplication([])
print("Qt initialized")
PY
```

RITM computes click distance maps through a Cython extension compiled on first import. Check that the compiler and Python environment can load it:

```bash
uv run python -c "from gui.ritm.utils.cython import get_dist_maps; print('Click-model extension loaded')"
```

These checks verify imports and extension loading. An interactive session still needs an accessible graphical desktop.

## Model files

Download both checkpoints before opening a task:

```bash
uv run python cutie/utils/download_models.py
```

The downloader stores them in the repository's `weights/` directory. GUI model loading calls the same downloader, checks existing files against the expected MD5 values, and downloads missing or mismatched files. A completed setup with matching files can load these weights without downloading them again.

For a manual transfer, use the exact files linked below and place them in `weights/`:

| File | Purpose | Expected MD5 |
| --- | --- | --- |
| [cutie-base-mega.pth](https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth) | Cutie mask propagation | `a6071de6136982e396851903ab4c083a` |
| [coco_lvis_h18_itermask.pth](https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth) | RITM click segmentation | `6fb97de7ea32f4856f2e63d146a09f31` |

```bash
md5sum weights/cutie-base-mega.pth weights/coco_lvis_h18_itermask.pth
```

These checksums match the downloader's integrity checks. The files originate from the original Cutie release.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Start a new shell after installing uv, or run `source "$HOME/.local/bin/env"` when using its default installer location. |
| Python version is rejected | Use `uv sync --locked` from the checkout. The project requires Python 3.12 rather than the system Python. |
| `CUDA available: False` | Run `nvidia-smi`, check driver compatibility, and check whether `CUDA_VISIBLE_DEVICES` hides the GPU. The standalone GUI uses CPU when CUDA is unavailable. |
| CUDA reports an unsupported kernel or device | Check the GPU architecture against the installed PyTorch build; a newer driver alone cannot add GPU architectures to that build. |
| Qt reports that it cannot connect to a display | Launch from a desktop session. For remote annotators, use the [SSH/VNC deployment](COOPERATION.md). The offscreen check above is for diagnostics. |
| Qt cannot load the `xcb` platform plugin | Check the system libraries required by [Qt's X11 plugin](https://doc.qt.io/qt-6.10/linux-requirements.html). On Ubuntu/Debian, `libxcb-cursor0` and `libxkbcommon-x11-0` are among the runtime packages needed. Use the command below to identify unresolved libraries. |
| Importing `get_dist_maps` fails with a compiler error | Install a C/C++ toolchain (`build-essential` on Ubuntu/Debian), then rerun the extension check above. |
| Model loading fails or repeatedly downloads a file | Compare both MD5 values above. Retry the download with a working connection, or manually transfer the matching files. |
| The browser shows no tasks | Check that each subset is below the selected task root and contains the [required input layout](WORKFLOW.md#tasks-and-their-files). For name-only categories, complete **Import → Save**. |

To inspect missing X11 plugin libraries in the installed PySide6 package:

```bash
uv run python - <<'PY'
from pathlib import Path
import subprocess
from PySide6.QtCore import QLibraryInfo

plugins = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))
subprocess.run(["ldd", str(plugins / "platforms" / "libqxcb.so")], check=True)
PY
```

Resolve entries marked `not found` through your distribution's package manager. If the cause remains unclear, launch once with `QT_DEBUG_PLUGINS=1 uv run python interactive_demo.py` and inspect the terminal output.

For task locking, review status, export failures, and incomplete modality pairs, see the [workflow guide](WORKFLOW.md). Report reproducible problems through the [CutiePlus issue tracker](https://github.com/ShawnMenz/CutiePlus/issues), including the command, terminal error, Python/PyTorch versions, and whether the failure occurs in local or cooperation mode.
