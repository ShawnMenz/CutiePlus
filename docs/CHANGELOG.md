# Changelog

This changelog covers CutiePlus releases. CutiePlus is developed independently from Cutie commit [`ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a). The inherited Git history records the upstream source; subsequent releases use CutiePlus's own version numbers.

## [1.0.0](https://github.com/ShawnMenz/CutiePlus/releases/tag/v1.0.0) — 2026-10-28

First official release of CutiePlus.

### Annotation and dataset construction

- Task browser for importing subsets, opening single-target sequences, and tracking annotation progress.
- Aligned RGB and thermal views sharing the same target masks, with clicks and propagation using the displayed modality.
- Human corrections recorded as anchors, preserved during propagation; empty anchors mark target absence and stop propagation.
- Multiple inclusive clip intervals per sequence, each with its own category and description.
- Review submission, completion, attention flags, discard and restore actions, and annotator records.
- Dataset export of completed clips with source frames, binary masks, and provenance metadata, plus optional demo-video output.
- Undo and redo for supported frame edits, navigation shortcuts, zoom, playback, and interface themes.

### Optional multi-annotator deployment

- Application accounts and roles, managed SSH/VNC desktops, and account-management controls.
- One shared GPU worker with a persistent propagation queue and revision checks when applying results.
- Service audit records, database backup and recovery tools, and systemd deployment scripts.

### Installation and documentation

- Python 3.12 environment managed with `uv.lock`, with PyTorch CUDA 13 wheels.
- Optional dependency extras for the retained Cutie training and benchmark-evaluation workflows.
- Step-by-step setup, input/output reference, annotation workflow, deployment, and architecture guides.
- Software citation metadata in `CITATION.cff`, with separate attribution to the original Cutie paper and code.

### Requirements and scope

- The committed environment targets Linux. Local annotation requires a graphical desktop; GPU use requires compatible NVIDIA hardware and drivers.
- RGB and thermal frames must be aligned before import. Each task labels one target.
- Remote annotators share a Linux account; application roles do not provide operating-system isolation. Read the [deployment guide](COOPERATION.md) before enabling the service.
- The original Cutie and bundled RITM license notices are retained. Model checkpoints are downloaded separately from the upstream Cutie release.

See the [README](../README.md) for installation and the [changes from Cutie](MODIFICATIONS.md) for the upstream comparison.
