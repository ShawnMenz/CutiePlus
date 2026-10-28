# Training

This guide covers the upstream Cutie training pipeline retained in this repository. For dataset annotation, see the [workflow guide](WORKFLOW.md). Complete the [Quick Start](../README.md#quick-start), then install training dependencies:

```bash
uv sync --extra train
```

Run the examples from the repository root. Training requires CUDA GPUs and uses distributed execution through `torchrun`, including for a single GPU. The BURST conversion command below includes the `eval` extra for `pycocotools`.

## Prepare datasets

The default dataset paths in [`cutie/config/data/datasets.yaml`](../cutie/config/data/datasets.yaml) are relative to the repository root. All presets use FSS, DUTS-TR, DUTS-TE, ECSSD, BIG, and HRSOD for pre-training. Their main-training datasets are:

| Preset | Main-training datasets |
| --- | --- |
| `base` | DAVIS and YouTubeVOS |
| `with-mose` | DAVIS, YouTubeVOS, and MOSE |
| `mega` | DAVIS, YouTubeVOS, MOSE, BURST, and OVIS |

Prepare the datasets required by the enabled stages and chosen preset. For example:

```text
parent/
├── CutiePlus/
├── static/                    # fss, DUTS-TR, DUTS-TE, ecssd, BIG_small, HRSOD_small
├── DAVIS/2017/trainval/        # JPEGImages/480p and Annotations/480p
├── YouTube/train/             # JPEGImages and Annotations
├── BURST/train-vos/           # JPEGImages and Annotations
├── OVIS-VOS-train/            # JPEGImages and Annotations
└── MOSE/train/                # JPEGImages and Annotations
```

Consult the dataset configuration for all paths and override it for your layout. Dataset preparation and licensing are the responsibility of the user.

DEVA has a script for downloading some of these datasets: <https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/docs/TRAINING.md>.

To generate `train-vos` for BURST, use the script `scripts/convert_burst_to_vos_train.py` which extracts masks from the JSON file into the DAVIS/YouTubeVOS format for training:

```bash
uv run --extra eval python scripts/convert_burst_to_vos_train.py --json_path ../BURST/train/train.json --frames_path ../BURST/frames/train --output_path ../BURST/train-vos
```

For OVIS, prepare the `JPEGImages/` and `Annotations/` directories using a conversion tool such as [vis2vos](https://github.com/youtubevos/vis2vos).

Dataset references:

- DAVIS: https://davischallenge.org/
- YouTubeVOS: https://youtube-vos.org/
- BURST: https://github.com/Ali2500/BURST-benchmark
- MOSE: https://henghuiding.github.io/MOSE/
- LVOS: https://lingyihongfd.github.io/lvos.github.io/
- OVIS: https://songbai.site/ovis/

## Launch training

The example launches four processes on four CUDA GPUs. Adjust the process count and batch sizes for your hardware.

```bash
OMP_NUM_THREADS=4 uv run --extra train torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=example model=base data=base
```

- Change `nproc_per_node` to match the number of GPUs. The configured batch size is divided across the processes; adjust it to fit your hardware.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Choose `model=small` or `model=base`, and `data=base`, `data=with-mose`, or `data=mega` as appropriate for your datasets.
- Models, training checkpoints, logs, and visualizations are written under `output/<exp_id>/`. Weight files and resumable checkpoints are separate files; checkpoint filenames contain `_ckpt_`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To initialize from model weights, specify `weights=/path/to/model.pth`. To resume a training checkpoint, use `checkpoint=/path/to/checkpoint.pth` and enable the matching stage. For a main-training checkpoint, also set `pre_training.enabled=False`; otherwise the checkpoint is loaded into the first enabled stage.

When both stages are enabled, the final pre-training weights are transferred to main training automatically. The global batch size and worker count are divided by the number of processes. Choose values that remain positive after division; use a batch size divisible by the process count.
