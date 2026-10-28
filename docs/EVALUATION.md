# Inference and evaluation

The benchmark entry point runs Cutie video object segmentation from supplied masks. It uses the upstream model and dataset readers; it does not run the CutiePlus GUI, review workflow, or joint RGB–thermal inference. For annotation and dataset export, see the [workflow guide](WORKFLOW.md).

## Setup

Complete the [Quick Start](../README.md#quick-start), including the model download, then install evaluation dependencies:

```bash
uv sync --extra eval
```

Run commands from the repository root. `cutie/eval_vos.py` requires a CUDA GPU. The examples use the downloaded `weights/cutie-base-mega.pth`; the configuration's default path is `output/cutie-base-mega.pth`, so pass `weights` explicitly. Use `model=small` only with a checkpoint for the small model.

## Run a benchmark dataset

Configure the dataset directories in [`eval_config.yaml`](../cutie/config/eval_config.yaml) or override them on the command line. For DAVIS 2017 validation:

```bash
uv run --extra eval python cutie/eval_vos.py dataset=d17-val weights=weights/cutie-base-mega.pth model=base exp_id=davis17
```

The available dataset names are `d16-val`, `d17-val`, `d17-test-dev`, `y18-val`, `y19-val`, `mose-val`, `burst-val`, `burst-test`, `lvos-val`, `lvos-test`, and `generic`. Each selects its own paths, resize policy, mask-use policy, and memory settings. See the [training guide](TRAINING.md) for the default dataset layout.

For LVOS validation, first create masks containing the first appearance of each object, as expected by its preset:

```bash
uv run --extra eval python scripts/data/preprocess_lvos.py ../LVOS/valid/Annotations ../LVOS/valid/Annotations_first_only
```

The larger-resolution "plus" configuration also adjusts temporal memory settings and DAVIS image directories:

```bash
uv run --extra eval python cutie/eval_vos.py --config-name eval_plus_config.yaml dataset=d17-val weights=weights/cutie-base-mega.pth exp_id=davis17_plus
```

Ensure that the full-resolution DAVIS images and the 480p size-reference directory required by [`eval_plus_config.yaml`](../cutie/config/eval_plus_config.yaml) are present.

## Run your own sequences

Use one subdirectory per sequence in both the image and mask roots. Use zero-padded frame names and matching `.png` mask basenames:

```text
images/
└── sequence_01/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
masks/
└── sequence_01/
    └── 000001.png
```

The loader discovers sequence names from the mask root and processes image filenames in sorted order. Keep only frame images in each image folder. Each sequence needs at least one initialization mask. Pixels with value 0 are background; nonzero values identify objects. Palette (`P`) and grayscale (`L`) PNGs store integer IDs; RGB PNGs encode long IDs. For a single target, a grayscale mask with values 0 and 1 is sufficient. Mask colors are for display; the stored IDs determine object identity.

```bash
uv run --extra eval python cutie/eval_vos.py dataset=generic image_directory=/path/to/images mask_directory=/path/to/masks weights=weights/cutie-base-mega.pth size=480 exp_id=my_sequences
```

Generic mode uses the first supplied mask by default and skips frames before that mask. Set `use_all_masks=True` to read later available masks, including introductions of new objects. `size` limits the input's shorter edge; `size=-1` keeps the input resolution. Predicted masks are restored to the original output size.

## Outputs and metrics

The default output root is `../output/<exp_id>/<dataset>/`. Override it with `output_dir=/path/to/results`. Mask predictions are written under `Annotations/<sequence>/`; `save_scores=True` and `visualize=True` enable additional score and visualization outputs. BURST produces a prediction JSON; dataset-specific packaging is performed at the end of inference.

The script logs inference time, frame count, throughput, and peak Torch CUDA allocation. It produces predictions rather than benchmark accuracy scores. Use the relevant benchmark's evaluation package or submission service:

- [DAVIS evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) or [vos-benchmark](https://github.com/hkchengrex/vos-benchmark) for DAVIS.
- [YouTubeVOS](https://youtube-vos.org/) for YouTubeVOS evaluation.
- [BURST benchmark](https://github.com/Ali2500/BURST-benchmark) for BURST.
- [LVOS evaluation](https://github.com/LingyiHongfd/lvos-evaluation) for LVOS.
- [MOSE](https://henghuiding.github.io/MOSE/) for MOSE.

The upstream [Cutie release](https://github.com/hkchengrex/Cutie/releases/tag/v1.0) provides additional pretrained checkpoints. When reporting results, record the checkpoint, model configuration, dataset preset, and overrides used.
