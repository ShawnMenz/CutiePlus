# Changes from Cutie

CutiePlus builds on [Cutie](https://github.com/hkchengrex/Cutie), starting from upstream commit [`ec5cdd4`](https://github.com/hkchengrex/Cutie/commit/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a). It adapts Cutie's interactive video object segmentation demo for single-target RGB–thermal dataset annotation and review. The original Cutie authors developed the underlying model; see the [README](../README.md#attribution-and-license) for attribution and licensing.

## Annotation workflow

| Capability | CutiePlus |
| --- | --- |
| Task management | A task browser organizes subsets and sequences, tracks annotation and review status, and prevents the same task from being opened in two windows under the same account. |
| RGB and thermal | Aligned visual and thermal frames are displayed as alternative views of the same target. Clicks and propagation use the displayed modality. |
| Corrections | Human edits become anchors; propagation preserves anchors and stops at an empty anchor. Users can correct results across the sequence and navigate between anchors. |
| Dataset construction | Each sequence can have one or more non-overlapping intervals with a category and description. Export assembles completed intervals into a dataset with frames, masks, and metadata. |
| Review | Tasks can be submitted for review, marked done, flagged for attention, or discarded. Edits and review actions record the responsible annotator. |
| Shared inference | An optional cooperation service supports remote annotators and queues inference through a shared GPU worker. |

A task labels **one target**. This differs from the original demo's multi-object interface. See the [workflow guide](WORKFLOW.md) for data preparation, annotation, review, and export; see the [cooperation guide](COOPERATION.md) for deployment and its trust boundaries.

## Inference and interaction

CutiePlus retains Cutie for propagation and RITM for click-based segmentation. It uses NoBRS without a click limit so each click reaches the network, and rebuilds permanent memory from selected anchors before propagation. A small-target policy can process a sequence at a larger internal resolution when the estimated target area falls below its threshold. Processing time and accuracy depend on the data, resolution, and hardware.

The interface also adds undo and redo for supported edits, modality switching, interval navigation, playback, keyboard shortcuts, and themes. Propagation is not undoable. The [labeling guide](../gui/TIPS.md) describes the controls.

## Packaging and compatibility

The project uses a Python 3.12 `uv` environment with PyTorch CUDA 13 wheels. Training and benchmark evaluation remain available as optional dependency extras; their instructions are inherited from upstream and are documented separately in [training](TRAINING.md) and [evaluation](EVALUATION.md). The default GUI workflow, input layout, and output format are specific to CutiePlus; see the [workflow guide](WORKFLOW.md) before migrating data or scripts from the original demo.
