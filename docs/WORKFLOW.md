# Annotation workflow and data reference

This guide covers task-based annotation, review, and dataset export. Complete the [Quick Start](../README.md#quick-start) first. Open **Tips** (F1) in the labeling window for the [complete labeling guide](../gui/TIPS.md). Administrators of a shared GPU host should also read the [multi-annotator deployment guide](COOPERATION.md).

## Annotate a sequence

1. Prepare a subset using the [input layout below](#tasks-and-their-files). Each sequence is a task for one target. Align paired RGB and thermal frames before importing them; CutiePlus does not perform registration.
2. Run `uv run python interactive_demo.py` from the repository directory. Select **Import**, check the listed subsets, then select **Save**. To import an external subset, select **+ Add a subset** and choose its folder and tab name. Import moves external folders and fills missing category colors, as described under [Import behavior](#import-behavior).
3. Select a sequence and choose **Open**. Choose the target category and enter a description. Left-click the target and right-click the background to refine the mask. Human mask edits become anchors automatically; press **C** to accept an already correct mask as an anchor.
4. Press **F** to propagate forward or **B** to propagate backward. Inspect the results and correct frames as needed. Press **T** to change the displayed modality; clicks and propagation use that modality and the same target masks.
5. Set the clip's start and end frames. Use **+ Add interval** for additional clips. Each interval has its own category and description. Frame ranges are inclusive and cannot overlap.
6. Press **Ctrl+S** to submit the sequence for review. After checking all intervals, choose **Mark done** in the task browser. If you label and review your own work, **Ctrl+Q** saves, marks the task done after confirmation, and closes its window.
7. Close the labeling windows for all sequences to be exported, then choose **Export** in the browser. The export includes every interval of every done sequence across all tabs. The default destination is `workspace/cutie_export/`.

Masks are saved as you work. **Save** also commits typed interval and description changes and submits the task for review. Reopen the same task to resume its saved work.

## Tasks and their files

The default input root is `workspace/cutie_tasks/` inside the repository. Each immediate child is a **subset**, displayed as a browser tab. Each sequence within that subset is a single-target task:

```text
workspace/cutie_tasks/
└── example/
    ├── visual/
    │   └── sequence_01/
    │       ├── 000001.jpg
    │       └── 000002.jpg
    ├── thermal/
    │   └── sequence_01/
    │       ├── 000001.jpg
    │       └── 000002.jpg
    ├── categories.json
    ├── targets.json                       # optional
    ├── annotation/                        # optional pregenerated masks
    │   └── sequence_01/
    │       └── 000001.png
    └── seqinfo/                           # optional source metadata
        └── sequence_01_seqinfo.ini
```

### Frames and modalities

Both `visual/` and `thermal/` directories are required by the Import dialog. A sequence present in only one modality produces a warning and can still be imported. Use paired sequences for RGB–thermal annotation.

- Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.bmp`.
- Frames are ordered by filename. Use zero-padded names such as `000001.jpg` to preserve chronological order.
- Modalities and masks are matched by the filename without its extension. For example, `visual/sequence_01/000001.jpg` and `thermal/sequence_01/000001.png` correspond to the same frame. Use only one file per frame basename in a directory.
- Paired frames must already be spatially aligned and use the same pixel dimensions, so one mask fits both. Keep the frame size consistent throughout a sequence.
- A sequence uses its visual frame list when available, otherwise its thermal frame list. The window falls back to the default modality when the other modality lacks a frame. Dataset export rejects an interval if any included frame is missing from an existing modality directory.

Import checks the directory structure and JSON files; successful import does not establish that all images are readable, aligned, or complete.

### Categories

`categories.json` is required. It can list unique category names:

```json
["car", "person"]
```

On **Import → Save**, names without colors receive colors and the file is updated. You can also provide colors explicitly:

```json
[
  {"name": "car", "color": "#00CF00"},
  {"name": "person", "color": "#E57A00"}
]
```

Colors must use the `#RRGGBB` format. CutiePlus reuses colors for matching category names in other subsets where possible. The selected interval's category controls its mask color.

### Target descriptions

The optional `targets.json` maps sequence names to their initial target category and description:

```json
{
  "sequence_01": {
    "category": "car",
    "description": "the car on the left"
  }
}
```

Without an entry, the sequence starts with no category (displayed as `null`) and an empty description. Choose the category and write a description in the labeling window. Subsequent edits are stored with the task's work, without changing `targets.json`.

### Pregenerated masks and source metadata

Place optional starting masks under `annotation/<sequence>/`, matched to frame basenames. Use single-channel or palette PNG masks at the frame's dimensions: zero is background and positive values represent the target. These masks are displayed until a frame has a saved mask of its own. Accept a correct starting mask with **C** or refine it with clicks before propagating.

The labeling window's **Advanced → Import mask** control uses a different convention: it expects a two-dimensional mask with values **0 and 1**, and resizes it to the current canvas. Convert an exported 0/255 mask to 0/1 before using this control. The input subset's `annotation/` loader accepts either convention because it binarizes positive values.

Optional `seqinfo/<sequence>_seqinfo.ini` files can supply a source `dataset` field for export. If these files contain a `[Sequence]` section with matching `name`, `category`, and `description` fields, generate target files with:

```bash
uv run python -m scripts.extract_targets --input workspace/cutie_tasks
```

This command writes `targets.json` and `categories.json` for subsets with source metadata. Review its messages: invalid entries are reported and omitted, and existing target files in those subsets are rewritten. It does not import source clip intervals.

## Import behavior

**Import** lists existing subsets and validates each row. **Save** is enabled once all rows pass validation. Warnings do not block import.

- A subset already at `<input root>/<tab>` stays in place.
- An external folder is **moved** to `<input root>/<tab>`. Across filesystems, it is copied and then the source is deleted after a successful copy. Keep a separate copy if you need to retain the original location.
- If `visual/`, `thermal/`, and the other subset files sit directly in the input root, Import offers to organize them into `main/`.
- Missing category colors are generated and written to each affected `categories.json`.

During annotation, frames and pregenerated masks are read in place. Task edits are written to the separate work directory.

## Anchors, propagation, and intervals

An **anchor** records a human decision about a frame. Clicks, imported masks, and **C** can establish an anchor. **Reset frame** creates an **empty anchor**, meaning that the target is absent. The frame label distinguishes anchors, empty anchors, propagated masks, pregenerated masks, and frames without masks.

Propagation preserves anchors and stops at an empty anchor. It uses the currently displayed modality. Starting inside a saved interval limits propagation to that interval; starting outside every interval allows propagation toward the beginning or end of the sequence. Permanent memory is rebuilt from selected anchors before a run.

The first interval uses the sequence's category and description. Added intervals begin with the same text and category, then retain their own values. Use **{** and **}** to navigate interval starts and **Shift+R** to preview intervals. Frame numbers are one-based positions in the ordered sequence, not values parsed from image filenames. For example, frames 15–650 contain 636 frames. If no interval has been saved, the whole sequence is used.

Undo and redo cover supported clicks, frame resets, mask imports, and anchor changes. Propagation and **Reset all frame masks** cannot be undone. Resetting all masks removes the task's saved masks and anchors; original pregenerated masks become visible again.

## Review and task ownership

| State | How it is reached |
| --- | --- |
| Not started | No edit or review has been recorded. |
| In progress | A mask, anchor, interval, category, or description changes. |
| Under review | **Save** or **Ctrl+S** submits the sequence. |
| Done | **Mark done** accepts a submitted sequence. |

An edit clears the previous review and done marks. **Ctrl+Q** combines submission and acceptance for self-reviewed work. [Video output](#demo-videos) also accepts the task after successful rendering if it has not changed meanwhile.

**Need Attention** flags a sequence without changing its underlying review state. It prevents marking the task done until cleared. A previously done task flagged afterward remains eligible for export, with a warning. **Discard** removes the done mark and excludes the sequence from export while keeping its masks and anchors. **Restore** removes the discard flag; the task must be reviewed and marked done again.

A task can be held by one window at a time within the same Linux account. **Open by** shows the holder. Close its window before opening it elsewhere or exporting it; a process that exits releases its lock. This is a same-host workflow, not distributed task locking across computers.

The annotator name recorded on edits comes from the signed-in identity in shared mode. In standalone mode, the order is `--labeler`, then `CUTIE_LABELER`, then the operating-system account name. The browser's **Annotators** column lists editors and reviewers who marked a task done.

## Work directories and custom paths

Default task data is organized as follows:

| Directory | Contents |
| --- | --- |
| `workspace/cutie_tasks/` | Input subsets, frames, categories, and optional starting metadata. |
| `workspace/cutie_output/<subset>/<sequence>/` | Saved `masks/` and `metadata.json` for annotation progress. |
| `workspace/cutie_export/` | The completed dataset assembled by **Export**. |
| `workspace/cutie_demo/<subset>/` | Optional rendered demo videos. |

`metadata.json` records the source, anchors, intervals, edited category and description, review actions, attention/discard flags, and a mask revision. Keep it with `masks/` to resume work correctly. Back up both the inputs and work directories; a dataset export does not preserve the full editing history.

## Launch options

Run commands from the repository directory. With no input arguments, the application opens the task browser at `workspace/cutie_tasks/`.

| Option | Purpose |
| --- | --- |
| `--tasks <root>` | Open the task browser for an input root containing subset folders. |
| `--task_split <subset>` | Select the initial browser tab. |
| `--labeler <name>` | Set the recorded annotator name in standalone mode. A signed-in identity takes precedence on a shared host. |
| `--images <folder>` | Open a plain workspace initialized from one image sequence. |
| `--video <file>` | Open a plain workspace initialized by extracting video frames. |
| `--workspace <folder>` | Choose the plain workspace to initialize or resume. |

For a task root outside the repository:

```bash
uv run python interactive_demo.py --tasks /data/project/cutie_tasks --task_split example --labeler alice
```

Work, dataset exports, and demo videos are placed **beside the input root**, as `/data/project/cutie_output/`, `/data/project/cutie_export/`, and `/data/project/cutie_demo/`. Input roots with the same parent share those output directories, so use a separate parent directory for independent projects. The `--workspace` option does not relocate task-browser outputs.

### Plain workspaces

For a single image sequence or video without the task browser:

```bash
uv run python interactive_demo.py --images /path/to/frames --workspace workspace/single_sequence
uv run python interactive_demo.py --video /path/to/video.mp4 --workspace workspace/single_video
```

These examples are alternatives. Use separate workspace folders for different inputs. The application copies/resizes images or extracts video frames into the workspace's `images/` directory. An existing `images/` directory takes precedence over `--images` and `--video`; resume it with:

```bash
uv run python interactive_demo.py --workspace workspace/single_sequence
```

Existing `masks/` are loaded when that workspace is reopened. Plain workspaces do not use subset categories, task review states, or the browser's interval-based dataset export. **Ctrl+S** exports binary masks to the workspace's `binary_masks/` directory, and the Advanced controls retain the original visualization export options. A plain workspace still labels one target.

On a host with the cooperation service configured, use the task browser. Plain workspace launches are refused because they would run inference independently of the shared GPU worker.

## Dataset export

**Export** includes all done sequences across tabs, regardless of the current filter or selected tab. Each saved interval becomes a separate clip; a single-modality sequence exports only its available modality.

```text
workspace/cutie_export/
├── visual/<clip>/000001.jpg
├── thermal/<clip>/000001.jpg
├── annotation/<clip>/000001.png
└── seqinfo/<clip>_seqinfo.ini
```

Clip names are numbered from 1, in subset order, sequence-name order, and interval-start order. They are zero-padded to the number of digits in the clip count: 3,000 clips produce `0001` through `3000`. Frames within each clip are renumbered from `000001`; image extensions are preserved.

Masks are grayscale PNGs containing 0 for background and 255 for the target. Each frame uses its saved mask, otherwise its pregenerated mask, otherwise an empty mask. Review every frame before completion: export does not require every frame to have a manually reviewed mask.

The `[Sequence]` metadata includes the interval's category and description, frame count and image dimensions, and source fields:

- `originalSource`: `<subset>/<sequence>`, shared by clips from the same sequence.
- `intervalStart` and `intervalEnd`: the inclusive source frame positions.
- `dataset`: copied from the input sequence's optional source metadata.

Export asks for confirmation and locks each included task. It copies source images, so allow space for the new dataset alongside any existing export. It builds a temporary export and replaces the previous export only after completion; cancellation or a write failure keeps the previous export.

## Demo videos

In the labeling window, **Output video** (**Ctrl+E**) renders one MP4 per interval using the current visualization mode and **Output FPS**. Output paths follow:

```text
workspace/cutie_demo/<subset>/<sequence>_<start>-<end>_<mode>.mp4
```

For paired data, RGB and thermal views are shown side by side with masks, a frame counter, and the interval's category and description. Output saves first and marks the task done after all videos are written, provided the task has not changed during rendering. The window remains open. Use **Cancel output** to stop rendering. This is a completion action; review the intervals and masks before using it.

## Working on a shared GPU host

When the cooperation service is installed, sign in with your application account. Your propagation may wait in the shared queue; the window remains editable while waiting. If you edit after requesting propagation, results may be held for review rather than applied immediately. **GPU queue** shows job status and available cancel/retry actions.

For account setup, SSH/VNC connections, administration, and the shared-account trust model, see [Multi-annotator deployment](COOPERATION.md).
