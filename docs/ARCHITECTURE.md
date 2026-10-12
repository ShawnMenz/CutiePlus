# CutiePlus architecture before the shared service (review, October 2026)

This document records how the labeling tool worked when the shared GPU service was started
(commit `039c919`). It is the reference for the refactoring described in
[SHARED_SERVICE.md](SHARED_SERVICE.md); the [modifications](MODIFICATIONS.md) list what CutiePlus
changed relative to the original Cutie.

## Processes

Every annotator connects through an SSH tunnel to their own TigerVNC desktop (`:2`, `:3`, ... on
ports 5902, 5903, ...) under the one shared Linux account, and runs `interactive_demo.py --tasks
<task root>` there. Each run is one PySide6 process with its own task browser
(`gui/task_browser.py`). The browser loads Cutie and RITM on the first *Open* and keeps them for
the life of the process, so N annotators hold N copies of both models on the one GPU (about
450-660 MB each when idle, measured with `nvidia-smi` on the lab host).

Annotators are identified from `DISPLAY`: screen `:N` is VNC port `5900 + N`, and
`<task root>/annotators.json` names the annotator of each port (`gui/annotators.py`).

## Model initialization and CUDA

- `interactive_demo.py` picks `cuda`, `mps` or `cpu` and disables autograd.
- `gui/main_controller.py: load_networks` builds `CUTIE(cfg)`, loads `weights/cutie-base-mega.pth`
  and builds `ClickController` (`gui/click_controller.py`), which loads the RITM HRNet-18
  (`weights/coco_lvis_h18_itermask.pth`) through `gui/ritm/inference/utils.load_is_model`
  (distance maps in Cython on the CPU, `cpu_dist_maps=True`).
- `download_models_if_needed` checks the MD5 of both weight files at every load.
- **Implicit CUDA initialization at import:** `gui/interactive_utils.py` creates its color tables
  with `.to(device)` at import time, choosing CUDA whenever a GPU is visible. Importing the
  module therefore creates a CUDA context, and the CPU mode fails on a host with a GPU (the
  colors are on CUDA, the images on the CPU). Found while recording the baseline tests.
- `MainController.update_gpu_gauges` calls `torch.cuda.mem_get_info` every 2 s. The "torch"
  gauge shows `max_memory_allocated`, the process's peak, not its current use.

## Mutable inference state

None of the following can be shared between users or between propagation jobs.

### Cutie (`cutie/inference/inference_core.py`)

One `InferenceCore` per labeling window (`MainController.processor`), holding:

- `curr_ti`, `last_mem_ti`: the frame counter that decides memory frames (`mem_every`) and
  sensory updates (`stagger_ti`, computed once from `mem_every` at construction).
- `ObjectManager`: object ids to tensor positions. It survives `clear_memory()`.
- `MemoryManager` (`memory_manager.py`): working and long-term key/value stores
  (`kv_memory_store.py`), each with a permanent part at the front of every bucket; sensory
  memory per object; object memory (`obj_v`); usage counters for consolidation.
- `ImageFeatureStore` (`image_feature_store.py`): encoded features per frame index, deleted after
  each step.
- `last_mask` (the previous output, an input of the next memory read) and `pad`.
- `max_internal_size` and `upscale_to_internal_size`, set per window by the small-target policy.

### RITM (`gui/click_controller.py`, `gui/ritm/controller.py`, `gui/interaction.py`)

One `ClickController` per process, reused by every task the browser opens:

- `anchored`: whether the predictor holds the current image.
- `InteractiveController`: `image`, `clicker` (ordered clicks), `predictor` (`BasePredictor`:
  `original_image`, `prev_prediction`, the transforms' state: zoom-in region, previous
  probabilities, `LimitLongestSide` crop), `states` (predictor and clicker state before every
  click, for undo, on the GPU) and `probs_history` (every click's prediction, on the GPU).
  `add_click` calls `torch.cuda.empty_cache()` after every click.
- `ClickInteraction` (per window): the previous probabilities (all objects), the target object,
  the ordered clicks drawn as dots, and the last object mask.

GPU memory of one interaction grows with its clicks: a full-resolution probability map in
`states` and one in `probs_history` per click (about 16 MB per click at 1920x1080).

## Temporal memory and permanent memory

- Memory frames every `mem_every` (5) frames and at every frame given a mask; sensory updates on
  `stagger_updates` frames; working memory of 5-10 frames, consolidated into at most 10,000
  long-term tokens (`gui_config.yaml`).
- Task mode rebuilds the permanent memory before every propagation
  (`MainController.rebuild_permanent_memory`): `clear_memory`, then every mask anchor as a
  permanent memory frame, nearest to the start frame first, up to `permanent_anchor_limit()`.
- The limit comes from `MEMORY_READ_BUDGET = 4e8` memory-token x query-token comparisons, a
  constant described as "~8 GB of GPU memory". It assumes a GPU with that much room and does
  not look at the GPU or at other processes. At 480 px on 16:9 frames it allows 136 anchors; at
  the small-target size (960 px) it allows one.
- Anchors left out of the permanent memory are fed as ordinary memory frames when reached.

## Small targets

At window open (`configure_task_resolution`), the median area of the mask anchors (else of up to
25 prefill masks) at the default internal size (480 px) decides the processing size: below
1,500 px the sequence is processed at 960 px with upscaling. It is decided once per window.

## Video propagation (`MainController.on_propagate`)

Runs in the Qt thread and keeps the window responsive with `processEvents`:

1. Refused when the current frame is an empty anchor.
2. Task mode: rebuild the permanent memory (above).
3. `clear_sensory_memory`; one step on the start frame with the shown probabilities as the mask
   (not saved).
4. For each following frame (read by `PropagationReader` through a `DataLoader` with
   `num_read_workers` processes, always in the anchor modality):
   - an empty anchor stops the run before the frame is processed;
   - a mask anchor is fed with its mask (`step(image, mask)`) and its file is left alone;
   - any other frame is segmented and its mask is queued for saving.
5. At the end (or when paused), `task.touch()` if anything was written.

The frame number is derived from the start frame and the step count, so frame keys pressed
during a run cannot redirect the saves.

## Caches

- `ResourceManager.get_image` and `get_mask`: LRU caches of `buffer_size` (20) entries; a mask
  entry is invalidated when its frame is saved or deleted. A missing mask falls back to the
  prefill (binarized), then to nothing.
- `ImageFeatureStore`: per-frame features, deleted after each step.

## Locking and identity

- `TaskLock` (`gui/task_state.py`): an exclusive `flock` on `<task>/lock.json`, released on close
  or by the kernel when the process dies; the file describes the holder (labeler, display, pid).
- The browser takes the same lock while it exports a task that is not open in its window.
- The labeler is `--labeler`, else the annotator of the screen's port, else `$CUTIE_LABELER`,
  else the account name.

## Official annotation data and the functions that change it

| Data | Written by |
|---|---|
| `masks/<frame>.png` (palette PNG) | `ResourceManager.save_mask` (background threads) from clicks, *Reset frame/object*, *Import mask*, *C* on a frame without a file, propagation; `delete_mask` (*Delete* anchor); `apply_frame_state` (undo/redo, written directly); `clear_sequence_masks` (*Reset all*) |
| `anchors.json` | `TaskState.set_anchor` (`mark_anchor` after every human edit, `commit_anchor`), `remove_anchor`, `restore_anchor` (undo/redo), `clear_mask_references` |
| `prefill.json` | `clear_mask_references` only (created by the pipeline) |
| `progress.json` | `TaskState.touch` (every change), `submit_for_review` (*Save binary PNGs*), `mark_done` (browser) |
| `interval.json`, `description.json` | `save_interval_pairs`, `on_interval_description_change`, `on_description_change`, `on_export_binary` |
| `binary_masks/`, `interval.txt`, `description.txt` | `on_export_binary` |
| export trees `cutie_masks_<modality>/<seq>` | the external export command (`task_set.json`), run by the window or the browser; removed by *Reset all* |

`visualization/` and `soft_masks/` are optional derived outputs, off by default.

JSON files are written to a temporary file and renamed (`_write_json`), without `fsync`. Before
this branch, mask PNGs were written in place.

## Undo and redo

`EditHistory` keeps 100 frame edits; an edit is the frame's mask file bytes and anchor entry
before and after. The clicks of one interaction form one edit; *Ctrl+Z* first takes back single
clicks (RITM's own state stack), then whole edits. Propagation is not undoable.

## Export and review

Not started, in progress (any change), under review (*Save binary PNGs*), done (*Mark done* in
the browser, which exports first when the export is missing or older than the last change). The
export runs the command of `task_set.json` with `QProcess`; `export_is_current` compares the
export's `exported_at` with `progress.json`'s `last_edited` (1 s resolution).

## GUI-to-GPU dependencies

`MainController` uses the GPU directly in: `click_fn`, `on_undo` (click level), `on_propagate`,
`rebuild_permanent_memory`, `on_commit` (non-task mode), `convert_current_image_mask_torch`,
`model_image_torch`, `update_current_image_fast` (visualization on the GPU),
`_try_load_layer`, the memory buttons and gauges, and `finish_close`
(`torch.cuda.empty_cache`). `gui/interactive_utils.py` holds GPU color tables.
`gui/task_browser.py` loads the models.

## Problems found in the review

1. **Save worker:** an exception in a save thread (disk full, permissions) ended the thread
   without `task_done()`. `save_queue.join()`, called on every frame change, then waited
   forever, and once the queue filled, every save blocked: the window froze. Fixed on this
   branch: `task_done()` always runs, mask files are written to a temporary file and renamed,
   and failures are reported in the window (`ResourceManager.check_saves`). Covered by
   `tests/test_resource_manager.py`.
2. CUDA is initialized by importing `gui/interactive_utils.py` (above).
3. A mask anchor without a mask file (and without a prefill) makes every propagation fail in
   `rebuild_permanent_memory` (`torch.from_numpy(None)`), after the run button has already
   switched to *Pause*.
4. `MEMORY_READ_BUDGET` encodes one GPU's capacity and ignores other processes.
5. Every process loads its own models; nothing coordinates GPU memory between annotators.
6. RITM empties the CUDA cache after every click, which also discards the cached blocks of any
   other work in the process.
