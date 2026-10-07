# Shared single-GPU annotation service: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (the owner asked for
> implementation in this session, without stopping after planning). Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** several CutiePlus GUI sessions (one per TigerVNC desktop, same Linux account) share one
GPU worker that holds the only Cutie and RITM models, through a CPU coordinator that owns sessions,
task leases, scheduling and every write of official annotation data.

**Architecture:** Qt clients talk to a coordinator over a Unix domain socket (length-prefixed JSON
header plus raw array buffers). The coordinator (no CUDA) keeps SQLite metadata and immutable mask
objects, keeps the legacy task files (`masks/`, `anchors.json`, `progress.json`, ...) as views for
the export pipeline and the task browser, schedules work, and drives one spawned GPU worker process
over a `multiprocessing` pipe. The worker owns CUDA, the models, the GPU resource manager, one
active RITM predictor state and one active Cutie propagation context. The original in-process mode
stays available as the local backend; both backends run the same engine code.

**Tech stack:** Python 3.12, PyTorch 2.14 (CUDA 13.0), PySide6, SQLite (stdlib `sqlite3`), uv,
pytest. No new runtime dependencies.

**Spec:** the task description pasted on 2026-10-07 ("CutiePlus: Hardware-Adaptive Single-GPU
Annotation Service"), sections 1-20.

## Global constraints

- Work on branch `shared-gpu-service` in the worktree `.worktrees/shared-gpu-service`; `main` and
  its checkout stay untouched (three live annotator sessions run from it on :2, :3, :4).
- Never write to `workspace/` or to the frame directories; tests and benchmarks use copies in temp
  or scratch directories. Frames are read in place, read-only.
- Exactly one GPU (`cuda` device from config or `CUDA_VISIBLE_DEVICES`); no multi-GPU code.
- No GPU-model-specific memory constants; budgets come from runtime queries and measurements.
- Do not change model weights, segmentation algorithms, or quality-sensitive defaults silently;
  every resource-driven change of quality settings is recorded and reported.
- GUI processes in shared mode never initialize CUDA (`CUDA_VISIBLE_DEVICES=''` before torch).
- Worker uses the `spawn` start method. Coordinator never imports CUDA code paths.
- Bounded queues and caches everywhere.
- Do not push, do not deploy to the live sessions.
- While annotators are active, GPU tests cap the worker's memory (configurable cap) and never
  allocate memory pressure on the physical GPU beyond a small ballast.

## Review focus

1. A task edited by the legacy GUI after the coordinator released it: the coordinator must re-import
   it rather than overwrite the newer files from stale SQLite state.
2. A propagation result arriving after the user cancelled, edited an anchor, or lost the lease must
   be rejected and never reach `masks/`.
3. A failed mask save (disk full, permissions) must surface as an error and never hang `join()`.
4. A GUI killed with SIGKILL must release its lease after the heartbeat timeout; the task reopens
   with every confirmed edit.
5. Worker death mid-propagation: replay must not rewrite committed frames, and must continue from
   the last confirmed frame.

---

## Phase 1: review, baseline tests, save-worker fix

**Files:** create `tests/conftest.py`, `tests/helpers.py` (synthetic RGB-T task factory),
`tests/test_task_state.py`, `tests/test_resource_manager.py`, `tests/test_edit_history.py`,
`tests/test_baseline_controller.py` (offscreen Qt, CPU, real weights: click, accept anchor,
propagate, undo/redo, remove anchor, empty-anchor stop, binary export), `docs/ARCHITECTURE.md`
(existing behaviour review); modify `gui/resource_manager.py` (save thread).

- [ ] Write baseline tests against the unchanged code; record golden outputs (masks, anchors) of the
  original click and propagation paths on CPU in `tests/data/baseline_*.npz` via
  `tests/make_baseline.py`.
- [ ] Failing test: a save that raises leaves `save_queue.join()` hanging. Fix: `task_done()` in a
  `finally`, atomic write (temp file + `os.replace`), errors kept and raised by
  `ResourceManager.check_saves()`, the controller reports them.
- [ ] Document the existing architecture and inference behaviour in `docs/ARCHITECTURE.md`.

## Phase 2: engine without Qt, local backend

**Files:** create `cutieplus/__init__.py`, `cutieplus/engine/{models,ritm,propagation,policies,images}.py`,
`cutieplus/backends/{base,local}.py`; modify `gui/main_controller.py`, `gui/click_controller.py`
(model-free session factory), `gui/interactive_utils.py` (no CUDA at import), `gui/task_browser.py`
(load models through the backend).

Interfaces produced:
- `engine.models.load_models(cfg, device) -> LoadedModels(cutie, ritm_net, identity)`.
- `engine.ritm.RitmEngine(net, device, amp)`: `open(record, image) / click(id, x, y, neg) ->
  ClickResult / undo(id) / close(id) / replay(record, image)`; one active GPU state; inactive
  state snapshots on CPU, bounded.
- `engine.propagation.PropagationPlan` (immutable snapshot) and `PropagationRuntime.step() ->
  StepResult`, phases `anchors -> seed -> frames -> finished`.
- `engine.policies`: `median_target_area`, `choose_resolution`, `memory_tokens_per_frame`,
  `select_permanent_anchors`.
- `backends.base.TaskBackend` protocol with `open_interaction / add_click / undo_click /
  close_interaction / start_propagation / poll_propagation / cancel_propagation /
  release_propagation / commit_frame / remove_anchor / restore_frame / set_intervals /
  set_description / submit_for_review / reset_all_masks / close`.
- `backends.local.LocalTaskBackend` reproduces the original behaviour exactly.

- [ ] Equivalence tests: RITM engine vs. original `ClickController`/`ClickInteraction`; runtime vs.
  original `rebuild_permanent_memory` + loop; golden outputs from Phase 1 still match.

## Phase 3: GPU worker, hardware detection, profiling, budgets

**Files:** create `cutieplus/gpu/{hardware,profiler,budget}.py`, `cutieplus/worker/{main,commands}.py`,
`cutieplus/engine/fake.py`; tests `tests/test_gpu_budget.py` (simulated devices),
`tests/gpu/test_worker_gpu.py`.

- [ ] `HardwareInfo` from PyTorch (name, index, UUID, capability, total/free, allocated/reserved,
  precisions). Clear error without CUDA; CPU only with `allow_cpu` (tests).
- [ ] Profiler: measured peaks for RITM click, Cutie step at two token counts and two resolutions,
  permanent-memory insert, RITM with an active Cutie context; linear model in query tokens and
  memory-token x query-token products.
- [ ] `MemoryBudget`: `available = free_device + (reserved - allocated)`, minus headroom (fraction
  and floor, configurable), minus interactive reserve; optional cap. Admission, anchor selection
  under budget (nearest to start, recorded), queue/reject, OOM handling with bounded retries.

## Phase 4: coordinator and client

**Files:** create `cutieplus/protocol.py`, `cutieplus/coordinator/{server,service,worker_proxy}.py`,
`cutieplus/client.py`, `cutieplus/backends/shared.py`, `cutieplus/cli.py`
(`python -m cutieplus.coordinator start|stop|status`); modify `interactive_demo.py`
(`--backend local|shared`, `--service-socket`), `gui/task_browser.py`.

- [ ] Single instance (flock), stale socket cleanup, worker spawn and restart, request/response with
  timeouts, heartbeats.

## Phase 5: RITM isolation, step propagation, scheduler

**Files:** create `cutieplus/coordinator/scheduler.py`; tests `tests/test_scheduler.py`,
`tests/test_service_scheduling.py` (fake engine, multi-process), `tests/gpu/test_interleaving.py`
(A100/B/A101/A102/C/A103 pattern compared with uninterrupted propagation).

- [ ] Interactive queue per session with round-robin, background jobs with one active context,
  anti-starvation (max interactive burst and max background wait), all configurable.

## Phase 6: sessions, leases, revisions

**Files:** create `cutieplus/coordinator/{sessions,tasks}.py`; tests `tests/test_sessions_leases.py`.

- [ ] Sessions keyed by UUID with VNC display and annotator from `annotators.json`; leases with
  increasing epoch, renewal by heartbeat, expiry; coordinator also holds the legacy flock while a
  lease is active. `input_revision` and `content_revision` per task.

## Phase 7: reliable storage

**Files:** create `cutieplus/storage/{db,objects,views,importer}.py`, `cutieplus/coordinator/commits.py`;
tests `tests/test_storage.py`, `tests/test_commits.py`.

- [ ] Commit protocol of spec 9.1, request IDs, edit history rows, legacy views written after the DB
  commit and before the confirmation, view stats recorded to detect external edits.

## Phase 8: failure recovery

**Files:** create `cutieplus/coordinator/recovery.py`; tests `tests/test_recovery.py`.

- [ ] Worker death: stop attempt, verify revision, restart worker, rebuild from the job snapshot,
  replay without commits to the last confirmed frame, continue. OOM: release, recover in the spec's
  order, bounded retries. Coordinator restart: kill stale worker, clean temp files and orphans,
  resync views, mark interrupted jobs.

## Phase 9: export and review

**Files:** modify `gui/task_browser.py`, `gui/main_controller.py`; extend coordinator with
`submit_for_review`, `begin_export`, `finish_export`, `mark_done`; tests `tests/test_export_review.py`.

- [ ] Exports carry the content revision; verification of exit status, `metadata.json` and an
  unchanged revision; mark done only for a current review and export.

## Phase 10: integration, benchmarks, documentation

**Files:** create `scripts/benchmark_service.py`, `docs/SHARED_SERVICE.md`,
`docs/benchmarks/2026-10-07-<gpu>.md`; update `README.md`, `docs/MODIFICATIONS.md`.

- [ ] Baseline numbers of the original local mode (offscreen Qt, real controller), then scenarios
  A-K on the shared service, on copies of real RGB-T tasks.

## Highest-risk changes

1. Cutie temporal memory across scheduler interleaving (step runtime must keep the whole
   `InferenceCore` untouched between steps; CUDA streams synchronized at step boundaries).
2. RITM state offload and reconstruction (predictor transform states, `prev_prediction`).
3. Memory accounting (no double counting of reserved and free memory; others' usage).
4. Legacy file views vs. SQLite authority during migration.
5. Refactoring `MainController` without changing local behaviour.
