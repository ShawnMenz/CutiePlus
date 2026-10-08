# The shared GPU service

CutiePlus can run in two modes:

- **Local** (the original mode, still the default): every labeling window is a process that loads
  Cutie and RITM onto the GPU and writes its task's files itself. Nothing changes for annotators
  who keep using it.
- **Shared**: one coordinator process per task root and one GPU worker hold the only copies of the
  models; every annotator's window sends clicks and propagation requests to them, and the
  coordinator is the only writer of official annotation data.

[ARCHITECTURE.md](ARCHITECTURE.md) describes the design before the service (the local mode);
[benchmarks](benchmarks/) holds the measurements; this document describes the shared mode.

## Processes

```
Linux host, one account
├── TigerVNC :2 ── interactive_demo.py --tasks <root> --backend shared   (Qt, CPU only)
├── TigerVNC :3 ── interactive_demo.py --tasks <root> --backend shared
├── ...                         │ Unix domain socket, one connection + one for heartbeats
│                               ▼
├── python -m cutieplus.coordinator start --tasks <root>                 (CPU only)
│     sessions · task leases · scheduler · commits · SQLite · mask objects · legacy files
│                               │ multiprocessing pipe (spawn), one command at a time
│                               ▼
└── GPU worker                                                           (the only CUDA user)
      GPU resource manager · Cutie · RITM · one click interaction and one propagation on the GPU
```

| Code | Role |
|---|---|
| `cutieplus/engine/` | Inference without Qt, used by both modes: model loading (`models.py`), RITM interactions with isolated state (`ritm.py`), step-based Cutie propagation (`propagation.py`), the small-target and permanent-anchor policies (`policies.py`) |
| `cutieplus/gpu/` | Device detection (`hardware.py`), memory profiling (`profiler.py`), memory budgets and admission (`budget.py`) |
| `cutieplus/worker/` | The GPU worker process (`main.py`, `engine.py`); `fake.py` is a CPU stand-in for tests |
| `cutieplus/coordinator/` | The coordinator: service and API (`service.py`), socket server, sessions, task leases and commits (`tasks.py`), scheduler, propagation jobs (`jobs.py`), click interactions, worker supervision |
| `cutieplus/storage/` | SQLite (`db.py`), immutable mask objects (`objects.py`), the legacy task files as views (`views.py`) |
| `cutieplus/backends/` | What the window talks to: `local.py` (the original behaviour) and `shared.py` (the service) |
| `cutieplus/client.py`, `cutieplus/protocol.py` | The window's connection and the socket message format |

### Model ownership and CUDA

- The worker loads one Cutie and one RITM model at start and keeps them. Weights are read only.
- The coordinator never imports torch. The window, in shared mode, starts with
  `CUDA_VISIBLE_DEVICES=''` and draws on the CPU; `gui/interactive_utils.py` no longer creates
  CUDA tensors at import. A test runs a whole shared-mode session in a separate process and checks
  that it never initialized CUDA.
- The worker is started with the `spawn` method and uses exactly one device: `service.device`
  (`cuda` is the first device `CUDA_VISIBLE_DEVICES` exposes). There is no multi-GPU code.
- Mutable model state is never shared: each click interaction has its own RITM predictor state
  (the network is shared), and each propagation attempt builds its own Cutie `InferenceCore`.

## GPU memory

### What is detected

At every start the worker reads, through PyTorch: the device name, index and UUID, compute
capability, multiprocessor count, total and free memory, PyTorch's allocated and reserved memory,
and the supported precisions (float16 from compute capability 5.3, bfloat16 as PyTorch reports it,
TF32 from 8.0). Without a CUDA device it stops with a clear message (`NoGpuError`); a CPU backend
exists only with `service.allow_cpu` (development and tests).

### What is measured

The profiler (`cutieplus/gpu/profiler.py`) runs warm-up work on synthetic frames of
`service.profile_frame_size` (memory does not depend on image content) and measures peaks with
PyTorch's allocator statistics:

- a RITM click (first click, then a zoomed-in second click) at two frame sizes;
- Cutie at the default processing size with one and with four permanent anchors, and at the
  small-target size: the state resident between steps, the peak of a permanent-anchor step, and
  the peak of a propagation step;
- a click while a propagation holds its memory.

From these it fits linear models (`MemoryModel`):

- click peak = a + b · pixels;
- resident state of a run = c + d · Q + e · N;
- transient peak of a step = f + g · Q + h · Q · N;

where Q is the query tokens of a frame (features at stride 16 at the processing size) and N the
memory tokens a step reads (permanent anchors, working memory, long-term memory). The intercepts
are then raised until no measurement exceeds the model. Model parameter sizes are not used to
estimate inference memory. The model is measured on the device in use at every start; nothing
measured on one GPU is assumed for another. On the RTX 4090 of the lab, h came out at about
20 bytes per memory × query token pair, which is what the original constant implied (see below).

### The budget

For a decision the worker takes a fresh snapshot of the device (`MemorySnapshot`):

```
available = free on the device (all processes) + (reserved - allocated in the worker)
budget    = available - headroom          headroom = max(headroom_fraction x total, headroom_min)
          (and at most max_worker_gib - allocated, when a cap is set)
```

Reserved memory is not counted twice: the device's free memory already excludes it, and only
the worker's own cached-but-unused blocks are added back. Memory held by other processes is seen
through the device's free memory, read again at every decision. Every estimate is multiplied by
`estimate_margin` (1.25). While a propagation holds memory, room for one click on the largest
frame is kept (`reserve_for_clicks`). Queued jobs hold no GPU memory.

### Admission of a propagation

When a job reaches the front of the queue, the worker plans it (`GPUResourceManager.plan_propagation`):

1. The quality cap: at most `max_affinity_elements` memory × query tokens per step (4e8, the
   value of the original release) bounds the number of permanent anchors, exactly as before.
2. The memory budget: the most anchors (up to the cap) whose estimated resident state plus step
   peak fits the budget.
3. Decision: **run** with that many anchors, **queue** (no context is created; the job asks again
   every 2 s), or **reject** when the job could not fit even on an empty GPU, or when it has waited
   `queue_wait_seconds`.

The anchors used are those nearest to the start frame, in time order (the original selection
policy). An anchor left out of the permanent memory is never removed from the task: propagation
feeds it as an ordinary memory frame when it reaches it. When memory, not the quality cap, limits
the anchors, the job's messages say so and the job records the selected anchors and the plan (in
`jobs.effective`, one entry per attempt).

How this adapts: on a larger GPU (or with more free memory) the budget allows every anchor up to
the quality cap, which on the lab GPU reproduces the original behaviour exactly; on a smaller GPU,
or when other processes hold memory, fewer anchors are used, then jobs wait, then they are
rejected. Clicks are admitted the same way against their own estimate.

### When memory runs short

Before each step of a running job the worker checks that the step still fits (other processes may
have grown). If not, it works through the recovery order of the specification as far as it can:

1. release the allocator's cached blocks;
2. move the active click interaction's state to CPU memory;
3. read fewer frames ahead;
4. the step waits (up to 20 times, 0.5 s apart), keeping the job's context;
5. the context is freed and the job is queued again; its next attempt may use fewer anchors
   (recorded), and it replays its confirmed frames without writing them;
6. it is rejected after `queue_wait_seconds`.

A CUDA out-of-memory error during a step discards the context (a half-updated temporal state is
never continued); the job is retried at most `max_oom_retries` times, and only when the plan or the
free memory has changed since the failure; otherwise it waits. An out-of-memory error during a
click drops that interaction's GPU state; its accepted clicks are replayed at the next click.
`torch.cuda.empty_cache()` is not used to free live tensors, only cached blocks.

### Quality-sensitive settings

| Setting | Behaviour |
|---|---|
| Processing size (480 px, 960 px for small targets) | Decided at task open by the original policy; never changed by the resource manager |
| Mixed precision (`amp`) | Kept; a GPU without float16 stops the worker unless `allow_precision_change`, which runs in float32 and records the change (`quality_changes` in the worker's status) |
| Memory sizes (`mem_every`, working and long-term memory) | Those the window's controls send; never changed |
| Permanent anchors | Quality cap as before; fewer only under memory limits, recorded, nearest to the start |

## Scheduling

Requests to the worker go through one scheduler (`cutieplus/coordinator/scheduler.py`):

- **Interactive** requests (clicks, click undo, rebuilding an interaction) come first, round-robin
  between sessions.
- **Background** work is propagation, one step at a time: each permanent anchor put into memory,
  the start frame, and each frame are separate steps. After `max_interactive_burst` interactive
  requests in a row (4), or once a ready step has waited `max_background_wait` (0.25 s), the step
  goes first, so continuous clicking cannot stall a propagation.
- **One propagation context** lives on the GPU at a time. Other jobs wait in order; a job waiting
  for memory gives up its turn without holding anything.
- A propagated mask is committed before the job's next step (a job waiting for its commit is not
  scheduled; clicks run meanwhile).

Between two steps the job's `InferenceCore` is untouched: Cutie's temporal memory is kept while
other annotators click. Tests compare a propagation with clicks between its steps against an
uninterrupted one (identical masks, on CPU and GPU), and the scheduler reproduces the pattern
A100, B, A101, A102, C, A103.

Click interactions: the coordinator keeps each interaction's record (task, frame, modality,
target, the committed mask it started from, the accepted clicks in order); the worker keeps its
state, only the active one on the GPU. Switching to another interaction keeps the previous one as a
CPU snapshot taken with RITM's own state functions (those its undo uses) and restores the next one
from its snapshot, or, when there is none (CPU budget `ritm_snapshot_mib` exceeded, or a restarted
worker), by replaying its accepted clicks in order. Replays write nothing; tests check that both
ways give exactly the original results.

## Sessions, tasks and versions

- A **session** per window: a UUID, the annotator (from `--labeler`, the VNC display's port in
  `annotators.json`, `$CUTIE_LABELER`, or the account, as before), the display, the process id,
  created and last-seen times, and a status. The window sends a heartbeat every 5 s from a thread
  of its own; a session silent for `session_timeout_seconds` (30) expires: its leases, jobs and
  interactions end.
- A **lease** per open task: one session edits a task at a time. Every lease has a new, larger
  epoch; every change names the lease epoch it was made under, and is refused when that lease is
  gone (`lease_lost`).
- The original **file lock** (`lock.json`, flock) is taken by the coordinator for each lease, with
  the session's annotator, display and process id, so the browser's *Open by* column keeps working
  and the original window cannot open a task the service holds (or the other way round).
- **Revisions** per task: `input_revision` changes with human inputs of inference (masks, anchors,
  reset); `content_revision` changes with every committed change, including each propagated mask.
  Propagation does not change `input_revision`. A propagated mask is refused when the task's
  `input_revision`, the lease or the job's attempt changed since the job started, or when the frame
  has become an anchor.

## Storage

- **SQLite** (`<state dir>/service.db`, write-ahead log, full synchronization) holds tasks,
  frames (mask object and anchor of each frame), sessions, leases, the edit history, request ids,
  jobs with their input snapshots, exports, worker starts and events. The coordinator refuses a
  state directory on a network filesystem.
- **Mask objects** (`<state dir>/objects/`) are PNG files named by the SHA-256 of their bytes,
  written to a unique temporary file, synced, renamed, and made read-only.
- **The legacy files** (`masks/<frame>.png`, `anchors.json`, `prefill.json`, `progress.json`,
  `interval.json`, `description.json`, and `binary_masks/` at submission) are written by the
  coordinator after each commit, in the same formats, for the export pipeline and the task
  browser. Their size and modification time are recorded.

The commit protocol of every change: encode and publish the mask object; in one short transaction,
replay a known request id, check the lease and revisions, update the mask reference, the anchor,
the edit history and the revisions; after the commit, write the legacy files; then answer. No
transaction is held during inference. Every change carries a request id: a request repeated after a
lost connection or a worker restart returns the first answer and is applied once.

Migration: the first time the service opens a task it imports its files. When it opens a task
again it compares the files with what it last wrote; a task changed by the original window in the
meantime is imported again (the files win, the change is logged). A commit whose files were not
written (the service stopped in between) is completed at the next start or open.

## Failures

| Failure | What happens |
|---|---|
| GPU worker crashes or hangs (no answer within `request_timeout_seconds`) | The coordinator kills it if needed, starts a new one (new generation; results of the old one are never used), and every job that had a context starts a new attempt from its input snapshot, with the same permanent anchors, replaying its confirmed frames without writing them (and comparing them with what was committed), then continues. Jobs whose task changed in the meantime stop. A click in flight fails once and is retried by the window. |
| Out of GPU memory | See *When memory runs short*. |
| A commit fails (disk full, permissions) | Nothing is half written (the transaction rolls back; files follow the database). The click or edit reports the error; a propagation stops. The worker's interaction state is rebuilt from the record. |
| A legacy file cannot be written after its commit | The commit stands and the answer says so. The task's files are marked behind (later commits do not mark them current); the coordinator retries every 5 s while it holds the task, and an export, a submission for review or the next open rewrites them first. |
| A click request times out while the worker runs it | The interaction is rebuilt from its accepted clicks at the next click, so a click the coordinator never accepted cannot linger in the worker. |
| An unexpected error during a propagation step | The job's context is freed and the job fails with the error (it does not stay scheduled). |
| The coordinator stops or crashes | At start: a second coordinator for the same root is refused (lock in the state directory); a worker left behind is stopped; temporary files and unreferenced objects are removed; interrupted jobs are marked; leases are marked interrupted; sessions get a full timeout to reconnect; interrupted file updates are completed. Windows reconnect and take their task again when it has not changed. |
| A window crashes or loses its connection | Its session expires; its lease, file lock, jobs and interaction end. Everything it committed stays. Reopening the task reads the committed state. |

## Export and review

The workflow is unchanged: not started, in progress, under review (*Save binary PNGs*), done
(*Mark done* in the browser). In shared mode:

- *Save binary PNGs* asks the coordinator, which writes `binary_masks/`, `interval.txt` and
  `description.txt` from the committed masks and records the content revision under review.
- An export (browser *Export*, *Mark done*, or the window's *Export task masks*) still runs the
  command of `task_set.json` from the window's process. The coordinator records it with the
  content revision it exports before it starts, and accepts it only when the command succeeded,
  wrote a `metadata.json` newer than the start, and the task did not change meanwhile (otherwise
  `failed` or `stale`).
- *Mark done* needs the task under review, a review of the current content, and a complete export
  of the current content; the browser exports first when the export is missing or stale.
- Any change puts the task back in progress and makes the review and the export obsolete.

## Running it

Start the coordinator in a terminal of the shared account (a `tmux` session survives logouts):

```bash
uv run python -m cutieplus.coordinator start --tasks workspace/cutie_tasks
```

It loads and profiles the models (about 8 s on the lab GPU) and prints the socket when ready. Its
log is in `<task root>/.cutieplus/logs/coordinator.log`, the worker's in `worker.log`.

Each annotator, in their own TigerVNC desktop, starts the browser in shared mode:

```bash
uv run python interactive_demo.py --tasks workspace/cutie_tasks --backend shared
```

The annotator is found from the VNC display as before. Without `--backend shared` (or
`CUTIEPLUS_BACKEND=shared`) the window runs in the original local mode; both modes respect each
other's locks, so they can be mixed while migrating.

```bash
uv run python -m cutieplus.coordinator status --tasks workspace/cutie_tasks
```

```bash
uv run python -m cutieplus.coordinator stop --tasks workspace/cutie_tasks
```

Settings are in `cutie/config/gui_config.yaml`, section `service`; `--set` overrides one at start,
e.g. `--set gpu.max_worker_gib=10` to cap the worker's memory.

### Diagnosing the GPU worker

- `status` shows the worker's pid, generation (it grows at each restart), hardware, measured
  memory model, quality-relevant changes, the scheduler's queues, sessions, leases, jobs and
  latency statistics.
- `<state dir>/logs/worker.log` has the worker's own log, including Python tracebacks.
- The database keeps every worker start (`worker_starts`: hardware, memory profile, model
  identity) and every failure and recovery (`events`):
  `sqlite3 workspace/cutie_tasks/.cutieplus/service.db "select at, kind, detail from events order by event_id desc limit 20"`.
- A worker that fails to start prints why (no CUDA device, out of memory while loading, missing
  weights); the coordinator then exits.

## Limitations

- **One shared Linux account.** Session ids and leases coordinate trusted collaborators and
  prevent accidental conflicts; they are not authentication. Anyone with the account can connect
  to the socket, act as any annotator, or edit the files directly.
- One GPU. Propagation jobs run one at a time in order of submission; a long job delays the
  next one (they are not time-sliced). The `long` split has sequences of thousands of frames
  (up to 26,490): at 15-25 frames/s such a run holds the GPU context for minutes, while the
  others wait. Clicks are not affected (they run between its steps).
- Nothing is pruned: every committed mask version stays as an object and in the edit history
  (about 5 KB per mask and 1 KB of database per edit in the benchmarks).
- After three failed restarts of the GPU worker the coordinator stops trying; windows get errors
  until the coordinator is restarted (`status` shows the problem).
- In shared mode a propagation starts from the committed mask of the start frame, while the
  original window used the soft probabilities of its last click on that frame. On the test
  scenario this changes 1 pixel in 2 of 8 propagated frames; everything else matches the original
  exactly (see Verification).
- The export command still runs in the window's (or browser's) process; the coordinator verifies
  it afterwards but does not run it.
- The permanent-memory and memory-gauge controls of the window have no effect in shared mode
  (each propagation builds its memory from the anchors); *Clear memory* does nothing there.
- The memory model is linear and fitted on synthetic frames of one size; estimates carry a 1.25
  margin, and the pressure check and out-of-memory handling cover the rest.

## Verification

```bash
uv run pytest
```

```bash
CUTIEPLUS_TEST_GPU=1 uv run pytest -m gpu
```

The CPU run (no CUDA visible) covers the engine against the original code with the real weights,
the storage and commit rules, the scheduler, and the coordinator end to end with a fake GPU worker
(crashes, out-of-memory errors, restarts, expiry, idempotency, exports). The GPU run compares the
worker with the original code on the GPU and runs a whole labeling session in shared mode against
the local mode. `tests/data/baseline_local_session.*` is the recording of the original window
(`tests/make_baseline.py`).

Benchmarks: [benchmarks/](benchmarks/) (`scripts/benchmark_service.py`).

### October 2026 benchmark completion

The [measured report](benchmarks/2026-10-07-NVIDIA-GeForce-RTX-4090.md) and its
[raw JSON](benchmarks/2026-10-07-NVIDIA-GeForce-RTX-4090.json) include the original Part 1
comparison and completed Parts 2/3: scenarios A-K, CPU-snapshot and click-replay reconstruction,
worker death, storage failure, a worker admission cap, and real memory held by another process.
Part 2 uses socket clients rather than additional visible Qt windows; Part 1 and the separate
GPU integration tests exercise the Qt path.

The default, unchanged-quality propagation runs with one and three concurrent clickers match
their uninterrupted baselines pixel for pixel. Recovered propagation after worker death and
after waiting for external GPU memory also matches. Under the explicit 2.2 GiB admission cap,
0240 uses 17 of its 27 permanent anchors: 292 propagated masks differ from the full-anchor run
(22,618 pixels total). All original human anchors remain intact. This is a measured consequence
of the documented anchor-selection policy, not a claim of unchanged quality under memory limits.

`gpu.max_worker_gib` limits estimated allocations used for admission. It is not a hard limit on
PyTorch's reserved memory, and startup profiling runs before this policy is applied. The report
distinguishes admission estimates, allocated memory, reserved memory and cumulative worker peaks.

Reproduce Parts 2/3 from this worktree using the existing environment and a **fresh** scratch
directory separate from the source tasks:

```bash
uv run --no-sync python scripts/benchmark_service.py \
  --source /path/to/cutie_tasks --scratch /tmp/cutieplus-benchmark-new \
  --original-repo /path/to/original/checkout --parts 2,3 --output docs/benchmarks
uv run --no-sync python scripts/bench/report.py \
  docs/benchmarks/<date>-<gpu>.json docs/benchmarks/<date>-<gpu>.md
```

The original checkout is used only by Part 1. Parts 2/3 copy annotation inputs into scratch;
original images are read in place. Existing Part 1 results are preserved when the JSON is updated.
Full worker profiles (including every calibration sample), job configurations and admission plans
are retained in the JSON; timings include scheduler queueing separately from inference and commits.
Queued-job throughput uses coordinator creation/completion times, even if the client polls late.
Recovery latency waits for commits from the reconstructed execution attempt; an already-computed
frame that finishes saving just after the worker dies does not count as recovered inference.

The command above records I2 as **not run**. Add `--external-pressure` only during an explicitly
approved maintenance window: it temporarily allocates most free device memory, which can make
other GPU applications fail. It does not stop or restart annotator windows itself. In the recorded
run the owner authorized suspending the three original GUI processes; they were resumed with
their existing windows afterwards. Use `--worker-cap-gib <GiB>` to record a different Part 2
admission limit. Every changed resource configuration must be considered when comparing results.

On hosts whose shell exports unrelated ROS Python packages, isolate pytest from those plugins:

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUTIEPLUS_TEST_GPU=1 \
  uv run --no-sync pytest -m gpu
```

Physical memory-pressure benchmarking tests admission and resume after release; deterministic
tests separately cover OOM errors and a lost context. The benchmark does not establish a
worst-case latency bound, physical mid-step OOM recovery, or throughput on other GPU models.
