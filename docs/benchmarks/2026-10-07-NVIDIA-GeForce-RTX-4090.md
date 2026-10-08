# Benchmark: shared GPU service vs. the original local mode (2026-10-07)

Measured with `scripts/benchmark_service.py` on copies of real RGB-T tasks (main/1629, main/0240, main/0468, main/1530, main/2363, main/0327); frames read in place. Timings are measured; admission estimates are predictions fitted from device measurements, not measured peaks. Part 1 results are retained from the original run; subsequent runs and commands are recorded in the JSON.

## Hardware and service

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Compute capability | 8.9 |
| Total memory | 24.0 GiB |
| Free memory at the start | 21.8 GiB (other processes: 1989469: 450 MiB, 1989470: 450 MiB, 1989471: 660 MiB) |
| Driver, PyTorch, CUDA | 580.173.02, 2.14.0+cu130, 13.0 |
| Precision (worker) | float16 autocast |
| Worker start (load + profile) | 7.8 s; model load 1.45 s |
| Model memory (both models) | 0.17 GiB |
| Measured memory model | click 187 MiB + 52.4 B/pixel; step 34 MiB + 31.6 KiB/query token + 15.0 B per memory x query token |

### Subsequent benchmark runs

- 2026-10-07T21:00:49: parts 2,3; free GPU memory 21.80 GiB; worker admission cap none; external pressure True; status superseded; base commit `b4075e7ec3e79b67b60462f9f260f0d45b2c1992`.
  Functional checks passed; performance rerun without concurrent CPU tests and with the final harness.
- 2026-10-07T21:12:29: parts 2,3; free GPU memory 21.80 GiB; worker admission cap none; external pressure True; status completed; base commit `b4075e7ec3e79b67b60462f9f260f0d45b2c1992`.

The three existing annotator GUI processes were suspended during the recorded Part 2/3 maintenance runs with the owner's permission; their existing CUDA allocations remained resident. They were resumed afterwards. No active task data was modified.

## One labeling window: original, refactored local, shared

The same window (`MainController`, offscreen) runs the same clicks (6, the first is a warm-up and is left out) and the same forward propagation from the first mask anchor, in the original checkout (its own environment), in this branch in local mode, and in this branch with the shared service (coordinator and GPU worker in their own processes). Click times include the window's drawing; propagation times run from the key press to the end of the run. Shared-mode clicks and propagated masks are committed durably before they return or continue.

| Task | Frames, size, anchors | Size | Click p50/p95 (ms) original | local | shared | Frames | Frames/s original | local | shared | GPU memory of the window process (MiB) original / local / shared | Masks vs. original (local; shared) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main/1629 | 480, 640x512 visual, 4+0 | 480 px | 29/35 | 32/41 | 41/48 | 325 | 64.2 | 64.0 | 53.7 | 2512 / 2456 / 0 | 0 files (0 px); 0 files (0 px) |
| main/0240 | 750, 1920x1080 visual, 27+1 | 480 px | 72/79 | 73/88 | 79/117 | 290 | 21.0 | 23.4 | 23.0 | 12404 / 12336 / 0 | 0 files (0 px); 0 files (0 px) |
| main/0468 | 732, 640x480 visual, 28+2 | 960 px | 30/34 | 30/37 | 39/49 | 187 | 17.1 | 17.7 | 16.1 | 10924 / 10858 / 0 | 0 files (0 px); 0 files (0 px) |
| main/1530 | 304, 1920x1080 visual, 3+0 | 960 px | 75/79 | 72/83 | 78/112 | 299 | 12.9 | 14.3 | 13.9 | 12364 / 13270 / 0 | 0 files (0 px); 0 files (0 px) |
| main/2363 | 66, 960x576 thermal, 2+0 | 480 px | 38/43 | 40/45 | 45/61 | 63 | 41.9 | 49.5 | 43.4 | 2394 / 2340 / 0 | 0 files (0 px); 0 files (0 px) |
| main/0327 | 90, 640x480 visual, 4+3 | 960 px | 31/35 | 31/41 | 37/49 | 50 | 15.8 | 17.2 | 16.1 | 19764 / 19736 / 0 | 0 files (0 px); 0 files (0 px) |

No shared-mode window process initialized CUDA (checked with `torch.cuda.is_initialized()` at the end of each run).

GPU worker after part 1: allocated 0.20 GiB, reserved 0.54 GiB, peak allocated 5.61 GiB, peak reserved 21.01 GiB.

## Scenarios against the service

Client sessions (no window) talk to one coordinator process; latencies are measured at the client, from request to answer. Propagation frames/s uses coordinator time from job creation to completion, including queueing, initialization, commits and recovery; it does not depend on when the client starts polling. Inference per step includes permanent-anchor and seed steps. Click queue time is measured by the dispatcher, separately from GPU execution and durable commits.

### A. One annotator clicks

| | p50 (ms) | p95 (ms) |
|---|---|---|
| Click, request to answer | 30 | 37 |
| of which scheduler queue | 0.1 | 0.1 |
| of which inference in the worker | 19 | 22 |
| of which commit (object, transaction, files) | 7 | 10 |

### B. Two annotators alternate clicks

Every click follows the other annotator's, so the worker rebuilds the interaction (kinds: snapshot).

| | p50 (ms) | p95 (ms) |
|---|---|---|
| alice | 67 | 91 |
| bob | 71 | 88 |
| Reconstruction from a CPU snapshot | 13 | 33 |
| Rebuilding by replaying 15 clicks | 315 | |

Without snapshots (`ritm_snapshot_mib: 0`), switching replays the clicks: 1 clicks 30 ms, 3 clicks 64 ms, 5 clicks 101 ms, 7 clicks 137 ms, 9 clicks 185 ms.

### C, H. One propagation

| Run | Frames | Frames/s | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Execution/admission attempts | End |
|---|---|---|---|---|---|---|---|
| C. One propagation (0240, 1920x1080, 27 anchors) | 292 | 24.0 | 24.8 | 10.7 | 0.00 | 1/1 | finished (empty_anchor) |
| H. Small target, 960 px (1530, 1920x1080) | 301 | 14.1 | 57.3 | 10.6 | 0.00 | 1/1 | finished (end) |

- C. One propagation (0240, 1920x1080, 27 anchors): 27 permanent anchors (quality cap 137, limited by nothing), 1620 query tokens, estimate 2.12 GiB.
- H. Small target, 960 px (1530, 1920x1080): 1 permanent anchors (quality cap 1, limited by quality_cap), 6420 query tokens, estimate 8.72 GiB.
- Cumulative worker peak through C (including warm-up and earlier clicks): allocated 1.73 GiB, reserved 11.92 GiB.

### D. One propagation while another annotator clicks

| Run | Frames | Frames/s | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Execution/admission attempts | End |
|---|---|---|---|---|---|---|---|
| propagation (0240, as C) | 292 | 23.5 | 24.7 | 8.9 | 0.00 | 1/1 | finished (empty_anchor) |

Clicks during the run: p50 59 ms, p95 84 ms (30 clicks); inference p50 23 ms. Masks compared with run C: 0 files differ (0 pixels).

Click scheduler wait p50/p95: 7/22 ms; durable commit p50/p95: 15/20 ms.

### E. One propagation while three annotators click

| Run | Frames | Frames/s | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Execution/admission attempts | End |
|---|---|---|---|---|---|---|---|
| propagation (0468, 960 px) | 189 | 14.4 | 47.8 | 5.4 | 0.00 | 1/1 | finished (empty_anchor) |

| Annotator | Clicks | p50 (ms) | p95 (ms) | Queue p50/p95 (ms) | Inference p50/p95 (ms) | Commit p50/p95 (ms) |
|---|---|---|---|---|---|---|
| carol | 20 | 71 | 99 | 29/59 | 24/33 | 8/11 |
| bob | 20 | 82 | 117 | 22/57 | 24/29 | 14/18 |
| dave | 20 | 88 | 142 | 42/97 | 21/25 | 8/12 |

Compared with a separate uninterrupted 0468 run: 0 files differ (0 pixels).

### F. Three propagations submitted together

| Run | Frames | Frames/s | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Execution/admission attempts | End |
|---|---|---|---|---|---|---|---|
| alice | 324 | 48.3 | 12.0 | 7.1 | 0.00 | 1/1 | finished (end) |
| carol | 63 | 7.8 | 11.8 | 5.9 | 6.71 | 1/1 | finished (end) |
| dave | 49 | 4.4 | 40.3 | 6.7 | 8.06 | 1/1 | finished (empty_anchor) |

The 100 ms sampler saw at most 1 running/committing propagation context at a time.

### G. Opening and closing tasks

| | Count | p50 (ms) | p95 (ms) |
|---|---|---|---|
| First open (imports the task) | 4 | 37 | 85 |
| Open again | 12 | 7 | 68 |
| Close | 16 | 2 | 3 |

### I. GPU memory limits

- I1, worker capped at 2.2 GiB (`gpu.max_worker_gib`), task 0240: 17 permanent anchors (quality cap 137, limited by memory), 292 frames at 26.4 frames/s, state finished. Messages: GPU memory allows 17 of 27 mask anchors in the permanent memory (those nearest to the start frame); the others guide the run when it reaches them.; Permanent memory rebuilt from 17 of 27 mask anchors.
  Compared with C using all 27 anchors: 292 propagated masks differ (22618 pixels). This is a recorded quality-sensitive memory adaptation; human anchors remain intact.
- I2, another process holds 20.2 GiB of the GPU, with release requested on the first poll after 8 s; task 1530 at 960 px: states seen committing, finished, running, waiting_memory, running after 10.0 s, queue wait 10.0 s, then 301 frames at 9.6 frames/s, state finished.
  Immutable propagated outputs vs. H: 0 files differ (0 pixels). The 6 admission checks created 1 execution context(s); no OOM retry occurred.

### J. The GPU worker is killed during a propagation

This scenario was repeated alone after fixing the timer: an in-flight commit from before the crash must not count as recovery. The timer now waits for a new execution attempt and then an increase in its committed-frame count. The same immutable anchor inputs and uninterrupted C baseline were used.

Killed (SIGKILL) after 60 committed frames of 0240. Worker replaced in 7.8 s (new generation 2, including model load and memory profile); new frames committed again after 10.4 s. The job finished in 2 attempts, replaying 61 confirmed frames without writing them (0 differed from what was committed). Masks compared with the undisturbed run C: 0 files differ (0 pixels).

### K. Writing annotations fails

The mask object store was made read-only. A click answered with an error after 34 ms (`internal`); a propagation stopped as failed after 0 frames (saving 000151 failed: [Errno 13] Permission denied: '/tmp/cutieplus-parts23-final-20261007/bench/part2/state/objects/3b/.3beebcaa2b109ace7b700afe7c0548617796500a3a835944fd340e77bfc4c4d6.png.d6aed3ae9cdc41f8b2b2542d0e9619da.tmp'). After the store was writable again, the same click request was applied once (1 accepted click).

## Worker memory and effective configuration

| Worker lifetime | Peak allocated (GiB) | Peak reserved (GiB) | Final allocated (GiB) | Final reserved (GiB) |
|---|---|---|---|---|
| Part 2 final worker generation | 1.67 | 11.60 | 0.19 | 0.43 |
| Part 3 2.2 GiB admission cap | 1.61 | 8.13 | 0.18 | 0.27 |
| Part 3 external pressure | 5.45 | 21.22 | 0.19 | 1.14 |

Peaks reset when the worker restarts; these are cumulative allocator peaks within each worker lifetime, not per-scenario peaks. Initial and restarted worker hardware and profiling samples, every job's precision/model configuration, resolution, temporal-memory settings and effective admission plans are retained in the JSON. The I1 admission cap applies after profiling and is not a hard allocator limit.

## Verification and limits

- CPU full suite: 89 passed, 14 GPU tests deselected (173.96 s). The subsequent benchmark-only regression run passed all 8 cases, including the additional recovery-timer case (90 distinct CPU tests verified).
- GPU integration suite: 14 passed, 81 CPU tests deselected (40.39 s); actual CUDA inference on the recorded RTX 4090.
- Final Part 2/3 run completed with CPU test suites stopped. Scenario J was then repeated alone with the corrected recovery timer; all scenario assertions passed.
- D and E match their uninterrupted propagation baselines; J and I2 match C and H respectively, with zero differing pixels.
- RITM replay matches committed masks and does not change task revisions; a failed click leaves revisions unchanged, and retry plus duplicate request commits exactly once.
- SQLite integrity checks passed for all 4 final service databases; 2254 current immutable object references matched SHA-256 and were read-only. No duplicate (job, frame) propagation commits were found.
- All 230 anchor references across the final job snapshots retain their mask object and anchor kind, including empty anchors.
- I1 deliberately selects fewer permanent anchors and changes propagated masks; its measured difference is disclosed above. Human anchors are preserved.
- Physical external pressure tested admission and resume, not a deliberately induced mid-step CUDA OOM. Deterministic unit/integration tests cover injected OOM and context loss.

These are single runs on one physical GPU, with finite scripted click samples. They do not establish a universal worst-case latency or performance improvement. Propagation jobs run FIFO with one active context; propagation-job time slicing is not implemented. Export compatibility is covered by automated tests; these benchmark tasks use an isolated stub export command, not the live external export pipeline.

