# Benchmark: shared GPU service vs. the original local mode (2026-10-07)

Measured with `scripts/benchmark_service.py` on copies of real RGB-T tasks (main/1629, main/0240, main/0468, main/1530, main/2363, main/0327); frames read in place. All numbers below were measured in this run; nothing is estimated.

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

Client sessions (no window) talk to one coordinator process; latencies are measured at the client, from request to answer.

### A. One annotator clicks

| | p50 (ms) | p95 (ms) |
|---|---|---|
| Click, request to answer | 29 | 37 |
| of which waiting for the GPU | 0 | 0 |
| of which inference in the worker | 19 | 22 |
| of which commit (object, transaction, files) | 8 | 10 |

### B. Two annotators alternate clicks

Every click follows the other annotator's, so the worker rebuilds the interaction (kinds: snapshot).

| | p50 (ms) | p95 (ms) |
|---|---|---|
| alice | 62 | 97 |
| bob | 66 | 83 |
| Reconstruction from a CPU snapshot | 13 | 32 |
| Rebuilding by replaying 15 clicks | 297 | |

Without snapshots (`ritm_snapshot_mib: 0`), switching replays the clicks: 1 click 34 ms, 3 clicks 63 ms, 5 clicks 113 ms, 7 clicks 155 ms, 9 clicks 199 ms.

### C, H. One propagation

| Run | Frames | Frames/s | Run (s) | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Attempts | End |
|---|---|---|---|---|---|---|---|---|
| C. One propagation (0240, 1920x1080, 27 anchors) | 292 | 24.4 | 12.0 | 24.4 | 10.7 | 0.00 | 1 | finished (empty_anchor) |
| H. Small target, 960 px (1530, 1920x1080) | 301 | 14.1 | 21.4 | 56.9 | 10.7 | 0.00 | 1 | finished (end) |

- C. One propagation (0240, 1920x1080, 27 anchors): 27 permanent anchors (quality cap 137, not limited), 1620 query tokens, estimate 2.12 GiB.
- H. Small target, 960 px (1530, 1920x1080): 1 permanent anchor (quality cap 1, limited by quality_cap), 6420 query tokens, estimate 8.72 GiB.
- Worker peak during C: allocated 1.73 GiB, reserved 11.92 GiB.

### D. One propagation while another annotator clicks

| Run | Frames | Frames/s | Run (s) | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Attempts | End |
|---|---|---|---|---|---|---|---|---|
| propagation (0240, as C) | 292 | 23.7 | 12.3 | 24.5 | 9.0 | 0.00 | 1 | finished (empty_anchor) |

Clicks during the run: p50 62 ms, p95 97 ms (30 clicks); waiting for the GPU p50 12.8 ms, p95 26.1 ms; inference p50 23 ms. Masks compared with run C: 0 files differ (0 pixels).

### E. One propagation while three annotators click

| Run | Frames | Frames/s | Run (s) | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Attempts | End |
|---|---|---|---|---|---|---|---|---|
| propagation (0468, 960 px) | 189 | 14.2 | 13.3 | 47.9 | 7.6 | 0.00 | 1 | finished (empty_anchor) |

| Annotator | Clicks | p50 (ms) | p95 (ms) | Waiting for the GPU p50/p95 (ms) |
|---|---|---|---|---|
| carol | 20 | 67 | 104 | 24.8/68.7 |
| bob | 20 | 85 | 117 | 24.2/44.8 |
| dave | 20 | 82 | 133 | 42.0/86.4 |

### F. Three propagations submitted together

| Run | Frames | Frames/s | Run (s) | Inference per step (ms) | Commit per frame (ms) | Queue wait (s) | Attempts | End |
|---|---|---|---|---|---|---|---|---|
| alice | 324 | 54.5 | 5.9 | 11.0 | 5.8 | 0.00 | 1 | finished (end) |
| carol | 63 | 43.6 | 1.4 | 12.4 | 6.6 | 5.95 | 1 | finished (end) |
| dave | 49 | 16.3 | 3.0 | 40.2 | 7.0 | 7.40 | 1 | finished (empty_anchor) |

At most 1 propagation context ran at a time.

### G. Opening and closing tasks

| | Count | p50 (ms) | p95 (ms) |
|---|---|---|---|
| First open (imports the task) | 4 | 36 | 82 |
| Open again | 12 | 6 | 50 |
| Close | 16 | 1 | 2 |

### I. GPU memory limits

- I1, worker capped at 2.2 GiB (`gpu.max_worker_gib`), task 0240: 17 permanent anchors (quality cap 137, limited by memory), 292 frames at 26.2 frames/s, state finished. Messages: GPU memory allows 17 of 27 mask anchors in the permanent memory (those nearest to the start frame); the others guide the run when it reaches them. Permanent memory rebuilt from 17 of 27 mask anchors.
- I2, another process holds 20.2 GiB of the GPU, released after 8 s; task 1530 at 960 px: states in order queued, waiting_memory, running, committing, finished, running after 10.0 s, queue wait 10.0 s, then 301 frames at 14.1 frames/s, state finished.

### J. The GPU worker is killed during a propagation

Killed (SIGKILL) after 60 committed frames of 0240 (61 by the time the new attempt started: a frame computed before the kill was committed after it). Worker replaced in 5.1 s (new generation 2, including model load and memory profile); new frames committed again after 7.7 s. The job finished in 2 attempts, replaying 61 confirmed frames without writing them (0 differed from what was committed). Masks compared with the undisturbed run C: 0 files differ (0 pixels).

### K. Writing annotations fails

The mask object store was made read-only. A click answered with an error after 49 ms (`internal`); a propagation from a changed anchor (masks the store does not hold yet) ended failed after 0 committed frames (saving 000152 failed: [Errno 13] Permission denied). After the store was writable again, the same click request was applied once (1 accepted click).

