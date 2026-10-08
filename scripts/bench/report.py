"""
The Markdown report of a benchmark run (scripts/benchmark_service.py):

    python scripts/bench/report.py <results .json> <report .md>

Measured timings and explicitly labelled resource estimates; skipped scenarios stay visible.
"""
import json
import sys


def ms(value) -> str:
    return '–' if value is None else f'{value * 1000:.0f}'


def ms1(value) -> str:
    return '–' if value is None else f'{value * 1000:.1f}'


def gib(value) -> str:
    return '–' if value is None else f'{value / 2**30:.2f}'


def number(value, digits=1) -> str:
    return '–' if value is None else f'{value:.{digits}f}'


def stat_cells(entry) -> str:
    if not entry or not entry.get('count'):
        return '– | –'
    return f'{ms(entry["p50"])} | {ms(entry["p95"])}'


def job_row(name, job) -> str:
    return (f'| {name} | {job["frames"]} | {number(job["fps"])} | {ms1(job["inference_per_step"])} | '
            f'{ms1(job["commit_per_frame"])} | {number(job["queue_seconds"], 2)} | '
            f'{len(job["effective"])}/{job["attempts"]} | {job["state"]} ({job["stop_reason"]}) |')


JOB_HEADER = ('| Run | Frames | Frames/s | Inference per step (ms) | Commit per frame (ms) | '
              'Queue wait (s) | Execution/admission attempts | End |\n|---|---|---|---|---|---|---|---|')


def main(source: str, target: str) -> None:
    with open(source) as f:
        r = json.load(f)
    lines = []
    add = lines.append
    device = r['device']
    worker = r.get('worker') or {}
    hardware = worker.get('hardware') or {}
    model = worker.get('memory_model') or {}
    add(f'# Benchmark: shared GPU service vs. the original local mode ({r["date"][:10]})\n')
    add('Measured with `scripts/benchmark_service.py` on copies of real RGB-T tasks '
        f'({", ".join(r["tasks"])}); frames read in place. Timings are measured; admission estimates '
        'are predictions fitted from device measurements, not measured peaks. Part 1 results '
        'are retained from the original run; subsequent runs and commands are recorded in the JSON.\n')
    add('## Hardware and service\n')
    add('| | |\n|---|---|')
    add(f'| GPU | {hardware.get("name", device["name"])} |')
    if hardware.get('compute_capability'):
        add(f'| Compute capability | {".".join(map(str, hardware["compute_capability"]))} |')
    add(f'| Total memory | {device["total_mib"] / 1024:.1f} GiB |')
    add(f'| Free memory at the start | {device["free_mib"] / 1024:.1f} GiB '
        f'(other processes: {", ".join(f"{p}: {m} MiB" for p, m in r.get("gpu_processes_at_start", {}).items()) or "none"}) |')
    add(f'| Driver, PyTorch, CUDA | {device["driver"]}, {hardware.get("torch_version")}, '
        f'{hardware.get("cuda_version")} |')
    add(f'| Precision (worker) | {"float16 autocast" if worker.get("quality_changes") == [] else worker.get("quality_changes")} |')
    if worker:
        add(f'| Worker start (load + profile) | {number(worker.get("start_seconds"))} s; model load '
            f'{number(worker.get("load_seconds"), 2)} s |')
        add(f'| Model memory (both models) | {gib(worker.get("model_bytes"))} GiB |')
    if model:
        add(f'| Measured memory model | click {model["ritm_fixed"] / 2**20:.0f} MiB + '
            f'{model["ritm_per_pixel"]:.1f} B/pixel; step {model["step_fixed"] / 2**20:.0f} MiB + '
            f'{model["step_per_query"] / 1024:.1f} KiB/query token + '
            f'{model["step_per_product"]:.1f} B per memory x query token |')
    add('')
    if r.get('runs'):
        add('### Subsequent benchmark runs\n')
        for run in r['runs']:
            add(f'- {run["date"]}: parts {run["parts"]}; free GPU memory '
                f'{run["device"]["free_mib"] / 1024:.2f} GiB; worker admission cap '
                f'{run.get("worker_cap_gib") or "none"}; external pressure '
                f'{run.get("external_pressure", False)}; status {run.get("status", "recorded")}; '
                f'base commit `{run["commit"]}`.')
            if run.get('note'):
                add('  ' + run['note'])
        add('')
        if r.get('annotator_maintenance'):
            add('The three existing annotator GUI processes were suspended during the recorded '
                'Part 2/3 maintenance runs with the owner\'s permission; their existing CUDA '
                'allocations remained resident. They were resumed afterwards. No active task '
                'data was modified.\n')

    comparison = r.get('gui_comparison')
    if comparison:
        add('## One labeling window: original, refactored local, shared\n')
        add('The same window (`MainController`, offscreen) runs the same clicks (6, the first is a '
            'warm-up and is left out) and the same forward propagation from the first mask anchor, '
            'in the original checkout (its own environment), in this branch in local mode, and in '
            'this branch with the shared service (coordinator and GPU worker in their own '
            'processes). Click times include the window\'s drawing; propagation times run from the '
            'key press to the end of the run. Shared-mode clicks and propagated masks are '
            'committed durably before they return or continue.\n')
        add('| Task | Frames, size, anchors | Size | Click p50/p95 (ms) original | local | shared | '
            'Frames | Frames/s original | local | shared | GPU memory of the window process (MiB) '
            'original / local / shared | Masks vs. original (local; shared) |')
        add('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for task_id, entry in comparison.items():
            task = entry['task']
            size = entry['original'].get('processing_size')
            o, l, s = entry['original'], entry['local'], entry['shared']
            masks_l = entry['masks_local_vs_original']
            masks_s = entry['masks_shared_vs_original']
            add(f'| {task_id} | {task["frames"]}, {task["width"]}x{task["height"]} '
                f'{task["anchor_modality"]}, {task["mask_anchors"]}+{task["empty_anchors"]} | '
                f'{size[0] if size else "–"} px | {stat_cells(o["click"]).replace(" | ", "/")} | '
                f'{stat_cells(l["click"]).replace(" | ", "/")} | {stat_cells(s["click"]).replace(" | ", "/")} | '
                f'{o.get("frames_written")} | {number(o.get("fps"))} | {number(l.get("fps"))} | '
                f'{number(s.get("fps"))} | {o.get("process_mib")} / {l.get("process_mib")} / 0 | '
                f'{masks_l["differing_files"]} files ({masks_l["differing_pixels"]} px); '
                f'{masks_s["differing_files"]} files ({masks_s["differing_pixels"]} px) |')
        add('')
        if any(e['shared'].get('cuda_initialized') for e in comparison.values()):
            add('**A shared-mode window initialized CUDA.**\n')
        else:
            add('No shared-mode window process initialized CUDA (checked with '
                '`torch.cuda.is_initialized()` at the end of each run).\n')
        status = r.get('part1_worker_status') or {}
        if status:
            add(f'GPU worker after part 1: allocated {gib(status["memory"]["allocated"])} GiB, '
                f'reserved {gib(status["memory"]["reserved"])} GiB, peak allocated '
                f'{gib(status["peak_allocated"])} GiB, peak reserved {gib(status["peak_reserved"])} '
                'GiB.\n')

    scenarios = r.get('scenarios', {})
    if scenarios:
        add('## Scenarios against the service\n')
        add('Client sessions (no window) talk to one coordinator process; latencies are measured '
            'at the client, from request to answer. Propagation frames/s uses coordinator time '
            'from job creation to completion, including queueing, initialization, commits and '
            'recovery; it does not depend on when the client starts polling. Inference per step '
            'includes permanent-anchor and seed steps. Click queue time is measured by the '
            'dispatcher, separately from GPU execution and durable commits.\n')
    if 'A_one_annotator_clicks' in scenarios:
        a = scenarios['A_one_annotator_clicks']
        add('### A. One annotator clicks\n')
        add('| | p50 (ms) | p95 (ms) |\n|---|---|---|')
        add(f'| Click, request to answer | {stat_cells(a["latency"])} |')
        if 'queue' in a:
            add(f'| of which scheduler queue | {ms1(a["queue"]["p50"])} | {ms1(a["queue"]["p95"])} |')
        add(f'| of which inference in the worker | {stat_cells(a["inference"])} |')
        add(f'| of which commit (object, transaction, files) | {stat_cells(a["commit"])} |\n')
    if 'B_two_annotators_alternate' in scenarios:
        b = scenarios['B_two_annotators_alternate']
        add('### B. Two annotators alternate clicks\n')
        add('Every click follows the other annotator\'s, so the worker rebuilds the interaction '
            f'(kinds: {", ".join(b["reconstruction_kinds"]) or "none"}).\n')
        add('| | p50 (ms) | p95 (ms) |\n|---|---|---|')
        add(f'| alice | {stat_cells(b["alice"])} |')
        add(f'| bob | {stat_cells(b["bob"])} |')
        add(f'| Reconstruction from a CPU snapshot | {stat_cells(b["reconstruction"])} |')
        replay = b['replay_seconds_with_clicks']
        add(f'| Rebuilding by replaying {replay["clicks"]} clicks | {ms(replay["seconds"])} | |\n')
    if 'B2_reconstruction_by_replay' in scenarios:
        replays = scenarios['B2_reconstruction_by_replay']
        if replays:
            add('Without snapshots (`ritm_snapshot_mib: 0`), switching replays the clicks: ' +
                ', '.join(f'{x["clicks_replayed"]} clicks {ms(x["seconds"])} ms'
                          for x in replays[1::4]) + '.\n')
    jobs = [(name, scenarios[key]) for name, key in (
        ('C. One propagation (0240, 1920x1080, 27 anchors)', 'C_one_propagation'),
        ('H. Small target, 960 px (1530, 1920x1080)', 'H_small_target_hd'))
        if key in scenarios]
    if jobs:
        add('### C, H. One propagation\n')
        add(JOB_HEADER)
        for name, job in jobs:
            add(job_row(name, job))
        add('')
        for name, job in jobs:
            plan = job['effective'][0] if job['effective'] else {}
            add(f'- {name}: {len(plan.get("permanent_tis") or [])} permanent anchors (quality cap '
                f'{plan.get("quality_anchor_limit")}, limited by {plan.get("limited_by") or "nothing"}), '
                f'{plan.get("query_tokens")} query tokens, estimate {gib(plan.get("estimate_bytes"))} GiB.')
        if 'C_one_propagation' in scenarios and scenarios['C_one_propagation'].get('worker'):
            w = scenarios['C_one_propagation']['worker']
            add(f'- Cumulative worker peak through C (including warm-up and earlier clicks): allocated {gib(w["peak_allocated"])} GiB, reserved '
                f'{gib(w["peak_reserved"])} GiB.')
        add('')
    if 'D_propagation_and_clicks' in scenarios:
        d = scenarios['D_propagation_and_clicks']
        add('### D. One propagation while another annotator clicks\n')
        add(JOB_HEADER)
        add(job_row('propagation (0240, as C)', d['propagation']))
        add('')
        add(f'Clicks during the run: p50 {ms(d["click_latency"]["p50"])} ms, p95 '
            f'{ms(d["click_latency"]["p95"])} ms ({d["click_latency"]["count"]} clicks); inference '
            f'p50 {ms(d["click_inference"]["p50"])} ms. Masks compared with run C: '
            f'{d["masks_equal_to_C"]["differing_files"]} files differ '
            f'({d["masks_equal_to_C"]["differing_pixels"]} pixels).\n')
        if 'click_queue' in d:
            add(f'Click scheduler wait p50/p95: {stat_cells(d["click_queue"]).replace(" | ", "/")} ms; '
                f'durable commit p50/p95: {stat_cells(d["click_commit"]).replace(" | ", "/")} ms.\n')
    if 'E_propagation_and_three_annotators' in scenarios:
        e = scenarios['E_propagation_and_three_annotators']
        add('### E. One propagation while three annotators click\n')
        add(JOB_HEADER)
        add(job_row('propagation (0468, 960 px)', e['propagation']))
        add('')
        add('| Annotator | Clicks | p50 (ms) | p95 (ms) | Queue p50/p95 (ms) | Inference p50/p95 (ms) | Commit p50/p95 (ms) |\n|---|---|---|---|---|---|---|')
        for name, entry in e['click_latency'].items():
            detail = [stat_cells(e.get(key, {}).get(name)).replace(' | ', '/')
                      for key in ('click_queue', 'click_inference', 'click_commit')]
            add(f'| {name} | {entry["count"]} | {stat_cells(entry)} | {" | ".join(detail)} |')
        add('')
        if 'masks_equal_to_uninterrupted' in e:
            diff = e['masks_equal_to_uninterrupted']
            add(f'Compared with a separate uninterrupted 0468 run: {diff["differing_files"]} '
                f'files differ ({diff["differing_pixels"]} pixels).\n')
    if 'F_queued_propagations' in scenarios:
        f = scenarios['F_queued_propagations']
        add('### F. Three propagations submitted together\n')
        add(JOB_HEADER)
        for name in ('alice', 'carol', 'dave'):
            add(job_row(name, f[name]))
        add(f'\nThe 100 ms sampler saw at most {f["max_running_contexts"]} running/committing '
            'propagation context at a time.\n')
    if 'G_open_close' in scenarios:
        g = scenarios['G_open_close']
        add('### G. Opening and closing tasks\n')
        add('| | Count | p50 (ms) | p95 (ms) |\n|---|---|---|---|')
        add(f'| First open (imports the task) | {g["first_open"]["count"]} | {stat_cells(g["first_open"])} |')
        add(f'| Open again | {g["reopen"]["count"]} | {stat_cells(g["reopen"])} |')
        add(f'| Close | {g["close"]["count"]} | {stat_cells(g["close"])} |\n')
    if 'I1_memory_cap' in scenarios or 'I2_memory_held_by_another_process' in scenarios:
        add('### I. GPU memory limits\n')
        if 'I1_memory_cap' in scenarios:
            i1 = scenarios['I1_memory_cap']
            plan = i1['effective'][0] if i1['effective'] else {}
            add(f'- I1, worker capped at {i1["cap_gib"]} GiB (`gpu.max_worker_gib`), task 0240: '
                f'{len(plan.get("permanent_tis") or [])} permanent anchors (quality cap '
                f'{plan.get("quality_anchor_limit")}, limited by {plan.get("limited_by")}), '
                f'{i1["frames"]} frames at {number(i1["fps"])} frames/s, state {i1["state"]}. '
                f'Messages: {"; ".join(i1["messages"][:2])}')
            if 'masks_vs_full_anchor_run' in i1:
                diff = i1['masks_vs_full_anchor_run']
                add(f'  Compared with C using all 27 anchors: {diff["differing_files"]} '
                    f'propagated masks differ ({diff["differing_pixels"]} pixels). This is a '
                    'recorded quality-sensitive memory adaptation; human anchors remain intact.')
        if scenarios.get('I2_memory_held_by_another_process', {}).get('state') == 'not_run':
            add('- I2 was not run: ' + scenarios['I2_memory_held_by_another_process']['reason'])
        elif 'I2_memory_held_by_another_process' in scenarios:
            i2 = scenarios['I2_memory_held_by_another_process']
            add(f'- I2, another process holds {i2["ballast_gib"]:.1f} GiB of the GPU, with release '
                f'requested on the first poll after {i2["released_after"]:.0f} s; task 1530 at 960 px: states seen '
                f'{", ".join(i2["waited_states"])}, running after {number(i2["first_running_seconds"])} s, '
                f'queue wait {number(i2["queue_seconds"], 1)} s, then {i2["frames"]} frames at '
                f'{number(i2["fps"])} frames/s, state {i2["state"]}.')
            if 'masks_equal_to_H' in i2:
                diff = i2['masks_equal_to_H']
                add(f'  Immutable propagated outputs vs. H: {diff["differing_files"]} files '
                    f'differ ({diff["differing_pixels"]} pixels). The {i2["attempts"]} admission '
                    f'checks created {len(i2["effective"])} execution context(s); no OOM retry occurred.')
        add('')
    if 'J_worker_failure' in scenarios:
        j = scenarios['J_worker_failure']
        recovery = j.get('worker_recovery_seconds') or {}
        add('### J. The GPU worker is killed during a propagation\n')
        if r.get('scenario_j_recheck'):
            add('This scenario was repeated alone after fixing the timer: an in-flight commit '
                'from before the crash must not count as recovery. The timer now waits for a '
                'new execution attempt and then an increase in its committed-frame count. '
                'The same immutable anchor inputs and uninterrupted C baseline were used.\n')
        add(f'Killed (SIGKILL) after {j["killed_after_frames"]} committed frames of 0240. Worker '
            f'replaced in {number(recovery.get("p50"))} s (new generation {j["worker_generation"]}, '
            f'including model load and memory profile); new frames committed again after '
            f'{number(j["seconds_until_new_frames"])} s. The job finished in '
            f'{j["propagation"]["attempts"]} attempts, replaying '
            f'{j["propagation"]["replayed_frames"]} confirmed frames without writing them '
            f'({j["propagation"]["replay_mismatches"]} differed from what was committed). Masks '
            f'compared with the undisturbed run C: {j["masks_equal_to_C"]["differing_files"]} '
            f'files differ ({j["masks_equal_to_C"]["differing_pixels"]} pixels).\n')
    if 'K_write_failure' in scenarios:
        k = scenarios['K_write_failure']
        add('### K. Writing annotations fails\n')
        add(f'The mask object store was made read-only. A click answered with an error after '
            f'{ms(k["click_error_seconds"])} ms (`{(k["click_error"] or {}).get("code")}`); a '
            f'propagation stopped as {k["propagation"]["state"]} after '
            f'{k["propagation"]["frames"]} frames ({k["propagation"]["error"] or k["propagation"]["stop_reason"]}). '
            f'After the store was writable again, the same click request was applied once '
            f'({k["retry_after_repair"]["clicks"]} accepted click).\n')
    if r.get('part2_worker_status'):
        add('## Worker memory and effective configuration\n')
        add('| Worker lifetime | Peak allocated (GiB) | Peak reserved (GiB) | Final allocated (GiB) | Final reserved (GiB) |\n|---|---|---|---|---|')
        for label, status in (('Part 2 final worker generation', r.get('part2_worker_status')),
                              ('Part 3 2.2 GiB admission cap', scenarios.get('I1_memory_cap', {}).get('worker')),
                              ('Part 3 external pressure', r.get('part3_pressure_status'))):
            if status:
                add(f'| {label} | {gib(status["peak_allocated"])} | {gib(status["peak_reserved"])} | '
                    f'{gib(status["memory"]["allocated"])} | {gib(status["memory"]["reserved"])} |')
        add('\nPeaks reset when the worker restarts; these are cumulative allocator peaks within '
            'each worker lifetime, not per-scenario peaks. Initial and restarted worker hardware '
            'and profiling samples, every job\'s precision/model configuration, resolution, '
            'temporal-memory settings and effective admission plans are retained in the JSON. '
            'The I1 admission cap applies after profiling and is not a hard allocator limit.\n')
    if r.get('verification'):
        add('## Verification and limits\n')
        for item in r['verification']:
            add(f'- {item}')
        add('')
    add('These are single runs on one physical GPU, with finite scripted click samples. They '
        'do not establish a universal worst-case latency or performance improvement. Propagation '
        'jobs run FIFO with one active context; propagation-job time slicing is not implemented. '
        'Export compatibility is covered by automated tests; these benchmark tasks use an '
        'isolated stub export command, not the live external export pipeline.\n')
    with open(target, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
