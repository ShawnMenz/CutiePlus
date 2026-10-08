"""
Benchmark of the shared GPU service against the original local implementation, on copies of real
RGB-T tasks (frames are read in place; nothing in the source task root is written).

    uv run python scripts/benchmark_service.py --source workspace/cutie_tasks \
        --scratch /path/to/scratch --original-repo /path/to/original/checkout --output docs/benchmarks

Part 1 drives one labeling window offscreen (scripts/bench/gui_driver.py) through the same clicks
and forward propagation per task, three times: the original checkout (its own environment), this
branch in local mode, and this branch in shared mode; the resulting masks are compared.
Part 2 runs the scenarios A-K against a coordinator process with client sessions.
Results go to <output>/<date>-<gpu>.json; scripts/bench/report.py writes the Markdown report.
"""
import argparse
import hashlib
import json
import os
import select
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
REPO = path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, path.join(HERE, 'bench'))

from common import click_plan, gpu_device, gpu_processes, prepare, stats, task_summary  # noqa: E402
from cutieplus.client import ClientError, ServiceClient, new_request_id  # noqa: E402

TASKS = [('main', '1629'), ('main', '0240'), ('main', '0468'), ('main', '1530'), ('main', '2363'),
         ('main', '0327')]
MEMORY = {'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4, 'max_mem_frames': 9,
          'max_num_tokens': 10000}  # what the window's controls send


def log(text: str) -> None:
    print(f'[{datetime.now():%H:%M:%S}] {text}', flush=True)


def mask_digests(workspace: str) -> dict:
    masks = path.join(workspace, 'masks')
    digests = {}
    for name in sorted(os.listdir(masks)):
        if name.endswith('.png'):
            from PIL import Image
            digests[name] = hashlib.sha256(np.array(Image.open(path.join(masks, name))).tobytes()
                                           ).hexdigest()
    return digests


def mask_differences(a: str, b: str) -> dict:
    from PIL import Image
    da, db = mask_digests(a), mask_digests(b)
    differing = []
    pixels = 0
    for name in sorted(set(da) | set(db)):
        if da.get(name) != db.get(name):
            differing.append(name)
            if name in da and name in db:
                ma = np.array(Image.open(path.join(a, 'masks', name)))
                mb = np.array(Image.open(path.join(b, 'masks', name)))
                pixels += int((ma != mb).sum())
    return {'files_a': len(da), 'files_b': len(db), 'differing_files': len(differing),
            'differing_pixels': pixels, 'examples': differing[:5]}


class Coordinator:
    """The service as deployed: python -m cutieplus.coordinator start, in its own process."""

    def __init__(self, task_root: str, state_dir: str, settings: dict = None, label: str = ''):
        self.task_root = task_root
        self.state_dir = state_dir
        from cutieplus.paths import default_socket_path
        self.socket = default_socket_path(state_dir)
        os.makedirs(state_dir, exist_ok=True)
        command = [sys.executable, '-m', 'cutieplus.coordinator', 'start', '--tasks', task_root,
                   '--state-dir', state_dir]
        for key, value in (settings or {}).items():
            command += ['--set', f'{key}={json.dumps(value)}']
        self.log_file = open(path.join(state_dir, f'coordinator{label}.out'), 'w')
        started = time.perf_counter()
        self.process = subprocess.Popen(command, cwd=REPO, stdout=self.log_file,
                                        stderr=subprocess.STDOUT)
        self.client = None
        while time.perf_counter() - started < 600:
            if self.process.poll() is not None:
                raise RuntimeError(f'coordinator exited: see {self.log_file.name}')
            if path.exists(self.socket):
                try:
                    probe = ServiceClient(self.socket, 'bench-admin', ':0', reconnect_seconds=1)
                    if probe.connect().get('worker_ready'):
                        self.client = probe
                        self.client.start_heartbeat()
                        break
                    probe.close()
                except ClientError:
                    pass
            time.sleep(0.3)
        self.ready_seconds = time.perf_counter() - started
        if self.client is None:
            raise RuntimeError('coordinator did not become ready')

    def status(self) -> dict:
        return self.client.call('status')

    def gpu_status(self) -> dict:
        return self.client.call('gpu_status')

    def client_for(self, name: str, display: str) -> ServiceClient:
        client = ServiceClient(self.socket, name, display)
        client.connect()
        client.start_heartbeat()
        return client

    def stop(self) -> None:
        try:
            self.client.call('shutdown')
        except Exception:
            pass
        try:
            self.process.wait(60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(30)
        if self.client is not None:
            self.client.close()
        self.log_file.close()


# ------------------------------------------------------------------------------------------------
def run_gui(python: str, repo: str, mode: str, workspace: str, plan_file: str, start: int,
            output: str, socket: str = None, propagate: bool = True) -> dict:
    command = [python, path.join(HERE, 'bench', 'gui_driver.py'), '--mode', mode, '--repo', repo,
               '--workspace', workspace, '--plan', plan_file, '--start', str(start), '--output',
               output]
    if socket:
        command += ['--socket', socket]
    if not propagate:
        command.append('--no-propagation')
    environment = dict(os.environ, PYTHONPATH=repo, PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True,
                               timeout=3600)
    if completed.returncode != 0:
        raise RuntimeError(f'{mode} driver failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}')
    with open(output) as f:
        return json.load(f)


def summarize_gui(result: dict) -> dict:
    clicks = result['click_seconds']
    summary = {k: result.get(k) for k in ('model_load_seconds', 'model_bytes', 'open_seconds',
                                          'processing_size', 'resolution_message',
                                          'propagation_seconds', 'save_join_seconds',
                                          'frames_written', 'peak_allocated', 'peak_reserved',
                                          'process_mib', 'cuda_initialized')}
    summary['first_click_seconds'] = clicks[0] if clicks else None
    summary['click'] = stats(clicks[1:])
    if result.get('frames_written'):
        summary['fps'] = result['frames_written'] / result['propagation_seconds']
    return summary


def part1(args, out: dict) -> None:
    """The same window session in the original checkout, this branch locally, and shared."""
    scratch = args.scratch
    comparison = out.setdefault('gui_comparison', {})
    roots = {}
    for mode in ('original', 'local', 'shared'):
        roots[mode] = prepare(args.source, [(s, q, s) for s, q in TASKS],
                              path.join(scratch, f'part1-{mode}', 'tasks'))
    coordinator = Coordinator(path.join(scratch, 'part1-shared', 'tasks'),
                              path.join(scratch, 'part1-shared', 'state'), label='-part1')
    out['coordinator_ready_seconds'] = coordinator.ready_seconds
    out['worker'] = coordinator.status()['worker']
    try:
        for split, sequence in TASKS:
            task_id = f'{split}/{sequence}'
            summary = task_summary(roots['original'][task_id])
            plan_file = path.join(scratch, f'plan-{sequence}.json')
            with open(plan_file, 'w') as f:
                json.dump(click_plan(roots['original'][task_id], 6), f)
            start = summary['first_mask_anchor']
            entry = {'task': {k: v for k, v in summary.items() if k != 'names'}}
            for mode, python, repo in (('original', args.original_python, args.original_repo),
                                       ('local', sys.executable, REPO),
                                       ('shared', sys.executable, REPO)):
                log(f'part 1: {task_id} {mode}')
                result = run_gui(python, repo, 'shared' if mode == 'shared' else 'local',
                                 roots[mode][task_id], plan_file, start,
                                 path.join(scratch, f'gui-{mode}-{sequence}.json'),
                                 coordinator.socket if mode == 'shared' else None)
                entry[mode] = summarize_gui(result)
            entry['masks_local_vs_original'] = mask_differences(roots['local'][task_id],
                                                                roots['original'][task_id])
            entry['masks_shared_vs_original'] = mask_differences(roots['shared'][task_id],
                                                                 roots['original'][task_id])
            comparison[task_id] = entry
            save(args, out)
        out['part1_worker_status'] = coordinator.gpu_status()
        out['part1_metrics'] = coordinator.status()['metrics']
    finally:
        coordinator.stop()


# ------------------------------------------------------------------------------------------------
def open_task(client, root, task_id):
    split, sequence = task_id.split('/')
    return client.call('open_task', workspace=path.join(root, split, sequence))


def lease_args(lease):
    return {'task_id': lease['task_id'], 'lease_epoch': lease['lease_epoch']}


def clicks_on(client, lease, root, task_id, count, think=0.0, results=None, label=''):
    """count clicks following the task's click plan; returns their latencies (s) and details."""
    split, sequence = task_id.split('/')
    plan = click_plan(path.join(root, split, sequence), count)
    names = task_summary(path.join(root, split, sequence))['names']
    latencies, details = [], []
    done = 0
    for ti, clicks in plan * 10:
        interaction = client.call('open_interaction', frame=names[ti], modality='visual'
                                  if task_summary(path.join(root, split, sequence))[
                                      'anchor_modality'] == 'visual' else 'thermal',
                                  target_object=1, **lease_args(lease))['interaction_id']
        for x, y, negative in clicks:
            begin = time.perf_counter()
            reply = client.call('click', interaction_id=interaction, x=x, y=y,
                                is_negative=negative, request_id=new_request_id(),
                                **lease_args(lease))
            latencies.append(time.perf_counter() - begin)
            details.append({k: reply.get(k) for k in ('reconstruction', 'reconstruction_seconds',
                                                      'inference_seconds', 'gpu_request_seconds',
                                                      'queue_seconds', 'commit_seconds')})
            done += 1
            if think:
                time.sleep(think)
            if done >= count:
                break
        client.call('close_interaction', interaction_id=interaction)
        if done >= count:
            break
    assert done == count, f'{task_id}: only {done}/{count} clicks completed'
    if results is not None:
        results[label] = {'latencies': latencies, 'details': details}
    return latencies, details


def start_job(client, lease, start_frame):
    return client.call('start_propagation', start_frame=start_frame, direction='forward',
                       memory=MEMORY, request_id=new_request_id(), **lease_args(lease))['job_id']


def wait_job(client, job_id, timeout=3600, on_poll=None):
    since = 0
    messages = []
    begin = time.perf_counter()
    while time.perf_counter() - begin < timeout:
        reply = client.call('poll_propagation', job_id=job_id, since=since,
                            messages_since=len(messages))
        since = reply['next']
        messages += reply['messages']
        if on_poll is not None:
            on_poll(reply)
        if reply['state'] in ('finished', 'cancelled', 'failed'):
            reply['all_messages'] = messages
            reply['poll_seconds'] = time.perf_counter() - begin
            reply['wall_seconds'] = reply['metrics']['elapsed_seconds']
            return reply
        time.sleep(0.05)
    raise RuntimeError('job did not finish')


def job_summary(reply):
    metrics = reply['metrics']
    steps = metrics['steps'] or 1
    return {'state': reply['state'], 'stop_reason': reply['stop_reason'], 'frames': reply['next'],
            'attempts': reply['attempt'], 'wall_seconds': reply['wall_seconds'],
            'fps': reply['next'] / reply['wall_seconds'] if reply['wall_seconds'] else None,
            'queue_seconds': metrics['queue_seconds'],
            'inference_seconds': metrics['inference_seconds'],
            'inference_per_step': metrics['inference_seconds'] / steps,
            'commit_seconds': metrics['commit_seconds'],
            'commit_per_frame': metrics['commit_seconds'] / max(1, reply['next']),
            'commit_wait_seconds': metrics['commit_wait_seconds'],
            'replayed_frames': metrics['replayed_frames'],
            'replay_mismatches': metrics['replay_mismatches'],
            'oom_retries': metrics['oom_retries'], 'worker_restarts': metrics['worker_restarts'],
            'run_seconds': metrics.get('run_seconds'),
            'effective': reply['effective'],
            'messages': reply['all_messages']}


def check_job(reply):
    assert reply['state'] == 'finished', (reply['state'], reply.get('error'), reply['all_messages'])
    assert reply['metrics']['replay_mismatches'] == 0, reply['metrics']


def check_masks(difference):
    assert difference['differing_files'] == 0, difference


def observe_recovery(killed, reply, now):
    """A pre-crash GPU result may still commit after SIGKILL. Start observing new commits
    only after a new execution attempt is visible, then wait for its count to advance."""
    if reply['attempt'] <= killed['attempt']:
        return
    if 'confirmed_before_replay' not in killed:
        killed['confirmed_before_replay'] = reply['next']
    elif 'resumed' not in killed and reply['next'] > killed['confirmed_before_replay']:
        killed['resumed'] = now - killed['at']


def database_counts(state_dir, task_id):
    """Independent read-only evidence that replay/failure did not commit extra writes."""
    with sqlite3.connect(f'file:{path.join(state_dir, "service.db")}?mode=ro', uri=True) as db:
        return dict(zip(('input_revision', 'content_revision'), db.execute(
            'SELECT input_revision, content_revision FROM tasks WHERE task_id = ?',
            (task_id,)).fetchone()))


def service_settings(args):
    return {'gpu.max_worker_gib': args.worker_cap_gib} if args.worker_cap_gib else {}


def job_configurations(state_dir):
    with sqlite3.connect(f'file:{path.join(state_dir, "service.db")}?mode=ro', uri=True) as db:
        records = db.execute('SELECT job_id, snapshot FROM jobs ORDER BY created_at').fetchall()
    keys = ('task_id', 'input_revision', 'model_identity', 'model_configuration',
            'anchor_modality', 'start_frame', 'direction', 'resolution', 'memory')
    return {job: {key: snapshot.get(key) for key in keys}
            for job, raw in records for snapshot in [json.loads(raw)]}


def worker_profiles(state_dir):
    with sqlite3.connect(f'file:{path.join(state_dir, "service.db")}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT generation, hello FROM worker_starts ORDER BY at').fetchall()
    return [{'generation': generation, **json.loads(hello)} for generation, hello in rows]


def compare_committed_masks(state_a, task_a, state_b, task_b):
    """Compare immutable propagation outputs, excluding any later human edits to task views."""
    from PIL import Image

    def frames(state, task):
        with sqlite3.connect(f'file:{path.join(state, "service.db")}?mode=ro', uri=True) as db:
            return dict(db.execute("SELECT frame, after_object FROM edits WHERE task_id = ? "
                                   "AND op = 'propagate' ORDER BY edit_id", (task,)).fetchall())

    a, b = frames(state_a, task_a), frames(state_b, task_b)
    differing, pixels = [], 0
    for frame in sorted(a.keys() | b.keys()):
        if a.get(frame) == b.get(frame):
            continue
        differing.append(frame)
        if frame in a and frame in b:
            masks = [np.array(Image.open(path.join(state, 'objects', sha[:2], sha + '.png')))
                     for state, sha in ((state_a, a[frame]), (state_b, b[frame]))]
            pixels += int(np.count_nonzero(masks[0] != masks[1]))
    return {'frames_a': len(a), 'frames_b': len(b), 'differing_files': len(differing),
            'differing_pixels': pixels, 'examples': differing[:5]}


def first_anchor(root, task_id):
    split, sequence = task_id.split('/')
    summary = task_summary(path.join(root, split, sequence))
    return summary['names'][summary['first_mask_anchor']]


def benchmark_worker_failure(args, out, root, coordinator, alice, comparison_workspace=None):
    # J: the GPU worker dies during a propagation
    scenarios = out.setdefault('scenarios', {})
    log('J: worker failure')
    j_lease = open_task(alice, root, 'copyj/0240')
    job = start_job(alice, j_lease, first_anchor(root, 'copyj/0240'))
    killed = {}

    def kill_when(reply):
        if 'at' not in killed and reply['next'] >= 60:
            pid = coordinator.status()['worker']['pid']
            os.kill(pid, signal.SIGKILL)
            killed.update(at=time.perf_counter(), frames=reply['next'], pid=pid,
                          attempt=reply['attempt'])
        if 'at' in killed:
            observe_recovery(killed, reply, time.perf_counter())

    recovered = wait_job(alice, job, on_poll=kill_when)
    check_job(recovered)
    assert recovered['metrics']['worker_restarts'] == 1
    assert recovered['metrics']['replayed_frames'] >= killed['frames']
    status = coordinator.status()
    assert killed['resumed'] >= status['metrics']['worker_recovery']['p50']
    scenarios['J_worker_failure'] = {
        'propagation': job_summary(recovered), 'killed_after_frames': killed.get('frames'),
        'confirmed_before_replay': killed.get('confirmed_before_replay'),
        'seconds_until_new_frames': killed.get('resumed'),
        'worker_recovery_seconds': status['metrics'].get('worker_recovery'),
        'worker_generation': status['worker']['generation'],
        'masks_equal_to_C': mask_differences(path.join(root, 'copyj', '0240'),
                                             comparison_workspace or path.join(root, 'copyc', '0240'))}
    check_masks(scenarios['J_worker_failure']['masks_equal_to_C'])
    alice.call('close_task', **lease_args(j_lease))
    save(args, out)


def part2(args, out: dict) -> None:
    scratch = path.join(args.scratch, 'part2')
    tasks = [(s, q, s) for s, q in TASKS] + [('main', '0240', 'copyc'), ('main', '0240', 'copyd'),
                                            ('main', '0240', 'copyj'), ('main', '0468', 'copye'),
                                            ('main', '1629', 'copyk')]
    root = path.join(scratch, 'tasks')
    prepare(args.source, tasks, root)
    scenarios = out.setdefault('scenarios', {})
    coordinator = Coordinator(root, path.join(scratch, 'state'), service_settings(args), label='-part2')
    out['part2_worker'] = coordinator.status()['worker']
    out['part2_settings'] = service_settings(args)
    alice = coordinator.client_for('alice', ':2')
    bob = coordinator.client_for('bob', ':3')
    carol = coordinator.client_for('carol', ':4')
    dave = coordinator.client_for('dave', ':5')
    try:
        # G: opening and closing tasks (the first open of a task imports it)
        log('G: open/close')
        first_open, reopen, close = [], [], []
        for task_id in ('main/1629', 'main/0240', 'main/0468', 'main/2363'):
            for index in range(4):
                begin = time.perf_counter()
                lease = open_task(alice, root, task_id)
                (first_open if index == 0 else reopen).append(time.perf_counter() - begin)
                begin = time.perf_counter()
                alice.call('close_task', **lease_args(lease))
                close.append(time.perf_counter() - begin)
        scenarios['G_open_close'] = {'first_open': stats(first_open), 'reopen': stats(reopen),
                                     'close': stats(close)}
        save(args, out)

        a1629 = open_task(alice, root, 'main/1629')
        b0240 = open_task(bob, root, 'main/0240')
        c2363 = open_task(carol, root, 'main/2363')
        d0327 = open_task(dave, root, 'main/0327')

        # A: one annotator clicks
        log('A: one annotator')
        latencies, details = clicks_on(alice, a1629, root, 'main/1629', 31)
        scenarios['A_one_annotator_clicks'] = {'latency': stats(latencies[1:]),
                                               'queue': stats([d['queue_seconds'] for d in details[1:]]),
                                               'inference': stats([d['inference_seconds'] for d in details[1:]]),
                                               'commit': stats([d['commit_seconds'] for d in details[1:]])}
        save(args, out)

        # B: two annotators alternate (every click rebuilds the other's interaction)
        log('B: two annotators alternate')
        names_a = task_summary(path.join(root, 'main', '1629'))['names']
        names_b = task_summary(path.join(root, 'main', '0240'))['names']
        plan_a = click_plan(path.join(root, 'main', '1629'), 30)
        plan_b = click_plan(path.join(root, 'main', '0240'), 30)
        ia = alice.call('open_interaction', frame=names_a[plan_a[0][0]], modality='visual',
                        target_object=1, **lease_args(a1629))['interaction_id']
        ib = bob.call('open_interaction', frame=names_b[plan_b[0][0]], modality='visual',
                      target_object=1, **lease_args(b0240))['interaction_id']
        pts_a = [c for _, clicks in plan_a for c in clicks]
        pts_b = [c for _, clicks in plan_b for c in clicks]
        alternated = {'alice': [], 'bob': []}
        rebuild = []
        for index in range(15):
            for name, client, lease, interaction, points in (
                    ('alice', alice, a1629, ia, pts_a), ('bob', bob, b0240, ib, pts_b)):
                x, y, negative = points[index % len(points)]
                begin = time.perf_counter()
                reply = client.call('click', interaction_id=interaction, x=x, y=y,
                                    is_negative=negative, request_id=new_request_id(),
                                    **lease_args(lease))
                alternated[name].append(time.perf_counter() - begin)
                if reply.get('reconstruction'):
                    rebuild.append((reply['reconstruction'], reply['reconstruction_seconds'],
                                    reply['clicks']))
        before_replay = database_counts(coordinator.state_dir, a1629['task_id'])
        # Bob's reply is the last loop result; check Alice against the committed mask on disk.
        from PIL import Image
        accepted_mask = np.array(Image.open(path.join(root, 'main', '1629', 'masks',
                                                       names_a[plan_a[0][0]] + '.png')))
        replay_check = alice.call('replay_interaction', interaction_id=ia)
        assert np.array_equal(accepted_mask, replay_check['mask'])
        assert database_counts(coordinator.state_dir, a1629['task_id']) == before_replay
        scenarios['B_two_annotators_alternate'] = {
            'alice': stats(alternated['alice'][1:]), 'bob': stats(alternated['bob'][1:]),
            'reconstruction': stats([seconds for kind, seconds, _ in rebuild]),
            'reconstruction_kinds': sorted({kind for kind, _, _ in rebuild}),
            'replay_matches_committed': True, 'replay_changes_revisions': False,
            'replay_seconds_with_clicks': {'clicks': replay_check['clicks'],
                                           'seconds': replay_check['reconstruction_seconds']}}
        alice.call('close_interaction', interaction_id=ia)
        bob.call('close_interaction', interaction_id=ib)
        save(args, out)

        # C: one propagation alone (0240 copy C, 1920x1080, 27 anchors)
        log('C: one propagation')
        c_lease = open_task(alice, root, 'copyc/0240')
        alone = wait_job(alice, start_job(alice, c_lease, first_anchor(root, 'copyc/0240')))
        check_job(alone)
        scenarios['C_one_propagation'] = job_summary(alone)
        scenarios['C_one_propagation']['worker'] = coordinator.gpu_status()
        alice.call('close_task', **lease_args(c_lease))
        save(args, out)

        # D: the same propagation while another annotator clicks
        log('D: propagation + one annotator clicking')
        d_lease = open_task(alice, root, 'copyd/0240')
        job = start_job(alice, d_lease, first_anchor(root, 'copyd/0240'))
        results = {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            clicker = pool.submit(clicks_on, bob, b0240, root, 'main/0240', 30, 0.3,
                                  results, 'bob')
            during = wait_job(alice, job)
            clicker.result()
        check_job(during)
        scenarios['D_propagation_and_clicks'] = {
            'propagation': job_summary(during),
            'click_latency': stats(results['bob']['latencies']),
            'click_inference': stats([d['inference_seconds'] for d in results['bob']['details']]),
            'click_queue': stats([d['queue_seconds'] for d in results['bob']['details']]),
            'click_commit': stats([d['commit_seconds'] for d in results['bob']['details']]),
            'masks_equal_to_C': mask_differences(path.join(root, 'copyd', '0240'),
                                                 path.join(root, 'copyc', '0240'))}
        check_masks(scenarios['D_propagation_and_clicks']['masks_equal_to_C'])
        alice.call('close_task', **lease_args(d_lease))
        save(args, out)

        # E: a propagation (small target, 960 px) while three annotators click
        log('E: propagation + three annotators clicking')
        e_baseline = open_task(alice, root, 'copye/0468')
        e_alone = wait_job(alice, start_job(alice, e_baseline, first_anchor(root, 'copye/0468')))
        check_job(e_alone)
        alice.call('close_task', **lease_args(e_baseline))
        e_lease = open_task(alice, root, 'main/0468')
        job = start_job(alice, e_lease, first_anchor(root, 'main/0468'))
        results = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(clicks_on, client, lease, root, task_id, 20, 0.3, results, name)
                       for name, client, lease, task_id in (('bob', bob, b0240, 'main/0240'),
                                                          ('carol', carol, c2363, 'main/2363'),
                                                          ('dave', dave, d0327, 'main/0327'))]
            during = wait_job(alice, job)
            for future in futures:
                future.result()
        check_job(during)
        scenarios['E_propagation_and_three_annotators'] = {
            'propagation': job_summary(during),
            'click_latency': {name: stats(r['latencies']) for name, r in results.items()},
            'click_queue': {name: stats([d['queue_seconds'] for d in r['details']])
                            for name, r in results.items()},
            'click_inference': {name: stats([d['inference_seconds'] for d in r['details']])
                                for name, r in results.items()},
            'click_commit': {name: stats([d['commit_seconds'] for d in r['details']])
                             for name, r in results.items()},
            'baseline': job_summary(e_alone),
            'masks_equal_to_uninterrupted': mask_differences(path.join(root, 'main', '0468'),
                                                           path.join(root, 'copye', '0468')),
            'all_clicks': stats([v for r in results.values() for v in r['latencies']])}
        check_masks(scenarios['E_propagation_and_three_annotators']['masks_equal_to_uninterrupted'])
        alice.call('close_task', **lease_args(e_lease))
        save(args, out)

        # F: three propagation jobs at once (one GPU context at a time)
        log('F: queued propagations')
        lease_f = {'alice': open_task(alice, root, 'main/1629'),
                   'carol': c2363, 'dave': d0327}
        clients = {'alice': alice, 'carol': carol, 'dave': dave}
        tasks_f = {'alice': 'main/1629', 'carol': 'main/2363', 'dave': 'main/0327'}
        jobs = {name: start_job(clients[name], lease_f[name], first_anchor(root, tasks_f[name]))
                for name in ('alice', 'carol', 'dave')}
        contexts = []
        sampler_stop = threading.Event()

        def sample():
            while not sampler_stop.is_set():
                status = coordinator.status()
                contexts.append(sum(1 for job in status['jobs'] if job['state'] in
                                    ('running', 'committing')))
                time.sleep(0.1)

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            done = {name: wait_job(clients[name], job) for name, job in jobs.items()}
        finally:
            sampler_stop.set()
            sampler.join()
        for reply in done.values():
            check_job(reply)
        assert max(contexts or [0]) == 1, contexts
        scenarios['F_queued_propagations'] = {
            name: job_summary(reply) for name, reply in done.items()}
        scenarios['F_queued_propagations']['max_running_contexts'] = max(contexts or [0])
        save(args, out)

        # H: small-target policy on an HD sequence (1530: 960 px, upscaled)
        log('H: small target, HD')
        h_lease = open_task(dave, root, 'main/1530')
        small = wait_job(dave, start_job(dave, h_lease, first_anchor(root, 'main/1530')))
        check_job(small)
        scenarios['H_small_target_hd'] = dict(job_summary(small), resolution=h_lease['resolution'])
        dave.call('close_task', **lease_args(h_lease))
        save(args, out)

        benchmark_worker_failure(args, out, root, coordinator, alice)

        # K: writing annotations fails (the object store becomes read-only)
        log('K: write failure')
        objects = path.join(scratch, 'state', 'objects')
        directories = [objects] + [path.join(objects, d) for d in os.listdir(objects)]
        modes = {d: os.stat(d).st_mode for d in directories}
        k_lease = open_task(alice, root, 'copyk/1629')
        names_k = task_summary(path.join(root, 'copyk', '1629'))['names']
        interaction = alice.call('open_interaction', frame=names_k[5], modality='visual',
                                 target_object=1, **lease_args(k_lease))['interaction_id']
        before_failure = database_counts(coordinator.state_dir, k_lease['task_id'])
        try:
            for directory in directories:
                os.chmod(directory, 0o555)
            begin = time.perf_counter()
            try:
                alice.call('click', interaction_id=interaction, x=50, y=50, is_negative=False,
                           request_id='k-click', **lease_args(k_lease))
                click_error = None
            except ClientError as exc:
                click_error = {'code': exc.code, 'message': str(exc)[:200]}
            click_error_seconds = time.perf_counter() - begin
            assert click_error is not None, 'write failure was not injected'
            assert database_counts(coordinator.state_dir, k_lease['task_id']) == before_failure
            job = start_job(alice, k_lease, first_anchor(root, 'copyk/1629'))
            failed = wait_job(alice, job)
            assert failed['state'] == 'failed', failed['state']
            assert 'saving' in (failed.get('error') or ''), failed.get('error')
        finally:
            for directory, mode in modes.items():
                os.chmod(directory, mode)
        before_retry = database_counts(coordinator.state_dir, k_lease['task_id'])
        retried = alice.call('click', interaction_id=interaction, x=50, y=50, is_negative=False,
                             request_id='k-click', **lease_args(k_lease))
        duplicate = alice.call('click', interaction_id=interaction, x=50, y=50, is_negative=False,
                               request_id='k-click', **lease_args(k_lease))
        after_retry = database_counts(coordinator.state_dir, k_lease['task_id'])
        assert retried['clicks'] == duplicate['clicks'] == 1
        assert duplicate['replayed']
        assert after_retry['content_revision'] == before_retry['content_revision'] + 1
        assert after_retry['input_revision'] == before_retry['input_revision'] + 1
        assert coordinator.gpu_status()['job'] is None
        scenarios['K_write_failure'] = {
            'click_error': click_error, 'click_error_seconds': click_error_seconds,
            'propagation': {'state': failed['state'], 'stop_reason': failed['stop_reason'],
                            'error': failed.get('error'), 'frames': failed['next']},
            'retry_after_repair': {'clicks': retried['clicks'],
                                   'content_revision': retried['content_revision'],
                                   'duplicate_replayed': duplicate['replayed']},
            'revisions_before_failure': before_failure, 'revisions_before_retry': before_retry,
            'revisions_after_retry': after_retry}
        alice.call('close_interaction', interaction_id=interaction)
        alice.call('close_task', **lease_args(k_lease))
        out['part2_metrics'] = coordinator.status()['metrics']
        out['part2_worker_status'] = coordinator.gpu_status()
        out['part2_job_configurations'] = job_configurations(coordinator.state_dir)
        save(args, out)
    finally:
        for client in (alice, bob, carol, dave):
            client.close()
        coordinator.stop()


BALLAST = '''
import sys, time, torch
size = int(sys.argv[1])
block = torch.empty(size, dtype=torch.uint8, device='cuda')
block.fill_(1)
torch.cuda.synchronize()
print('allocated', flush=True)
sys.stdin.readline()
'''


def part3(args, out: dict) -> None:
    """Memory limits: a cap that leaves room for few anchors, a real allocation by another
    process that makes a job wait, and the replay of interactions without snapshots."""
    scratch = path.join(args.scratch, 'part3')
    root = path.join(scratch, 'tasks')
    prepare(args.source, [('main', '0240', 'main'), ('main', '1530', 'main'),
                          ('main', '1629', 'main')], root)
    scenarios = out.setdefault('scenarios', {})

    log('I1: worker memory cap')
    coordinator = Coordinator(root, path.join(scratch, 'state-cap'),
                              {'gpu.max_worker_gib': 2.2, 'ritm_snapshot_mib': 0}, label='-cap')
    alice = coordinator.client_for('alice', ':2')
    try:
        lease = open_task(alice, root, 'main/0240')
        capped = wait_job(alice, start_job(alice, lease, first_anchor(root, 'main/0240')))
        check_job(capped)
        assert capped['effective'][0]['limited_by'] == 'memory', capped['effective']
        scenarios['I1_memory_cap'] = dict(job_summary(capped), cap_gib=2.2,
                                          worker=coordinator.gpu_status())
        baseline_state = path.join(args.scratch, 'part2', 'state')
        if path.isfile(path.join(baseline_state, 'service.db')):
            scenarios['I1_memory_cap']['masks_vs_full_anchor_run'] = compare_committed_masks(
                coordinator.state_dir, lease['task_id'], baseline_state, 'copyc/0240')
        out['part3_cap_worker'] = coordinator.status()['worker']
        out['part3_cap_job_configurations'] = job_configurations(coordinator.state_dir)
        alice.call('close_task', **lease_args(lease))
        # B without snapshots: every switch replays the interaction's clicks
        log('B2: reconstruction by replay')
        a = open_task(alice, root, 'main/1629')
        bob = coordinator.client_for('bob', ':3')
        b = open_task(bob, root, 'main/0240')
        names_a = task_summary(path.join(root, 'main', '1629'))['names']
        names_b = task_summary(path.join(root, 'main', '0240'))['names']
        ia = alice.call('open_interaction', frame=names_a[10], modality='visual', target_object=1,
                        **lease_args(a))['interaction_id']
        ib = bob.call('open_interaction', frame=names_b[10], modality='visual', target_object=1,
                      **lease_args(b))['interaction_id']
        replays = []
        for index in range(10):
            for client, lease_x, interaction in ((alice, a, ia), (bob, b, ib)):
                reply = client.call('click', interaction_id=interaction, x=100 + index, y=100,
                                    is_negative=False, request_id=new_request_id(),
                                    **lease_args(lease_x))
                if reply.get('reconstruction'):
                    replays.append({'kind': reply['reconstruction'],
                                    'seconds': reply['reconstruction_seconds'],
                                    'clicks_replayed': reply['clicks'] - 1})
        scenarios['B2_reconstruction_by_replay'] = replays
        for client, lease_x, interaction, expected in ((alice, a, ia, names_a),
                                                      (bob, b, ib, names_b)):
            before = database_counts(coordinator.state_dir, lease_x['task_id'])
            checked = client.call('replay_interaction', interaction_id=interaction)
            from PIL import Image
            committed = np.array(Image.open(path.join(root, lease_x['task_id'], 'masks',
                                                       expected[10] + '.png')))
            assert np.array_equal(checked['mask'], committed)
            assert database_counts(coordinator.state_dir, lease_x['task_id']) == before
        assert any(r['kind'] == 'replay' and r['clicks_replayed'] >= 9 for r in replays)
        scenarios['B2_replay_verified'] = True
        bob.close()
    finally:
        alice.close()
        coordinator.stop()
    save(args, out)

    if not args.external_pressure:
        scenarios['I2_memory_held_by_another_process'] = {
            'state': 'not_run', 'reason': 'Requires --external-pressure in a maintenance window.'}
        save(args, out)
        return

    log('I2: another process holds the GPU memory')
    coordinator = Coordinator(root, path.join(scratch, 'state-pressure'), label='-pressure')
    alice = coordinator.client_for('alice', ':2')
    ballast_script = path.join(scratch, 'ballast.py')
    with open(ballast_script, 'w') as f:
        f.write(BALLAST)
    try:
        status = coordinator.gpu_status()
        free = status['memory']['free_device']
        # Deliberately leave less free memory than this HD job's admission requirement.
        take = max(0, free - (900 << 20))
        ballast = subprocess.Popen([sys.executable, ballast_script, str(take)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        if not select.select([ballast.stdout], [], [], 30)[0]:
            raise RuntimeError('ballast allocation timed out')
        assert ballast.stdout.readline().strip() == 'allocated', 'ballast allocation failed'
        lease = open_task(alice, root, 'main/1530')
        job = start_job(alice, lease, first_anchor(root, 'main/1530'))
        observed = []
        release_after = 8.0
        begin = time.perf_counter()
        release_times = {}

        def watch(reply):
            observed.append((round(time.perf_counter() - begin, 2), reply['state'], reply['next']))
            if ballast.poll() is None and time.perf_counter() - begin > release_after:
                release_times['requested_seconds'] = time.perf_counter() - begin
                ballast.stdin.write('\n')
                ballast.stdin.flush()
                ballast.wait(30)
                release_times['completed_seconds'] = time.perf_counter() - begin

        pressured = wait_job(alice, job, on_poll=watch)
        check_job(pressured)
        assert ballast.poll() == 0, 'ballast was not released before the job finished'
        states = [state for _, state, _ in observed]
        first_running = next((t for t, state, _ in observed if state in ('running', 'committing')),
                             None)
        assert 'waiting_memory' in states, states
        assert first_running is not None and first_running >= release_after, first_running
        scenarios['I2_memory_held_by_another_process'] = dict(
            job_summary(pressured), ballast_gib=take / 2**30, released_after=release_after,
            release_times=release_times,
            waited_states=sorted(set(states)), first_running_seconds=first_running)
        out['part3_pressure_worker'] = coordinator.status()['worker']
        out['part3_pressure_status'] = coordinator.gpu_status()
        out['part3_pressure_job_configurations'] = job_configurations(coordinator.state_dir)
        scenarios['I2_memory_held_by_another_process']['observations'] = observed
        baseline_state = path.join(args.scratch, 'part2', 'state')
        if path.isfile(path.join(baseline_state, 'service.db')):
            difference = compare_committed_masks(coordinator.state_dir, lease['task_id'],
                                                 baseline_state, 'main/1530')
            check_masks(difference)
            scenarios['I2_memory_held_by_another_process']['masks_equal_to_H'] = difference
        alice.call('close_task', **lease_args(lease))
    finally:
        if 'ballast' in locals() and ballast.poll() is None:
            ballast.kill()
            ballast.wait(30)
        alice.close()
        coordinator.stop()
    save(args, out)


def save(args, out: dict) -> None:
    os.makedirs(args.output, exist_ok=True)
    temporary = args.result_file + '.tmp'
    with open(temporary, 'w') as f:
        json.dump(out, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, args.result_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--original-repo', required=True)
    parser.add_argument('--original-python', default=None)
    parser.add_argument('--output', default=path.join(REPO, 'docs', 'benchmarks'))
    parser.add_argument('--parts', default='1,2,3')
    parser.add_argument('--worker-cap-gib', type=float,
                        help='Explicit worker admission cap for Part 2; recorded in results')
    parser.add_argument('--external-pressure', action='store_true',
                        help='Maintenance window only: occupy most available GPU memory in I2')
    args = parser.parse_args()
    args.source = path.abspath(args.source)
    args.scratch = path.abspath(args.scratch)
    source_real, scratch_real = path.realpath(args.source), path.realpath(args.scratch)
    if path.commonpath([source_real, scratch_real]) in (source_real, scratch_real):
        parser.error('--scratch and --source must be separate, non-overlapping directories')
    for part in args.parts.split(','):
        if part in ('2', '3') and path.exists(path.join(args.scratch, f'part{part}')):
            parser.error(f'part{part} already exists in --scratch; use a fresh scratch directory')
    args.original_python = args.original_python or path.join(args.original_repo, '.venv', 'bin',
                                                             'python')
    device = gpu_device()
    stamp = datetime.now().strftime('%Y-%m-%d')
    args.result_file = path.join(args.output, f'{stamp}-{device["name"].replace(" ", "-")}.json')
    out = {}
    if path.exists(args.result_file):
        with open(args.result_file) as f:
            out = json.load(f)
    out.setdefault('date', datetime.now().isoformat(timespec='seconds'))
    out.setdefault('device', device)
    out.setdefault('gpu_processes_at_start', gpu_processes())
    out['tasks'] = [f'{s}/{q}' for s, q in TASKS]
    run = {'date': datetime.now().isoformat(timespec='seconds'),
                                       'parts': args.parts, 'device': device,
                                       'gpu_processes_at_start': gpu_processes(),
                                       'worker_cap_gib': args.worker_cap_gib,
                                       'external_pressure': args.external_pressure,
                                       'command': sys.argv,
                                       'commit': subprocess.check_output(
                                           ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                                       'completed_parts': [], 'status': 'running'}
    run['source_sha256'] = {name: hashlib.sha256(open(path.join(REPO, name), 'rb').read()).hexdigest()
                            for name in ('scripts/benchmark_service.py',
                                         'cutieplus/coordinator/service.py',
                                         'cutieplus/coordinator/interactions.py',
                                         'cutieplus/coordinator/jobs.py')}
    out.setdefault('runs', []).append(run)
    for part in args.parts.split(','):
        try:
            {'1': part1, '2': part2, '3': part3}[part](args, out)
        except BaseException as exc:
            run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save(args, out)
            raise
        states = {'1': [('part1-shared', 'state')], '2': [('part2', 'state')],
                  '3': [('part3', 'state-cap'), ('part3', 'state-pressure')]}[part]
        for section, state in states:
            state_dir = path.join(args.scratch, section, state)
            if path.isfile(path.join(state_dir, 'service.db')):
                out.setdefault('worker_profiles', {})[f'{section}/{state}'] = worker_profiles(state_dir)
        run['completed_parts'].append(part)
        save(args, out)
    run['status'] = 'completed'
    save(args, out)
    log(f'results in {args.result_file}')


if __name__ == '__main__':
    main()
