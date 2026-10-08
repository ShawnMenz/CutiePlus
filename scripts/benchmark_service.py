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
import signal
import subprocess
import sys
import threading
import time
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
                                                      'queue_seconds', 'inference_seconds',
                                                      'gpu_request_seconds', 'commit_seconds')})
            done += 1
            if think:
                time.sleep(think)
            if done >= count:
                break
        client.call('close_interaction', interaction_id=interaction)
        if done >= count:
            break
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
            reply['wall_seconds'] = time.perf_counter() - begin
            return reply
        time.sleep(0.05)
    raise RuntimeError('job did not finish')


def job_summary(reply):
    metrics = reply['metrics']
    steps = metrics['steps'] or 1
    # timed by the coordinator, from admission to the last commit: a client that starts polling
    # late (F waits for its jobs one after the other) does not change it
    run = metrics.get('run_seconds')
    return {'state': reply['state'], 'stop_reason': reply['stop_reason'], 'frames': reply['next'],
            'attempts': reply['attempt'], 'run_seconds': run,
            'elapsed_seconds': metrics.get('elapsed_seconds'),
            'fps': reply['next'] / run if run else None,
            'queue_seconds': metrics['queue_seconds'],
            'inference_seconds': metrics['inference_seconds'],
            'inference_per_step': metrics['inference_seconds'] / steps,
            'commit_seconds': metrics['commit_seconds'],
            'commit_per_frame': metrics['commit_seconds'] / max(1, reply['next']),
            'commit_wait_seconds': metrics['commit_wait_seconds'],
            'replayed_frames': metrics['replayed_frames'],
            'replay_mismatches': metrics['replay_mismatches'],
            'oom_retries': metrics['oom_retries'], 'worker_restarts': metrics['worker_restarts'],
            'effective': [{k: e.get(k) for k in ('attempt_id', 'permanent_tis', 'limited_by',
                                                  'estimate_bytes', 'query_tokens',
                                                  'memory_tokens', 'quality_anchor_limit',
                                                  'memory_anchor_limit')}
                          for e in reply['effective']],
            'messages': reply['all_messages']}


def first_anchor(root, task_id):
    split, sequence = task_id.split('/')
    summary = task_summary(path.join(root, split, sequence))
    return summary['names'][summary['first_mask_anchor']]


def part2(args, out: dict) -> None:
    scratch = path.join(args.scratch, 'part2')
    tasks = [(s, q, s) for s, q in TASKS] + [('main', '0240', 'copyc'), ('main', '0240', 'copyd'),
                                            ('main', '0240', 'copyj'), ('main', '1530', 'copyi')]
    root = path.join(scratch, 'tasks')
    prepare(args.source, tasks, root)
    scenarios = out.setdefault('scenarios', {})
    coordinator = Coordinator(root, path.join(scratch, 'state'), label='-part2')
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
        replay_check = alice.call('replay_interaction', interaction_id=ia)
        scenarios['B_two_annotators_alternate'] = {
            'alice': stats(alternated['alice'][1:]), 'bob': stats(alternated['bob'][1:]),
            'reconstruction': stats([seconds for kind, seconds, _ in rebuild]),
            'reconstruction_kinds': sorted({kind for kind, _, _ in rebuild}),
            'replay_seconds_with_clicks': {'clicks': replay_check['clicks'],
                                           'seconds': replay_check['reconstruction_seconds']}}
        alice.call('close_interaction', interaction_id=ia)
        bob.call('close_interaction', interaction_id=ib)
        save(args, out)

        # C: one propagation alone (0240 copy C, 1920x1080, 27 anchors)
        log('C: one propagation')
        c_lease = open_task(alice, root, 'copyc/0240')
        alone = wait_job(alice, start_job(alice, c_lease, first_anchor(root, 'copyc/0240')))
        scenarios['C_one_propagation'] = job_summary(alone)
        scenarios['C_one_propagation']['worker'] = coordinator.gpu_status()
        alice.call('close_task', **lease_args(c_lease))
        save(args, out)

        # D: the same propagation while another annotator clicks
        log('D: propagation + one annotator clicking')
        d_lease = open_task(alice, root, 'copyd/0240')
        job = start_job(alice, d_lease, first_anchor(root, 'copyd/0240'))
        results = {}
        clicker = threading.Thread(target=clicks_on, args=(bob, b0240, root, 'main/0240', 30, 0.3,
                                                           results, 'bob'))
        clicker.start()
        during = wait_job(alice, job)
        clicker.join()
        scenarios['D_propagation_and_clicks'] = {
            'propagation': job_summary(during),
            'click_latency': stats(results['bob']['latencies']),
            'click_queue': stats([d['queue_seconds'] for d in results['bob']['details']]),
            'click_inference': stats([d['inference_seconds'] for d in results['bob']['details']]),
            'masks_equal_to_C': mask_differences(path.join(root, 'copyd', '0240'),
                                                 path.join(root, 'copyc', '0240'))}
        alice.call('close_task', **lease_args(d_lease))
        save(args, out)

        # E: a propagation (small target, 960 px) while three annotators click
        log('E: propagation + three annotators clicking')
        e_lease = open_task(alice, root, 'main/0468')
        job = start_job(alice, e_lease, first_anchor(root, 'main/0468'))
        results = {}
        threads = [threading.Thread(target=clicks_on, args=(client, lease, root, task_id, 20, 0.3,
                                                            results, name))
                   for name, client, lease, task_id in (('bob', bob, b0240, 'main/0240'),
                                                        ('carol', carol, c2363, 'main/2363'),
                                                        ('dave', dave, d0327, 'main/0327'))]
        for thread in threads:
            thread.start()
        during = wait_job(alice, job)
        for thread in threads:
            thread.join()
        scenarios['E_propagation_and_three_annotators'] = {
            'propagation': job_summary(during),
            'click_latency': {name: stats(r['latencies']) for name, r in results.items()},
            'click_queue': {name: stats([d['queue_seconds'] for d in r['details']])
                            for name, r in results.items()},
            'all_clicks': stats([v for r in results.values() for v in r['latencies']])}
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
        done = {name: wait_job(clients[name], job) for name, job in jobs.items()}
        sampler_stop.set()
        sampler.join()
        scenarios['F_queued_propagations'] = {
            name: job_summary(reply) for name, reply in done.items()}
        scenarios['F_queued_propagations']['max_running_contexts'] = max(contexts or [0])
        save(args, out)

        # H: small-target policy on an HD sequence (1530: 960 px, upscaled)
        log('H: small target, HD')
        h_lease = open_task(dave, root, 'main/1530')
        small = wait_job(dave, start_job(dave, h_lease, first_anchor(root, 'main/1530')))
        scenarios['H_small_target_hd'] = dict(job_summary(small), resolution=h_lease['resolution'])
        dave.call('close_task', **lease_args(h_lease))
        save(args, out)

        # J: the GPU worker dies during a propagation
        log('J: worker failure')
        j_lease = open_task(alice, root, 'copyj/0240')
        job = start_job(alice, j_lease, first_anchor(root, 'copyj/0240'))
        killed = {}

        def kill_when(reply):
            if 'at' not in killed and reply['next'] >= 60:
                pid = coordinator.status()['worker']['pid']
                os.kill(pid, signal.SIGKILL)
                killed.update(at=time.perf_counter(), attempt=reply['attempt'], pid=pid,
                              frames=reply['next'])
                return
            if 'at' not in killed or 'resumed' in killed:
                return
            # a frame computed before the kill may still be committed after it: recovery is the
            # first frame committed by the new attempt, past those it replays
            if reply['attempt'] > killed['attempt']:
                if 'baseline' not in killed:
                    killed['baseline'] = reply['next']
                elif reply['next'] > killed['baseline']:
                    killed['resumed'] = time.perf_counter() - killed['at']

        recovered = wait_job(alice, job, on_poll=kill_when)
        status = coordinator.status()
        scenarios['J_worker_failure'] = {
            'propagation': job_summary(recovered), 'killed_after_frames': killed.get('frames'),
            'committed_when_new_attempt_started': killed.get('baseline'),
            'seconds_until_new_frames': killed.get('resumed'),
            'worker_recovery_seconds': status['metrics'].get('worker_recovery'),
            'worker_generation': status['worker']['generation'],
            'masks_equal_to_C': mask_differences(path.join(root, 'copyj', '0240'),
                                                 path.join(root, 'copyc', '0240'))}
        alice.call('close_task', **lease_args(j_lease))
        save(args, out)

        # K: writing annotations fails (the object store becomes read-only)
        log('K: write failure')
        objects = path.join(scratch, 'state', 'objects')
        directories = [objects] + [path.join(objects, d) for d in os.listdir(objects)]
        modes = {d: os.stat(d).st_mode for d in directories}
        k_lease = open_task(alice, root, 'main/1629')
        names_k = task_summary(path.join(root, 'main', '1629'))['names']
        # F propagated this task from the same anchor, and a mask the store already holds is not
        # written again: a shifted anchor mask makes the propagation compute masks that are new
        from PIL import Image
        anchor_k = first_anchor(root, 'main/1629')
        seed = np.array(Image.open(path.join(root, 'main', '1629', 'masks', anchor_k + '.png')))
        alice.call('commit_frame', frame=anchor_k, mask=np.roll(seed, 8, axis=1), anchor='edit',
                   request_id=new_request_id(), **lease_args(k_lease))
        interaction = alice.call('open_interaction', frame=names_k[5], modality='visual',
                                 target_object=1, **lease_args(k_lease))['interaction_id']
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
        job = start_job(alice, k_lease, first_anchor(root, 'main/1629'))
        failed = wait_job(alice, job)
        for directory, mode in modes.items():
            os.chmod(directory, mode)
        retried = alice.call('click', interaction_id=interaction, x=50, y=50, is_negative=False,
                             request_id='k-click', **lease_args(k_lease))
        scenarios['K_write_failure'] = {
            'click_error': click_error, 'click_error_seconds': click_error_seconds,
            'propagation': {'state': failed['state'], 'stop_reason': failed['stop_reason'],
                            'error': failed.get('error'), 'frames': failed['next']},
            'retry_after_repair': {'clicks': retried['clicks'],
                                   'content_revision': retried['content_revision']}}
        alice.call('close_interaction', interaction_id=interaction)
        alice.call('close_task', **lease_args(k_lease))
        out['part2_metrics'] = coordinator.status()['metrics']
        out['part2_worker_status'] = coordinator.gpu_status()
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
        scenarios['I1_memory_cap'] = dict(job_summary(capped), cap_gib=2.2,
                                          worker=coordinator.gpu_status())
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
        bob.close()
    finally:
        alice.close()
        coordinator.stop()
    save(args, out)

    log('I2: another process holds the GPU memory')
    coordinator = Coordinator(root, path.join(scratch, 'state-pressure'), label='-pressure')
    alice = coordinator.client_for('alice', ':2')
    ballast_script = path.join(scratch, 'ballast.py')
    with open(ballast_script, 'w') as f:
        f.write(BALLAST)
    try:
        status = coordinator.gpu_status()
        free = status['memory']['free_device']
        # leave the worker less than the 960 px HD job needs (about 2-3 GiB with its anchors)
        take = max(0, free - (900 << 20))
        ballast = subprocess.Popen([sys.executable, ballast_script, str(take)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        ballast.stdout.readline()
        lease = open_task(alice, root, 'main/1530')
        job = start_job(alice, lease, first_anchor(root, 'main/1530'))
        observed = []
        release_after = 8.0
        begin = time.perf_counter()

        def watch(reply):
            observed.append((round(time.perf_counter() - begin, 2), reply['state'], reply['next']))
            if ballast.poll() is None and time.perf_counter() - begin > release_after:
                ballast.stdin.write('\n')
                ballast.stdin.flush()
                ballast.wait(30)

        pressured = wait_job(alice, job, on_poll=watch)
        states = [state for _, state, _ in observed]
        first_running = next((t for t, state, _ in observed if state in ('running', 'committing')),
                             None)
        scenarios['I2_memory_held_by_another_process'] = dict(
            job_summary(pressured), ballast_gib=take / 2**30, released_after=release_after,
            waited_states=list(dict.fromkeys(states)), first_running_seconds=first_running)
        alice.call('close_task', **lease_args(lease))
    finally:
        if 'ballast' in locals() and ballast.poll() is None:
            ballast.kill()
        alice.close()
        coordinator.stop()
    save(args, out)


def save(args, out: dict) -> None:
    os.makedirs(args.output, exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(out, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--original-repo', required=True)
    parser.add_argument('--original-python', default=None)
    parser.add_argument('--output', default=path.join(REPO, 'docs', 'benchmarks'))
    parser.add_argument('--parts', default='1,2,3')
    args = parser.parse_args()
    args.source = path.abspath(args.source)
    args.original_python = args.original_python or path.join(args.original_repo, '.venv', 'bin',
                                                             'python')
    device = gpu_device()
    stamp = datetime.now().strftime('%Y-%m-%d')
    args.result_file = path.join(args.output, f'{stamp}-{device["name"].replace(" ", "-")}.json')
    out = {}
    if path.exists(args.result_file):
        with open(args.result_file) as f:
            out = json.load(f)
    out.update({'date': datetime.now().isoformat(timespec='seconds'), 'device': device,
                'gpu_processes_at_start': gpu_processes(),
                'tasks': [f'{s}/{q}' for s, q in TASKS]})
    for part in args.parts.split(','):
        {'1': part1, '2': part2, '3': part3}[part](args, out)
        save(args, out)
    log(f'results in {args.result_file}')


if __name__ == '__main__':
    main()
