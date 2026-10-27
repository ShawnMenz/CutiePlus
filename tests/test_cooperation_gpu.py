"""
The one GPU queue (cutieplus/cooperation/gpu.py, worker_link.py, cutieplus/gpu_worker) with a real
worker process running the fake engine: clicks and propagations, one GPU command at a time in
every case, FIFO propagations with clicks between their steps, duplicate submissions, cancellation
that waits for the worker, a crashed worker's interrupted job and its retry, out of memory, the GPU
lock, and who may see or cancel which job.
"""
import hashlib
import io
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image

from cutieplus.cooperation.client import ClientError
from cutieplus.gpu_worker.lock import GpuBusy, acquire_gpu_lock
from tests.gpu_harness import REPO, GpuHarness, no_overlap

pytestmark = pytest.mark.slow


@pytest.fixture
def gpu(tmp_path):
    harness = GpuHarness(tmp_path)
    yield harness
    harness.close()


def users(gpu):
    owner = gpu.owner()
    return owner, gpu.add(owner, 'alice'), gpu.add(owner, 'bob')


def test_a_click_runs_on_the_worker_and_is_recorded(gpu):
    gpu.start_worker()
    owner, alice, _ = users(gpu)
    opened = gpu.call('open_interaction', alice['token'], workspace=gpu.task,
                      image_path=gpu.frame_paths[0])
    reply = gpu.call('click', alice['token'], interaction_id=opened['interaction_id'], x=10,
                     y=8, is_negative=False, idempotency_key='click-0001')
    assert reply['mask'][8, 10] == 1 and reply['mask'].sum() > 50 and reply['clicks'] == 1
    assert reply['prob'].shape == (2, 24, 32)
    undone = gpu.call('undo_click', alice['token'], interaction_id=opened['interaction_id'],
                      idempotency_key='undo-0001')
    assert undone['mask'].sum() == 0
    rows = gpu.db.query("SELECT operation_type, status FROM gpu_jobs ORDER BY queue_sequence")
    assert rows == [{'operation_type': 'click', 'status': 'succeeded'},
                    {'operation_type': 'click_undo', 'status': 'succeeded'}]
    # an interaction is its session's
    other = gpu.login('alice')
    assert gpu.code('click', other, interaction_id=opened['interaction_id'], x=1, y=1,
                    is_negative=False, idempotency_key='click-0002') == 'forbidden'
    # a frame outside the workspace is refused before the worker sees it
    assert gpu.code('open_interaction', alice['token'], workspace=gpu.task,
                    image_path='/etc/passwd') == 'invalid'


def test_a_propagation_stages_its_frames_and_records_them(gpu):
    gpu.start_worker()
    _, alice, _ = users(gpu)
    job = gpu.submit(alice['token'], 'propagate-001', revision=7)
    assert job['status'] == 'queued'
    done = gpu.wait_job(alice['token'], job['id'])
    assert done['status'] == 'succeeded' and done['result_state'] == 'pending'
    result = gpu.call('job_result', alice['token'], job_id=job['id'])
    assert [frame['name'] for frame in result['frames']] == gpu.frame_names[1:]
    assert result['revision'] == 7
    for frame in result['frames']:
        staged = os.path.join(result['staging'], f'{frame["name"]}.png')
        data = open(staged, 'rb').read()
        assert hashlib.sha256(data).hexdigest() == frame['sha256']
        mask = np.array(Image.open(io.BytesIO(data)))
        assert mask[6, 10] == 1 and mask[0, 0] == 0  # the start frame's mask, carried on
    # the snapshot is the service's, read-only, and named by its content
    row = gpu.db.query_one('SELECT * FROM gpu_jobs WHERE id = ?', (job['id'], ))
    assert os.stat(row['input_reference']).st_mode & 0o777 == 0o400
    assert hashlib.sha256(open(row['input_reference'], 'rb').read()).hexdigest() == \
        row['input_revision']


def test_a_repeated_submission_is_the_same_job(gpu):
    _, alice, bob = users(gpu)
    first = gpu.submit(alice['token'], 'same-key-01')
    again = gpu.submit(alice['token'], 'same-key-01')
    assert first['id'] == again['id']
    assert gpu.db.query_one("SELECT count(*) AS n FROM gpu_jobs")['n'] == 1
    # the key is the caller's own
    assert gpu.submit(bob['token'], 'same-key-01')['id'] != first['id']


def test_two_users_never_share_the_gpu_and_clicks_run_between_steps(tmp_path):
    gpu = GpuHarness(tmp_path, frames=30, step_seconds=0.03)
    try:
        gpu.start_worker()
        _, alice, bob = users(gpu)
        jobs = {}

        def submit(user, key):
            jobs[key] = gpu.submit(user['token'], key)

        threads = [threading.Thread(target=submit, args=(user, key))
                   for user, key in ((alice, 'alice-job-1'), (bob, 'bob-job-01'))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        opened = gpu.call('open_interaction', bob['token'], workspace=gpu.task,
                          image_path=gpu.frame_paths[3])
        gpu.wait_for(lambda: any(entry['op'] == 'step' for entry in gpu.log()))
        started = time.monotonic()
        gpu.call('click', bob['token'], interaction_id=opened['interaction_id'], x=5, y=5,
                 is_negative=False, idempotency_key='bob-click-1')
        clicked = time.monotonic() - started
        for key in jobs:
            assert gpu.wait_job(alice['token'] if 'alice' in key else bob['token'],
                                jobs[key]['id'])['status'] == 'succeeded'
        log = gpu.log()
        assert no_overlap(log)  # one GPU command at a time, whoever sent it
        steps = {key: [entry for entry in log if entry['job'] == job['id']]
                 for key, job in jobs.items()}
        first, second = sorted(steps.values(), key=lambda entries: entries[0]['start'])
        assert first[-1]['end'] <= second[0]['start']  # FIFO: one run after the other
        click = next(entry for entry in log if entry['op'] == 'click')
        assert first[0]['start'] < click['start'] < first[-1]['end'] or \
            second[0]['start'] < click['start'] < second[-1]['end']
        assert clicked < 2.0  # it did not wait for a run to end (each takes about a second)
    finally:
        gpu.close()


def test_cancelling_waits_for_the_worker_and_keeps_what_was_done(tmp_path):
    gpu = GpuHarness(tmp_path, frames=60, step_seconds=0.05)
    try:
        gpu.start_worker()
        _, alice, _ = users(gpu)
        running = gpu.submit(alice['token'], 'long-run-01')
        queued = gpu.submit(alice['token'], 'long-run-02')
        assert gpu.call('cancel_job', alice['token'], job_id=queued['id'])['status'] == \
            'cancelled'
        gpu.wait_for(lambda: gpu.job(alice['token'], running['id'])['status'] == 'running')
        gpu.wait_for(lambda: len(gpu.log()) >= 5)
        assert gpu.call('cancel_job', alice['token'],
                        job_id=running['id'])['status'] == 'cancel_requested'
        done = gpu.wait_job(alice['token'], running['id'])
        assert done['status'] == 'cancelled' and done['result_state'] == 'pending'
        result = gpu.call('job_result', alice['token'], job_id=running['id'])
        assert 0 < len(result['frames']) < 59
        finished = gpu.db.query_one('SELECT finished_at FROM gpu_jobs WHERE id = ?',
                                    (running['id'], ))
        assert finished['finished_at'] is not None
        assert len([entry for entry in gpu.log() if entry['job'] == running['id']]) == \
            len(result['frames']) + 1  # the start frame's step, then one per staged frame
    finally:
        gpu.close()


def test_a_crashed_worker_leaves_an_interrupted_job_and_retry_makes_a_new_one(tmp_path):
    gpu = GpuHarness(tmp_path, frames=20, step_seconds=0.01)
    try:
        crashing = gpu.start_worker(crash_at_step=6)
        _, alice, _ = users(gpu)
        job = gpu.submit(alice['token'], 'crash-job-1')
        done = gpu.wait_job(alice['token'], job['id'])
        assert done['status'] == 'interrupted' and crashing.wait(10) == 3
        result = gpu.call('job_result', alice['token'], job_id=job['id'])
        assert len(result['frames']) == 4  # the steps that completed
        assert gpu.service.gpu.link.ready is False
        state = gpu.db.query_one("SELECT * FROM gpu_state")
        assert state['state'] in ('offline', 'recovering') and state['active_job_id'] is None
        generation = gpu.service.gpu.link.generation
        gpu.start_worker()
        assert gpu.service.gpu.link.generation == generation + 1
        retried = gpu.call('retry_job', alice['token'], job_id=job['id'],
                           idempotency_key='crash-retry')
        assert retried['retry_of'] == job['id']
        assert gpu.wait_job(alice['token'], retried['id'])['status'] == 'succeeded'
        # the old token cannot record the new result
        assert gpu.code('record_result', alice['token'], job_id=retried['id'],
                        execution_token=result['token'], outcome='applied') == 'conflict'
    finally:
        gpu.close()


def test_out_of_memory_fails_the_job_and_the_queue_moves_on(tmp_path):
    gpu = GpuHarness(tmp_path, frames=8)
    try:
        gpu.start_worker(oom_at_step=3)
        _, alice, _ = users(gpu)
        failing = gpu.submit(alice['token'], 'oom-job-01')
        after = gpu.submit(alice['token'], 'oom-job-02')
        done = gpu.wait_job(alice['token'], failing['id'])
        assert done['status'] == 'failed' and done['error_code'] == 'oom'
        assert gpu.wait_job(alice['token'], after['id'])['status'] == 'succeeded'
        assert gpu.service.gpu.link.ready  # the same worker, still serving
    finally:
        gpu.close()


def test_the_gpu_lock_admits_one_worker_and_outlives_a_crashed_parent(gpu, tmp_path):
    held = acquire_gpu_lock(str(gpu.lock_file))
    try:
        process = gpu.start_worker(wait=False)
        assert process.wait(30) == 75  # EX_TEMPFAIL: another holds the GPU
        assert b'GPU lock' in process.stderr.read()
    finally:
        held.close()
    # a child that inherited the lock keeps it after its parent died (/proc/locks still names
    # the parent, which took it): no new worker can start until the child is gone too
    child_file = tmp_path / 'child.pid'
    script = ('import os, sys, time, fcntl\n'
              f'f = open({str(gpu.lock_file)!r}, "rb"); fcntl.flock(f, fcntl.LOCK_EX)\n'
              'if os.fork() == 0:\n    os.close(1); os.close(2)\n'
              f'    open({str(child_file)!r}, "w").write(str(os.getpid())); time.sleep(30)\n'
              'else:\n    os._exit(0)\n')
    parent = subprocess.run([sys.executable, '-c', script], capture_output=True, timeout=10)
    assert parent.returncode == 0
    gpu.wait_for(lambda: child_file.exists() and child_file.read_text())
    with pytest.raises(GpuBusy):
        acquire_gpu_lock(str(gpu.lock_file))
    os.kill(int(child_file.read_text()), 9)
    gpu.wait_for(lambda: try_lock(str(gpu.lock_file)))


def try_lock(lock_path: str) -> bool:
    try:
        acquire_gpu_lock(lock_path).close()
        return True
    except GpuBusy:
        return False


def test_a_second_worker_is_refused_while_the_first_runs(gpu):
    gpu.start_worker()
    generation = gpu.service.gpu.link.generation
    # a second worker gets the lock only if the first released it: simulate a stray one that
    # skips the lock by using its own lock file
    other_lock = gpu.tmp_path / 'other.lock'
    other_lock.write_text('')
    config = gpu.tmp_path / 'other.toml'
    config.write_text(gpu.config_file.read_text().replace(str(gpu.lock_file), str(other_lock)))
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    stray = subprocess.Popen([sys.executable, '-m', 'cutieplus.gpu_worker', '--engine', 'fake',
                              '--config', str(config)], cwd=REPO, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        time.sleep(3)
        assert gpu.service.gpu.link.generation == generation  # not taken over
        refused = gpu.db.query("SELECT * FROM audit_events WHERE action = "
                               "'gpu_worker_connected'")
        assert len(refused) == 1
    finally:
        stray.kill()
        stray.wait(10)


def test_each_role_sees_and_cancels_whose_jobs_it_may(gpu):
    owner, alice, bob = users(gpu)
    sam = gpu.add(owner, 'sam', 'supervisor')
    alices = gpu.submit(alice['token'], 'alice-job-9')
    sams = gpu.submit(sam['token'], 'sam-job-009')
    assert gpu.code('cancel_job', bob['token'], job_id=alices['id']) == 'forbidden'
    assert gpu.code('job_events', bob['token'], job_id=alices['id']) == 'forbidden'
    seen = gpu.call('gpu_status', bob['token'])['queue']
    assert [job['owner'] for job in seen] == [None, None]  # someone's jobs, not whose
    assert [job['owner'] for job in gpu.call('gpu_status', sam['token'])['queue']] == [
        'alice', 'sam']
    assert gpu.code('cancel_job', sam['token'], job_id=alices['id']) == 'ok'
    assert gpu.code('cancel_job', alice['token'], job_id=sams['id']) == 'forbidden'
    assert gpu.code('cancel_job', owner, job_id=sams['id']) == 'ok'
    # no role goes first: the queue is in submission order
    first = gpu.submit(alice['token'], 'alice-job-a')
    second = gpu.submit(owner, 'owner-job-a')
    assert gpu.job(owner, second['id'])['position'] > gpu.job(alice['token'],
                                                              first['id'])['position']


def test_a_user_may_queue_a_few_jobs_and_a_suspension_cancels_them(gpu):
    owner, alice, _ = users(gpu)
    limit = gpu.service.setting('pending_jobs_per_user')
    jobs = [gpu.submit(alice['token'], f'limit-job-{index}') for index in range(limit)]
    with pytest.raises(ClientError) as refused:
        gpu.submit(alice['token'], 'limit-job-x')
    assert refused.value.code == 'limit'
    gpu.call('suspend_user', owner, user_id=alice['id'])
    statuses = gpu.db.query("SELECT status, error_code FROM gpu_jobs WHERE owner_user_id = ?",
                            (alice['id'], ))
    assert {row['status'] for row in statuses} == {'cancelled'} and len(jobs) == limit
    assert {row['error_code'] for row in statuses} == {'owner_inactive'}


def test_a_snapshot_is_checked_before_it_is_queued(gpu):
    _, alice, _ = users(gpu)
    bad = [dict(frame_paths=['/etc/passwd'] * 12), dict(direction='sideways'),
           dict(start_ti=40), dict(anchors={'000002': 'maybe'}),
           dict(seed_prob=np.zeros((2, 3, 3), np.float16)), dict(seed_prob=[[0.5]]),
           dict(anchor_masks={'000002': b'png'})]
    for index, change in enumerate(bad):
        snapshot = dict(gpu.snapshot(), **change)
        assert gpu.code('submit_propagation', alice['token'], idempotency_key=f'bad-snap-{index}',
                        workspace=gpu.task, revision=1, snapshot=snapshot) == 'invalid', change
    assert gpu.code('submit_propagation', alice['token'], idempotency_key='bad-workspace',
                    workspace='/tmp/elsewhere', revision=1, snapshot=gpu.snapshot()) == 'invalid'
