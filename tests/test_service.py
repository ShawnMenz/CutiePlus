"""
The coordinator end to end, with real sockets, SQLite and a spawned worker process (fake engine,
so no GPU is needed): sessions and leases, clicks and propagation through one worker, scheduling
between annotators, cancellation, stale results, idempotent retries, and recovery from a crashed
worker, from memory errors and from a restarted coordinator.
"""
import json
import multiprocessing as mp
import os
import time
from os import path

import numpy as np
import pytest

from cutieplus.client import ClientError
from cutieplus.coordinator.service import AlreadyRunning, Service
from gui.task_state import TaskLock, TaskLocked, lock_holder
from tests.helpers import ellipse_mask, make_task, read_mask
from tests.service_env import ServiceEnv, fake_overrides, run_propagation, wait_for

pytestmark = pytest.mark.slow


@pytest.fixture
def two_tasks(tmp_path):
    first = make_task(str(tmp_path), sequence='0001', frame_count=12, mask_anchors=(0, 6),
                      empty_anchors=(10, ))
    second = make_task(str(tmp_path), sequence='0002', frame_count=12, mask_anchors=(0, ))
    return first, second


@pytest.fixture
def env_factory(tmp_path):
    environments = []

    def make(task_root, **fake):
        settings = fake_overrides(**{k: v for k, v in fake.items() if k != 'settings'})
        settings.update(fake.get('settings', {}))
        env = ServiceEnv(task_root, str(tmp_path / 'state'), settings)
        environments.append(env)
        return env

    yield make
    for env in environments:
        env.close()


def open_task(client, task):
    return client.call('open_task', workspace=task.workspace)


def click(client, reply, frame, x, y, negative=False, request_id=None):
    interaction = client.call('open_interaction', task_id=reply['task_id'],
                              lease_epoch=reply['lease_epoch'], frame=frame, modality='thermal',
                              target_object=1)['interaction_id']
    result = client.call('click', task_id=reply['task_id'], lease_epoch=reply['lease_epoch'],
                         interaction_id=interaction, x=x, y=y, is_negative=negative,
                         request_id=request_id or os.urandom(8).hex())
    return interaction, result


def test_two_windows_share_one_worker(two_tasks, env_factory):
    first, second = two_tasks
    env = env_factory(first.root)
    alice = env.client('alice', ':2')
    bob = env.client('bob', ':3')
    a = open_task(alice, first)
    b = open_task(bob, second)
    with pytest.raises(ClientError) as error:
        open_task(bob, first)
    assert error.value.code == 'locked' and 'alice' in str(error.value)
    _, result_a = click(alice, a, '000004', 40, 40)
    _, result_b = click(bob, b, '000003', 60, 50)
    assert result_a['mask'][40, 40] == 1 and result_b['mask'][50, 60] == 1
    # committed by the coordinator: the legacy files show them, as anchors
    assert np.array_equal(first.mask('000004'), result_a['mask'])
    assert first.anchors()['000004']['by'] == 'alice'
    assert second.anchors()['000003']['by'] == 'bob'
    status = alice.call('status')
    assert status['worker']['generation'] == 1
    assert {s['annotator_name'] for s in status['sessions']} == {'alice', 'bob'}
    assert len(status['leases']) == 2
    assert lock_holder(first.workspace)['display'] == ':2'


def test_propagation_commits_frames_and_keeps_anchors(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root)
    alice = env.client('alice', ':2')
    reply = open_task(alice, first)
    anchor_before = first.mask('000007')
    done = run_propagation(alice, reply['task_id'], reply['lease_epoch'], '000001')
    assert done['state'] == 'finished' and done['stop_reason'] == 'empty_anchor'
    assert done['stopped_at'] == 10
    assert done['all_frames'] == [1, 2, 3, 4, 5, 7, 8, 9]  # 6 is an anchor
    assert np.array_equal(first.mask('000007'), anchor_before)
    # the fake propagation shifts the mask one pixel per frame
    assert np.array_equal(first.mask('000004'), np.roll(first.mask('000001'), 3, axis=1))
    rows = env.service.db.query("SELECT frame FROM edits WHERE op = 'propagate'")
    assert len(rows) == 8
    revision = env.service.tasks.task_row('main/0001')
    assert revision['input_revision'] == 1  # propagation never changes the inputs


def test_clicks_are_served_between_propagation_steps(two_tasks, env_factory):
    first, second = two_tasks
    env = env_factory(first.root, step_seconds=0.15, settings={
        'scheduler': {'max_interactive_burst': 1, 'max_background_wait': 10}})
    alice = env.client('alice', ':2')
    bob = env.client('bob', ':3')
    a = open_task(alice, first)
    b = open_task(bob, second)
    job = alice.call('start_propagation', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                     start_frame='000001', direction='forward',
                     memory={'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4,
                             'max_mem_frames': 9, 'max_num_tokens': 10000})['job_id']
    wait_for(lambda: alice.call('poll_propagation', job_id=job)['next'] >= 1)
    latencies = []
    for index in range(3):
        start = time.perf_counter()
        click(bob, b, f'00000{index + 2}', 50, 40)
        latencies.append(time.perf_counter() - start)
    final = wait_for(lambda: (lambda r: r if r['state'] == 'finished' else None)(
        alice.call('poll_propagation', job_id=job)))
    assert final['next'] == 8
    # a click waits for at most the step that is running, not for the whole propagation
    assert max(latencies) < 1.0, latencies
    history = [kind for kind, *_ in env.service.scheduler.history]
    first_click = history.index('interactive')
    assert 'job' in history[first_click:]  # steps resumed after the clicks


def test_cancel_stops_writing(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, step_seconds=0.1)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    job = alice.call('start_propagation', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                     start_frame='000001', direction='forward',
                     memory={'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4,
                             'max_mem_frames': 9, 'max_num_tokens': 10000})['job_id']
    wait_for(lambda: alice.call('poll_propagation', job_id=job)['next'] >= 2)
    alice.call('cancel_propagation', job_id=job)
    stopped = wait_for(lambda: (lambda r: r if r['state'] in ('cancelled', 'finished') else None)(
        alice.call('poll_propagation', job_id=job)))
    assert stopped['state'] == 'cancelled'
    committed = stopped['next']
    time.sleep(0.5)
    assert alice.call('poll_propagation', job_id=job)['next'] == committed
    assert len(env.service.db.query("SELECT * FROM edits WHERE op = 'propagate'")) == committed
    assert env.service.scheduler.snapshot()['active_job'] is None


def test_results_computed_from_old_inputs_are_rejected(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, step_seconds=0.1)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    job = alice.call('start_propagation', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                     start_frame='000001', direction='forward',
                     memory={'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4,
                             'max_mem_frames': 9, 'max_num_tokens': 10000})['job_id']
    wait_for(lambda: alice.call('poll_propagation', job_id=job)['next'] >= 1)
    # the window refuses edits during a run; a human input changing anyway (here directly in the
    # coordinator) makes the job's later results obsolete
    with pytest.raises(ClientError) as error:
        alice.call('commit_frame', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                   frame='000005', mask=ellipse_mask(4, 12), anchor='edit')
    assert error.value.code == 'busy'
    env.service.tasks.commit_frame(a['task_id'], alice.session_id, a['lease_epoch'], '000009',
                                   mask=ellipse_mask(8, 12), anchor='edit')
    stopped = wait_for(lambda: (lambda r: r if r['state'] in ('cancelled', 'finished') else None)(
        alice.call('poll_propagation', job_id=job)))
    assert stopped['state'] == 'cancelled' and 'stale' in stopped['stop_reason']
    assert first.anchors()['000009']['kind'] == 'mask'  # the human edit stands


def test_a_retried_click_is_applied_once(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    interaction, result = click(alice, a, '000004', 40, 40, request_id='click-1')
    again = alice.call('click', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                       interaction_id=interaction, x=40, y=40, is_negative=False,
                       request_id='click-1')
    assert again['replayed'] and again['clicks'] == 1
    assert np.array_equal(again['mask'], result['mask'])
    assert len(env.service.db.query("SELECT * FROM edits WHERE op = 'click'")) == 1


def test_a_crashed_worker_is_replaced_and_the_run_resumes(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, step_seconds=0.02)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    alice.call('test_worker', command='test_inject', crash_on_ti=5)
    done = run_propagation(alice, a['task_id'], a['lease_epoch'], '000001')
    assert done['state'] == 'finished' and done['attempt'] == 2
    assert done['all_frames'] == [1, 2, 3, 4, 5, 7, 8, 9]
    # replayed frames were not written again
    rows = env.service.db.query("SELECT frame, attempt_id FROM edits WHERE op = 'propagate'")
    assert sorted(row['frame'] for row in rows) == ['000002', '000003', '000004', '000005',
                                                     '000006', '000008', '000009', '000010']
    job = env.service.jobs.jobs[done['job_id']]
    assert job.metrics['replayed_frames'] == 4 and job.replay_mismatches == 0
    assert env.service.worker.generation == 2
    assert env.service.metrics.values('worker_recovery')
    # the masks equal those of an undisturbed run (the fake propagation is deterministic)
    assert np.array_equal(first.mask('000006'), np.roll(first.mask('000001'), 5, axis=1))
    # a click after the restart rebuilds its interaction from the record
    _, result = click(alice, a, '000003', 30, 30)
    assert result['mask'][30, 30] == 1


def test_out_of_memory_waits_for_changed_conditions(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, step_seconds=0.01, others_gib=2.0)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    alice.call('test_worker', command='test_inject', oom_on_ti=4)
    job = alice.call('start_propagation', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                     start_frame='000001', direction='forward',
                     memory={'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4,
                             'max_mem_frames': 9, 'max_num_tokens': 10000})['job_id']
    wait_for(lambda: (lambda r: r if r['state'] == 'waiting_memory' and r['next'] == 3 else None)(
        alice.call('poll_propagation', job_id=job)))
    time.sleep(2.5)
    # the same work under the same conditions is not retried
    assert alice.call('poll_propagation', job_id=job)['next'] == 3
    alice.call('test_worker', command='test_set_memory', others_gib=0.0)
    done = wait_for(lambda: (lambda r: r if r['state'] == 'finished' else None)(
        alice.call('poll_propagation', job_id=job)), timeout=30)
    assert done['next'] == 8
    record = env.service.jobs.jobs[job]
    assert record.oom_retries == 1 and record.attempt == 2


def test_an_expired_session_loses_its_task(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, settings={'session_timeout_seconds': 1.5})
    alice = env.client('alice', ':2', heartbeat=False)
    bob = env.client('bob', ':3')
    a = open_task(alice, first)
    wait_for(lambda: not env.service.sessions.is_active(alice.session_id), timeout=10)
    # an expired session can commit nothing; its lease and file lock are released right after
    wait_for(lambda: lock_holder(first.workspace) is None, timeout=5)
    b = open_task(bob, first)
    assert b['lease_epoch'] == a['lease_epoch'] + 1
    with pytest.raises(ClientError) as error:
        alice.call('commit_frame', task_id=a['task_id'], lease_epoch=a['lease_epoch'],
                   frame='000005', mask=ellipse_mask(4, 12), anchor='edit')
    assert error.value.code == 'lease_lost'  # a new session, without the old lease


def _sleep_forever():
    time.sleep(600)


def test_a_restarted_coordinator_recovers(two_tasks, env_factory, tmp_path):
    first, _ = two_tasks
    env = env_factory(first.root)
    with pytest.raises(AlreadyRunning):
        Service(env.cfg, first.root, env.state_dir, env.settings)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    click(alice, a, '000004', 40, 40)
    revision = env.service.tasks.task_row('main/0001')['content_revision']
    env.stop_service()
    # what a crash leaves: a worker of the old coordinator, an object without a commit
    os.environ['CUTIEPLUS_WORKER'] = '1'
    stale = mp.get_context('spawn').Process(target=_sleep_forever, daemon=True)
    stale.start()
    os.environ.pop('CUTIEPLUS_WORKER')
    from cutieplus.storage.db import Database
    from cutieplus.storage.objects import ObjectStore
    db = Database(path.join(env.state_dir, 'service.db'))
    with db.transaction() as conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('worker_pid', ?)", (str(stale.pid), ))
    db.close()
    orphan = ObjectStore(path.join(env.state_dir, 'objects')).put(b'not committed')
    env.start()
    report = env.service.recovery_report
    assert report['stale_worker_stopped'] is True
    assert report['orphan_objects_removed'] == 1
    assert not env.service.objects.exists(orphan)
    stale.join(10)
    assert not stale.is_alive()
    # the window reconnects with its session and takes its task again (nothing changed)
    again = alice.call('open_task', workspace=first.workspace)
    assert again['content_revision'] == revision and again['lease_epoch'] > a['lease_epoch']
    assert again['sync'] == 'current'
    _, result = click(alice, again, '000005', 50, 50)
    assert result['content_revision'] == revision + 1


def test_concurrent_opens_of_one_task_give_one_lease(two_tasks, env_factory):
    import threading
    first, _ = two_tasks
    env = env_factory(first.root)
    clients = [env.client(f'annotator{i}', f':{i + 2}') for i in range(4)]
    outcomes = []

    def try_open(client):
        try:
            outcomes.append(('ok', open_task(client, first)['lease_epoch']))
        except ClientError as exc:
            outcomes.append((exc.code, None))

    threads = [threading.Thread(target=try_open, args=(client, )) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(code for code, _ in outcomes) == ['locked', 'locked', 'locked', 'ok']
    assert len(env.service.tasks.leases) == 1


def test_clicks_use_the_shown_modality_and_propagation_the_anchor_modality(tmp_path, env_factory):
    visual = make_task(str(tmp_path / 'a'), sequence='0001', frame_count=6, mask_anchors=(0, ))
    thermal = make_task(str(tmp_path / 'a'), sequence='0002', frame_count=6, mask_anchors=(0, ),
                        anchor_modality='thermal')
    env = env_factory(visual.root)
    alice = env.client('alice', ':2')
    for task, anchor_modality in ((visual, 'visual'), (thermal, 'thermal')):
        reply = open_task(alice, task)
        interaction, _ = click(alice, reply, '000002', 40, 40)  # clicked on the thermal view
        record = env.service.interactions.interactions[interaction]
        assert f'{path.sep}thermal{path.sep}' in record.image_path
        done = run_propagation(alice, reply['task_id'], reply['lease_epoch'], '000001')
        snapshot = env.service.jobs.jobs[done['job_id']].snapshot
        assert snapshot['anchor_modality'] == anchor_modality
        assert all(f'{path.sep}{anchor_modality}{path.sep}' in p for p in snapshot['frame_paths'])
        alice.call('close_task', task_id=reply['task_id'], lease_epoch=reply['lease_epoch'])


def test_memory_pressure_postpones_a_step_without_losing_the_run(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root, step_seconds=0.01)
    alice = env.client('alice', ':2')
    a = open_task(alice, first)
    alice.call('test_worker', command='test_inject', pressure_on_ti=3, pressure_times=3)
    done = run_propagation(alice, a['task_id'], a['lease_epoch'], '000001')
    assert done['state'] == 'finished' and done['attempt'] == 1  # the context was kept
    assert done['all_frames'] == [1, 2, 3, 4, 5, 7, 8, 9]
    assert any('memory is short' in message for message in done['all_messages'])


def test_a_worker_that_dies_while_idle_is_replaced(two_tasks, env_factory):
    first, _ = two_tasks
    env = env_factory(first.root)
    alice = env.client('alice', ':2')
    old_pid = env.service.worker.pid
    os.kill(old_pid, 9)
    wait_for(lambda: env.service.worker.generation == 2 and env.service.worker.alive(), timeout=60)
    assert env.service.worker.pid != old_pid
    reply = open_task(alice, first)
    _, result = click(alice, reply, '000004', 40, 40)
    assert result['clicks'] == 1
