"""Scheduling of GPU work (pure logic, fake clock) and the socket message format."""
import socket

import numpy as np
import pytest

from cutieplus.coordinator.scheduler import QueueFull, Scheduler, SchedulerConfig
from cutieplus.protocol import receive_message, send_message


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make(**config):
    clock = Clock()
    return Scheduler(SchedulerConfig(**config), clock), clock


def take(scheduler):
    decision = scheduler.next(timeout=0)
    if decision is None:
        return None
    if decision.kind == 'interactive':
        return ('click', decision.item.session_id, decision.item.label)
    return ('job', decision.job_id, 'admission' if decision.admission else 'step')


def test_interactive_requests_go_first_and_round_robin_between_sessions():
    scheduler, _ = make(max_interactive_burst=10, max_background_wait=10)
    scheduler.add_job('A', 'sa')
    assert take(scheduler) == ('job', 'A', 'admission')
    scheduler.job_ready('A')
    for label in ('b1', 'b2', 'b3'):
        scheduler.submit('sb', lambda: None, label)
    for label in ('c1', 'c2'):
        scheduler.submit('sc', lambda: None, label)
    order = [take(scheduler) for _ in range(5)]
    assert order == [('click', 'sb', 'b1'), ('click', 'sc', 'c1'), ('click', 'sb', 'b2'),
                     ('click', 'sc', 'c2'), ('click', 'sb', 'b3')]
    assert take(scheduler) == ('job', 'A', 'step')


def test_required_pattern_a100_b_a101_a102_c_a103():
    """A propagation step after every burst of clicks, clicks between steps."""
    scheduler, clock = make(max_interactive_burst=1, max_background_wait=10)
    scheduler.add_job('A', 'sa')
    assert take(scheduler) == ('job', 'A', 'admission')
    sequence = []
    scheduler.job_ready('A')
    sequence.append(take(scheduler))  # A: frame 100
    scheduler.submit('sb', lambda: None, 'B click')
    scheduler.job_ready('A')
    sequence.append(take(scheduler))  # B's click first
    sequence.append(take(scheduler))  # A: frame 101
    scheduler.job_ready('A')
    sequence.append(take(scheduler))  # A: frame 102
    scheduler.submit('sc', lambda: None, 'C click')
    scheduler.job_ready('A')
    sequence.append(take(scheduler))
    sequence.append(take(scheduler))  # A: frame 103
    assert sequence == [('job', 'A', 'step'), ('click', 'sb', 'B click'), ('job', 'A', 'step'),
                        ('job', 'A', 'step'), ('click', 'sc', 'C click'), ('job', 'A', 'step')]


def test_continuous_clicks_cannot_starve_propagation():
    scheduler, clock = make(max_interactive_burst=3, max_background_wait=10)
    scheduler.add_job('A', 'sa')
    take(scheduler)
    scheduler.job_ready('A')
    for index in range(20):
        scheduler.submit(f's{index % 2}', lambda: None, str(index))
    kinds = [take(scheduler)[0] for _ in range(8)]
    assert kinds == ['click', 'click', 'click', 'job', 'click', 'click', 'click', 'click']


def test_a_step_that_waited_too_long_goes_first():
    scheduler, clock = make(max_interactive_burst=100, max_background_wait=0.25)
    scheduler.add_job('A', 'sa')
    take(scheduler)
    scheduler.job_ready('A')
    scheduler.submit('sb', lambda: None, 'b1')
    scheduler.submit('sb', lambda: None, 'b2')
    assert take(scheduler)[0] == 'click'
    clock.now += 0.3
    assert take(scheduler) == ('job', 'A', 'step')


def test_one_context_and_queue_positions():
    scheduler, clock = make()
    for job in ('A', 'B', 'C'):
        scheduler.add_job(job, 's' + job)
    assert take(scheduler) == ('job', 'A', 'admission')
    assert scheduler.queue_position('A') is None
    assert scheduler.queue_position('B') == 1 and scheduler.queue_position('C') == 2
    # A holds the context but is not ready (committing): B and C may not start
    scheduler.job_blocked('A')
    assert take(scheduler) is None
    scheduler.remove_job('A')
    assert take(scheduler) == ('job', 'B', 'admission')
    # B waits for memory: it gives up the context; C may try meanwhile, B again later
    scheduler.requeue('B', clock.now + 2)
    assert take(scheduler) == ('job', 'C', 'admission')
    scheduler.requeue('C', clock.now + 5)
    assert take(scheduler) is None
    clock.now += 2.5
    assert take(scheduler) == ('job', 'B', 'admission')


def test_bounded_queues_and_dropped_sessions():
    scheduler, _ = make(max_interactive_queue=2, max_jobs=1)
    scheduler.submit('s', lambda: None)
    scheduler.submit('s', lambda: None)
    with pytest.raises(QueueFull):
        scheduler.submit('s', lambda: None)
    scheduler.add_job('A', 's')
    with pytest.raises(QueueFull):
        scheduler.add_job('B', 's')
    assert len(scheduler.drop_session('s')) == 2
    assert take(scheduler) == ('job', 'A', 'admission')


def test_messages_round_trip_arrays_and_bytes():
    left, right = socket.socketpair()
    mask = np.arange(12, dtype=np.uint8).reshape(3, 4)
    prob = np.random.default_rng(0).random((2, 5, 6)).astype(np.float32)
    message = {'op': 'x', 'args': {'mask': mask, 'nested': [prob, {'png': b'\x89PNG'}],
                                   'n': np.int64(7), 'text': 'ü'}}
    send_message(left, message)
    received = receive_message(right)
    assert np.array_equal(received['args']['mask'], mask)
    assert received['args']['mask'].dtype == np.uint8
    assert np.array_equal(received['args']['nested'][0], prob)
    assert received['args']['nested'][1]['png'] == b'\x89PNG'
    assert received['args']['n'] == 7 and received['args']['text'] == 'ü'
    left.close()
    with pytest.raises(ConnectionError):
        receive_message(right)
