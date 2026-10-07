"""
The engine against the original code (tests/reference_original.py), on the CPU with the real
weights: same inputs, same weights, same settings, and results compared exactly.
"""
import numpy as np
import pytest
import torch

from cutie.inference.inference_core import InferenceCore
from cutieplus.engine import policies
from cutieplus.engine.propagation import PropagationPlan, PropagationRuntime
from cutieplus.engine.ritm import RitmEngine
from gui.interactive_utils import index_numpy_to_one_hot_torch
from tests.conftest import compose_gui_config
from tests.helpers import ellipse_mask, frame_name, render_frames
from tests.reference_original import original_clicks, original_propagation

FRAMES = 14


@pytest.fixture(scope='module')
def sequence():
    visual, thermal = render_frames(FRAMES)
    masks = {ti: ellipse_mask(ti, FRAMES) for ti in range(FRAMES)}
    names = [frame_name(ti) for ti in range(FRAMES)]
    return visual, thermal, masks, names


def one_hot(mask, device='cpu'):
    return index_numpy_to_one_hot_torch(mask, 2).to(device)


CLICKS_A = [(40, 50, False), (60, 50, True), (35, 45, False)]
CLICKS_B = [(70, 48, False), (55, 70, True)]


@pytest.mark.models
def test_clicks_match_the_original(networks, device, sequence):
    visual, _, masks, _ = sequence
    _, click_ctrl = networks
    prev = one_hot(masks[3], device)
    expected = original_clicks(click_ctrl.session(), visual[3], prev, 1, CLICKS_A, device, True,
                               undo_after=3)
    engine = RitmEngine(click_ctrl, device, True)
    engine.open('a', visual[3], prev, 1)
    results = [engine.click('a', *click).prob for click in CLICKS_A]
    results.append(engine.undo('a').prob)
    assert len(results) == len(expected)
    for got, want in zip(results, expected):
        assert torch.equal(got, want)
    assert engine.undo('a').clicks == 1
    assert engine.undo('a') is None  # one click left: the window undoes the whole edit


@pytest.mark.models
@pytest.mark.parametrize('budget', [2 << 30, 0], ids=['snapshot', 'replay'])
def test_interleaved_interactions_keep_their_state(networks, device, sequence, budget):
    """Two annotators click in turn: each interaction is rebuilt (from its CPU snapshot, or by
    replaying its clicks when there is no snapshot) and gives what it gives alone."""
    visual, thermal, masks, _ = sequence
    _, click_ctrl = networks
    prev_a, prev_b = one_hot(masks[3], device), one_hot(np.zeros_like(masks[8]), device)
    alone_a = original_clicks(click_ctrl.session(), visual[3], prev_a, 1, CLICKS_A, device, True,
                              undo_after=3)
    alone_b = original_clicks(click_ctrl.session(), thermal[8], prev_b, 1, CLICKS_B, device, True)

    engine = RitmEngine(click_ctrl, device, True, snapshot_budget_bytes=budget)
    engine.open('a', visual[3], prev_a, 1)
    engine.open('b', thermal[8], prev_b, 1)  # the thermal view of another task
    got_a, got_b, rebuilt = [], [], []
    for click_a, click_b in zip(CLICKS_A, CLICKS_B + [None]):
        outcome = engine.click('a', *click_a)
        got_a.append(outcome.prob)
        rebuilt.append(outcome.reconstruction)
        if click_b is not None:
            outcome = engine.click('b', *click_b)
            got_b.append(outcome.prob)
            rebuilt.append(outcome.reconstruction)
    engine.click('b', 1, 1, True)  # b is active, a is not
    outcome = engine.undo('a')
    got_a.append(outcome.prob)
    rebuilt.append(outcome.reconstruction)

    for got, want in zip(got_a, alone_a):
        assert torch.equal(got, want)
    for got, want in zip(got_b, alone_b):
        assert torch.equal(got, want)
    kind = 'snapshot' if budget else 'replay'
    assert rebuilt == [None, None, kind, kind, kind, kind]
    assert engine.active.interaction_id == 'a'
    assert all(item.live is None for item in engine.interactions.values() if item is not engine.active)


@pytest.mark.models
def test_replay_reproduces_the_live_state(networks, device, sequence):
    visual, _, masks, _ = sequence
    engine = RitmEngine(networks[1], device, True)
    engine.open('a', visual[5], one_hot(masks[5], device), 1)
    for click in CLICKS_A:
        live = engine.click('a', *click)
    replayed = engine.replay('a')
    assert replayed.reconstruction == 'replay'
    assert torch.equal(replayed.prob, live.prob)
    assert np.array_equal(replayed.mask, live.mask)


def task_anchors(names):
    return {names[2]: {'kind': 'mask'}, names[9]: {'kind': 'mask'}, names[12]: {'kind': 'empty'}}


def make_runtime(cutie, cfg, sequence, start, direction, limit, device='cpu'):
    visual, _, masks, names = sequence
    anchors = task_anchors(names)
    anchor_tis = [ti for ti, name in enumerate(names) if anchors.get(name, {}).get('kind') == 'mask']
    plan = PropagationPlan(tuple(names), start, direction,
                           {name: entry['kind'] for name, entry in anchors.items()},
                           tuple(policies.select_permanent_anchors(anchor_tis, start, limit)), True,
                           len(anchor_tis))
    return PropagationRuntime(InferenceCore(cutie, cfg), plan, lambda ti: visual[ti],
                              lambda ti: masks[ti], one_hot(masks[start], device), device, True,
                              cfg)


def run_to_end(runtime, between=None):
    saved, kinds = {}, []
    while not runtime.finished:
        result = runtime.step()
        kinds.append((result.kind, result.ti))
        if result.needs_commit:
            saved[result.ti] = result.mask
        if between is not None:
            between()
    return saved, kinds


@pytest.mark.models
@pytest.mark.parametrize('start,direction,limit', [(4, 'forward', 5), (4, 'forward', 1),
                                                   (11, 'backward', 5)])
def test_runtime_matches_the_original_loop(networks, device, sequence, start, direction, limit):
    cutie, _ = networks
    cfg = compose_gui_config(device=device)
    visual, _, masks, names = sequence
    expected = original_propagation(InferenceCore(cutie, cfg), cfg, visual, names,
                                    task_anchors(names), masks, start, direction,
                                    one_hot(masks[start], device), device, True, limit)
    saved, kinds = run_to_end(make_runtime(cutie, cfg, sequence, start, direction, limit, device))
    assert sorted(saved) == sorted(expected)
    for ti in expected:
        assert np.array_equal(saved[ti], expected[ti])
    # anchors are fed, never saved; the empty anchor stops a forward run
    assert 9 not in saved and 2 not in saved
    if direction == 'forward':
        assert kinds[-1] == ('stop', 12)


@pytest.mark.models
def test_clicks_between_steps_leave_propagation_unchanged(networks, device, sequence):
    """Task A propagates while task B clicks between its steps (A100, B, A101, A102, C, A103):
    A's temporal memory is untouched and its masks are those of an uninterrupted run."""
    cutie, click_ctrl = networks
    cfg = compose_gui_config(device=device)
    visual, thermal, masks, _ = sequence
    alone, _ = run_to_end(make_runtime(cutie, cfg, sequence, 4, 'forward', 5, device))

    engine = RitmEngine(click_ctrl, device, True)
    engine.open('b', thermal[7], one_hot(masks[7], device), 1)
    engine.open('c', visual[1], one_hot(masks[1], device), 1)
    clicks = iter([('b', CLICKS_B[0]), None, ('c', CLICKS_A[0]), ('b', CLICKS_B[1]), None,
                   ('c', CLICKS_A[1])] * 4)

    def click():
        item = next(clicks)
        if item is not None:
            engine.click(item[0], *item[1])

    interleaved, _ = run_to_end(make_runtime(cutie, cfg, sequence, 4, 'forward', 5, device), click)
    assert sorted(interleaved) == sorted(alone)
    for ti in alone:
        assert np.array_equal(interleaved[ti], alone[ti])


def test_policies_reproduce_the_original_numbers():
    # 1920x1080 at the default 480 px and at the small-target 960 px (upscaled)
    tokens = policies.memory_tokens_per_frame(1080, 1920, 480, False)
    assert tokens == 30 * 54
    other = policies.other_memory_tokens(tokens, 9, True, 10000)
    assert policies.permanent_anchor_limit(tokens, other) == 136
    big = policies.memory_tokens_per_frame(1080, 1920, 960, True)
    assert big == 60 * 107
    assert policies.permanent_anchor_limit(big, policies.other_memory_tokens(big, 9, True, 10000)) == 1
    assert policies.select_permanent_anchors([0, 10, 20, 30], 14, 2) == [10, 20]
    assert policies.select_permanent_anchors([0, 10, 20, 30], 15, 1) == [10]  # tie: earlier
    small = policies.choose_resolution(1000, 1080, 1920, 480)
    assert small.small_target and small.max_internal_size == 960 and small.upscale_to_internal_size
    assert not policies.choose_resolution(20000, 1080, 1920, 480).small_target
    assert policies.choose_resolution(None, 1080, 1920, 480).message() is None
