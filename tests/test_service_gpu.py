"""
The service with the real GPU worker: a propagation while another session clicks between its steps
gives the same masks as the same propagation alone (Cutie's temporal memory is kept across the
interleaving), and the worker holds one propagation context and returns its memory afterwards.
Run with CUTIEPLUS_TEST_GPU=1 pytest -m gpu.
"""
import os
import threading

import numpy as np
import pytest

from tests.helpers import make_task
from tests.service_env import ServiceEnv, run_propagation

pytestmark = [pytest.mark.gpu, pytest.mark.models, pytest.mark.slow]

MEMORY = {'mem_every': 5, 'top_k': 30, 'min_mem_frames': 4, 'max_mem_frames': 9,
          'max_num_tokens': 10000}


def test_clicks_between_steps_keep_the_propagation_identical(tmp_path):
    base = str(tmp_path)
    alone = make_task(base, sequence='0001', frame_count=40, mask_anchors=(0, 25),
                      empty_anchors=(38, ))
    interleaved = make_task(base, sequence='0002', frame_count=40, mask_anchors=(0, 25),
                            empty_anchors=(38, ))
    other = make_task(base, sequence='0003', frame_count=40, mask_anchors=(0, ))
    env = ServiceEnv(alone.root, str(tmp_path / 'state'),
                     {'device': 'cuda', 'profile_frame_size': [96, 128],
                      'scheduler': {'max_interactive_burst': 1, 'max_background_wait': 10}},
                     device='cuda')
    try:
        alice = env.client('alice', ':2')
        bob = env.client('bob', ':3')
        a1 = alice.call('open_task', workspace=alone.workspace)
        done = run_propagation(alice, a1['task_id'], a1['lease_epoch'], '000001', memory=MEMORY)
        assert done['state'] == 'finished'

        a2 = alice.call('open_task', workspace=interleaved.workspace)
        b = bob.call('open_task', workspace=other.workspace)
        clicks = []
        stop = threading.Event()

        def click_loop():
            index = 0
            while not stop.is_set():
                frame = f'{5 + index % 30:06d}'
                interaction = bob.call('open_interaction', task_id=b['task_id'],
                                       lease_epoch=b['lease_epoch'], frame=frame,
                                       modality='thermal', target_object=1)['interaction_id']
                reply = bob.call('click', task_id=b['task_id'], lease_epoch=b['lease_epoch'],
                                 interaction_id=interaction, x=40 + index % 20, y=48,
                                 is_negative=False, request_id=os.urandom(8).hex())
                clicks.append(reply['clicks'])
                index += 1

        clicker = threading.Thread(target=click_loop)
        clicker.start()
        try:
            done = run_propagation(alice, a2['task_id'], a2['lease_epoch'], '000001',
                                   memory=MEMORY)
        finally:
            stop.set()
            clicker.join()
        assert done['state'] == 'finished'
        assert len(clicks) >= 5
        # clicks really ran between the propagation's steps
        history = [kind for kind, *_ in env.service.scheduler.history]
        switches = sum(1 for x, y in zip(history, history[1:]) if x != y)
        assert switches >= 10
        for frame in alone.names:
            a, b2 = alone.mask(frame), interleaved.mask(frame)
            assert (a is None) == (b2 is None), frame
            if a is not None:
                assert np.array_equal(a, b2), frame
        status = alice.call('gpu_status')
        assert status['job'] is None
        assert status['memory']['reserved'] < status['peak_reserved']  # cache given back
    finally:
        env.close()
