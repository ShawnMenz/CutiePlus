"""
A scripted labeling session driven through MainController's public handlers, the same way the
window's buttons and keys call them. It is run against the original code to record the baseline
(tests/make_baseline.py) and against the refactored code by the tests, which compare the results.
"""
import os
from os import path
from typing import Callable, Dict

import numpy as np

from tests.helpers import RADII, WIDTH, HEIGHT, SyntheticTask, make_task, read_mask, strip_times

FRAME_COUNT = 16
EMPTY_ANCHOR = 13
PREFILL = (8, )


def scenario_task(base: str) -> SyntheticTask:
    return make_task(base, frame_count=FRAME_COUNT, mask_anchors=(0, ),
                     empty_anchors=(EMPTY_ANCHOR, ), prefill=PREFILL)


def ellipse_center(index: int) -> tuple:
    ry, rx = RADII
    cx = rx + 4 + (WIDTH - 2 * rx - 8) * index / (FRAME_COUNT - 1)
    cy = HEIGHT / 2 + 6 * np.sin(index / 3)
    return int(round(cx)), int(round(cy))


def go_to(controller, ti: int) -> None:
    controller.gui.tl_slider.setValue(ti)


def run_session(controller, task: SyntheticTask, log: Callable[[str], None] = lambda _: None) -> Dict:
    """Clicks, an accepted anchor, forward and backward propagation, anchor removal with undo and
    redo, and a submission for review. Returns what ended up on disk."""
    steps = {}

    # two clicks on frame 4, then the second is undone
    go_to(controller, 4)
    x, y = ellipse_center(4)
    controller.click_fn('left', x, y)
    controller.click_fn('right', x + 22, y)
    steps['two_clicks'] = task.mask(task.names[4]).copy()
    controller.on_undo()
    steps['after_click_undo'] = task.mask(task.names[4]).copy()

    # forward: stops before the empty anchor
    controller.on_forward_propagation()
    log('forward done')
    steps['after_forward'] = {name: task.mask(name) for name in task.names}

    # accept frame 9's propagated mask as an anchor, then propagate backward through the anchors
    go_to(controller, 9)
    controller.on_commit()
    controller.on_backward_propagation()
    log('backward done')
    steps['after_backward'] = {name: task.mask(name) for name in task.names}

    # remove the anchor on frame 9, undo, redo, undo
    go_to(controller, 9)
    controller.on_remove_anchor()
    steps['removed_9'] = task.mask(task.names[9])
    controller.on_undo()
    controller.on_redo()
    controller.on_undo()
    steps['restored_9'] = task.mask(task.names[9])

    # reset the object on frame 11 (an edit), then submit for review
    go_to(controller, 11)
    controller.on_reset_object()
    controller.on_export_binary()

    controller.gui.close()
    return {
        'steps': steps,
        'masks': {name: task.mask(name) for name in task.names},
        'anchors': strip_times(task.anchors()),
        'progress_keys': sorted(task.progress()),
        'binary_masks': sorted(os.listdir(path.join(task.workspace, 'binary_masks'))),
    }


def flatten(result: Dict) -> Dict[str, np.ndarray]:
    """Arrays of a session result, keyed for np.savez; a missing mask is an empty array."""
    arrays = {}

    def put(key, mask):
        arrays[key] = np.zeros((0, ), np.uint8) if mask is None else np.asarray(mask, np.uint8)

    for key, value in result['steps'].items():
        if isinstance(value, dict):
            for name, mask in value.items():
                put(f'steps/{key}/{name}', mask)
        else:
            put(f'steps/{key}', value)
    for name, mask in result['masks'].items():
        put(f'masks/{name}', mask)
    return arrays
