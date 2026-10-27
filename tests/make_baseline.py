"""
Records the baseline of tests/scenarios.py with the code as it is checked out (run it on the
original implementation; the tests compare later code against the recording):

    uv run python -m tests.make_baseline
"""
import json
import os
import tempfile
from os import path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# the original gui/interactive_utils.py puts its colors on CUDA when it can see a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from tests.conftest import compose_gui_config
from tests.scenarios import flatten, run_session, scenario_task

DATA = path.join(path.dirname(path.abspath(__file__)), 'data')


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from PySide6.QtWidgets import QApplication
    from gui.main_controller import MainController, load_networks
    app = QApplication.instance() or QApplication([])
    networks = load_networks(compose_gui_config(device='cpu'))
    with tempfile.TemporaryDirectory() as base:
        task = scenario_task(base)
        controller = MainController(compose_gui_config(task.workspace), networks=networks)
        result = run_session(controller, task, print)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(path.join(DATA, 'baseline_local_session.npz'), **flatten(result))
    with open(path.join(DATA, 'baseline_local_session.json'), 'w', encoding='utf-8') as f:
        json.dump({key: result[key] for key in ('anchors', 'progress_keys')}, f, indent=2)
    print('baseline written to', DATA)


if __name__ == '__main__':
    main()
