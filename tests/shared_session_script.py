"""
Runs the scripted labeling session in shared mode in a process of its own (the window side and
the coordinator; the GPU worker is spawned) and saves the results:

    python -m tests.shared_session_script <base dir> <worker device> <output .npz>

The output records whether this process initialized CUDA (it must not: the window draws on the
CPU and the coordinator never uses the GPU).
"""
import json
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def main(base: str, device: str, output: str) -> None:
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from tests.scenarios import flatten
    from tests.test_shared_gui import run_shared_session
    import pathlib
    result, _ = run_shared_session(pathlib.Path(base), device)
    np.savez_compressed(output, **flatten(result))
    with open(output + '.json', 'w') as f:
        json.dump({'cuda_initialized': torch.cuda.is_initialized(), 'anchors': result['anchors']}, f)
    del app


if __name__ == '__main__':
    main(*sys.argv[1:4])
