import os
from os import path

# headless Qt for the GUI tests; set before PySide6 is imported anywhere
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# the CPU tests run without CUDA; the GPU tests are a separate run:
#   CUTIEPLUS_TEST_GPU=1 uv run pytest -m gpu
if os.environ.get('CUTIEPLUS_TEST_GPU') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pytest
import torch
from omegaconf import DictConfig, open_dict

REPO = path.dirname(path.dirname(path.abspath(__file__)))
WEIGHTS = path.join(REPO, 'weights')
CUTIE_WEIGHTS = path.join(WEIGHTS, 'cutie-base-mega.pth')
RITM_WEIGHTS = path.join(WEIGHTS, 'coco_lvis_h18_itermask.pth')

torch.set_grad_enabled(False)
# CPU results are compared with recorded ones; the thread count changes float reductions
torch.set_num_threads(4)


def have_weights() -> bool:
    return path.isfile(CUTIE_WEIGHTS) and path.isfile(RITM_WEIGHTS)


def pytest_collection_modifyitems(config, items):
    skip_weights = pytest.mark.skip(reason='model weights not found in weights/')
    skip_gpu = pytest.mark.skip(reason='no CUDA device')
    for item in items:
        if 'models' in item.keywords and not have_weights():
            item.add_marker(skip_weights)
        if 'gpu' in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


def compose_gui_config(workspace=None, device: str = 'cpu', **overrides) -> DictConfig:
    """The GUI's config as interactive_demo.py builds it."""
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base='1.3.2', config_dir=path.join(REPO, 'cutie', 'config'),
                               job_name='test'):
        cfg = compose(config_name='gui_config')
    arguments = {
        'images': None,
        'video': None,
        'workspace': workspace,
        'num_objects': 1,
        'workspace_init_only': False,
        'tasks': None,
        'task_split': None,
        'labeler': 'tester',
        'device': device,
    }
    with open_dict(cfg):
        for key, value in arguments.items():
            cfg[key] = value
        cfg['weights'] = CUTIE_WEIGHTS
        cfg['ritm_weights'] = RITM_WEIGHTS
        cfg['num_read_workers'] = 0
        for key, value in overrides.items():
            cfg[key] = value
    return cfg


@pytest.fixture(autouse=True)
def runtime_folder(tmp_path_factory, monkeypatch):
    # the task locks live in the user's runtime folder (gui/task_state.py: lock_path): the tests
    # keep theirs in a temporary one
    folder = tmp_path_factory.getbasetemp() / 'runtime'
    folder.mkdir(exist_ok=True)
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(folder))
    return folder


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


_networks = {}


def networks_on(device: str):
    """Cutie and RITM on a device, loaded once per test session."""
    if not have_weights():
        pytest.skip('model weights not found in weights/')
    if device not in _networks:
        from cutieplus.engine.models import load_networks
        _networks[device] = load_networks(compose_gui_config(device=device), device)
    return _networks[device]


@pytest.fixture(scope='session')
def cpu_networks():
    return networks_on('cpu')


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def device(request):
    """Runs a test on the CPU, and on the GPU in the GPU run (-m gpu)."""
    return request.param


@pytest.fixture
def networks(device):
    return networks_on(device)
