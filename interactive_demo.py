import logging
import os
import sys
# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    """
    Priority 1: If a "images" folder exists in the workspace, we will read from that directory
    Priority 2: If --images is specified, we will copy/resize those images to the workspace
    Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

    In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask
    That way, you can continue annotation from an interrupted run as long as the same workspace is used.
    """
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace',
                        help='directory for storing buffered images (if needed) and output masks',
                        default=None)
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument('--workspace_init_only', action='store_true',
                        help='initialize the workspace and exit')
    parser.add_argument('--tasks', default=None,
                        help='task root with task_set.json: start the task browser')
    parser.add_argument('--task_split', default=None, help='browser tab to show first')
    parser.add_argument('--labeler', default=None,
                        help='name recorded on task edits; default $CUTIE_LABELER, else the OS user')
    parser.add_argument('--backend', choices=('local', 'shared'),
                        default=os.environ.get('CUTIEPLUS_BACKEND', 'local'),
                        help='local: this process runs the models (the original mode); shared: '
                        'the CutiePlus service runs them on its GPU worker '
                        '(python -m cutieplus.coordinator start --tasks ...)')
    parser.add_argument('--service-socket', default=None,
                        help='the service socket; default <task root>/.cutieplus/coordinator.sock')

    args = parser.parse_args()
    return args


def task_root_for(args):
    from os import path
    if args.tasks is not None:
        return path.abspath(args.tasks)
    if args.workspace is not None:
        return path.dirname(path.dirname(path.abspath(args.workspace)))
    return None


def connect_to_service(args):
    """The shared mode's connection to the coordinator, with its session; exits with a clear
    message when the service is not running."""
    from os import path
    from cutieplus.client import ClientError, ServiceClient
    from gui.task_state import labeler_name
    task_root = task_root_for(args)
    if task_root is None:
        sys.exit('The shared service works on tasks: give --tasks or a task --workspace.')
    socket_path = args.service_socket or path.join(task_root, '.cutieplus', 'coordinator.sock')
    client = ServiceClient(socket_path, labeler_name(args.labeler, task_root),
                           reconnect_seconds=5)
    try:
        hello = client.connect()
    except ClientError as exc:
        sys.exit(f'{exc}\nStart it with: python -m cutieplus.coordinator start --tasks {task_root}')
    client.reconnect_seconds = 60
    client.start_heartbeat()
    log.info(f'Connected to the CutiePlus service ({hello.get("gpu")}), session '
             f'{client.session_id}')
    return client


if __name__ in "__main__":
    # input arguments
    args = get_arguments()
    # logging
    log = logging.getLogger()
    if args.backend == 'shared':
        # the GPU belongs to the service's worker: this window never initializes CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # perform slow imports after parsing args
    import torch
    from omegaconf import open_dict
    from hydra import compose, initialize
    from PySide6.QtWidgets import QApplication
    from gui.main_controller import MainController

    # getting hydra's config without using its decorator
    initialize(version_base='1.3.2', config_path="cutie/config", job_name="gui")
    cfg = compose(config_name="gui_config")

    # general setup
    torch.set_grad_enabled(False)
    if args.backend == 'shared':
        device = 'cpu'  # only for drawing; the models run in the service
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    args.device = device
    log.info(f'Using device: {device}')
    service = connect_to_service(args) if args.backend == 'shared' else None

    from gui.task_state import TaskLocked, TaskState
    if (args.tasks is not None or TaskState.is_task(args.workspace)) and args.num_objects != 1:
        log.warning('A Cutie task labels one target; ignoring --num_objects.')
        args.num_objects = 1

    # merge arguments into config
    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            if k in ('backend', 'service_socket'):
                continue
            assert k not in cfg, f'Argument {k} already exists in config'
            cfg[k] = v

    # start everything
    app = QApplication(sys.argv)
    # the theme package is optional (apply_theme falls back to Qt's built-in style); the theme is
    # the one last chosen with the browser's theme button on this screen, Auto at first
    from gui.theme import apply_theme, saved_theme
    apply_theme(saved_theme())
    if cfg['tasks'] is not None:
        from gui.task_browser import TaskBrowser
        browser = TaskBrowser(cfg, cfg['tasks'], cfg['task_split'], service=service)
        code = app.exec()
        if service is not None:
            service.close()
        sys.exit(code)
    try:
        if service is not None:
            from cutieplus.backends.shared import SharedTaskBackend
            ex = MainController(cfg, backend_factory=SharedTaskBackend.factory(service))
        else:
            ex = MainController(cfg)
    except TaskLocked as exc:
        print(f'{exc} Close it there first.', file=sys.stderr)
        sys.exit(1)
    if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
        sys.exit(0)
    else:
        sys.exit(app.exec())
