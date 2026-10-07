"""
One labeling window, offscreen, driven through MainController's handlers: clicks, then a forward
propagation, with timings. It runs against whichever implementation is first on PYTHONPATH (the
original checkout, or this branch in local or shared mode), so the same steps measure all three.

    PYTHONPATH=<repo> python gui_driver.py --mode local|shared --repo <repo> --workspace <task>
        --plan <clicks.json> --start <frame index> --output <result.json> [--socket <path>]
"""
import argparse
import json
import os
import sys
import time
from os import path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('local', 'shared'), required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--socket', default=None)
    parser.add_argument('--no-propagation', action='store_true')
    args = parser.parse_args()

    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    if args.mode == 'shared':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.chdir(args.repo)
    sys.path.insert(0, args.repo)
    import torch
    torch.set_grad_enabled(False)
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from PySide6.QtWidgets import QApplication
    from gui.main_controller import MainController, load_networks

    with initialize_config_dir(version_base='1.3.2', config_dir=path.join(args.repo, 'cutie', 'config'),
                               job_name='bench'):
        cfg = compose(config_name='gui_config')
    with open_dict(cfg):
        cfg.update({'images': None, 'video': None, 'workspace': args.workspace, 'num_objects': 1,
                    'workspace_init_only': False, 'tasks': None, 'task_split': None,
                    'labeler': 'bench', 'device': 'cuda' if args.mode == 'local' else 'cpu'})
    app = QApplication.instance() or QApplication([])
    result = {'mode': args.mode, 'repo': args.repo, 'pid': os.getpid()}

    start = time.perf_counter()
    if args.mode == 'local':
        networks = load_networks(cfg)
        torch.cuda.synchronize()
        result['model_load_seconds'] = time.perf_counter() - start
        result['model_bytes'] = torch.cuda.memory_allocated()
        start = time.perf_counter()
        controller = MainController(cfg, networks=networks)
    else:
        from cutieplus.backends.shared import SharedTaskBackend
        from cutieplus.client import ServiceClient
        client = ServiceClient(args.socket, 'bench', ':99')
        client.connect()
        client.start_heartbeat()
        controller = MainController(cfg, backend_factory=SharedTaskBackend.factory(client))
    result['open_seconds'] = time.perf_counter() - start
    processor = getattr(controller, 'processor', None) or getattr(getattr(controller, 'backend', None),
                                                                 'processor', None)
    if processor is not None:
        result['processing_size'] = [processor.max_internal_size, processor.upscale_to_internal_size]
    else:
        result['resolution_message'] = controller.backend.resolution_message

    with open(args.plan, encoding='utf-8') as f:
        plan = json.load(f)
    latencies = []
    for ti, clicks in plan:
        controller.gui.tl_slider.setValue(ti)
        for x, y, negative in clicks:
            begin = time.perf_counter()
            controller.click_fn('right' if negative else 'left', int(x), int(y))
            if args.mode == 'local':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - begin)
        controller.gui.tl_slider.setValue(min(ti + 1, controller.T - 1))
    result['click_seconds'] = latencies

    if not args.no_propagation:
        masks_dir = path.join(args.workspace, 'masks')
        before = len(os.listdir(masks_dir))
        controller.gui.tl_slider.setValue(args.start)
        begin = time.perf_counter()
        controller.on_forward_propagation()
        if args.mode == 'local':
            torch.cuda.synchronize()
        result['propagation_seconds'] = time.perf_counter() - begin
        begin = time.perf_counter()
        if hasattr(controller, 'res_man'):
            controller.res_man.save_queue.join()
        result['save_join_seconds'] = time.perf_counter() - begin
        result['frames_written'] = len(os.listdir(masks_dir)) - before
        result['stopped_at'] = controller.curr_ti
    if args.mode == 'local':
        result['peak_allocated'] = torch.cuda.max_memory_allocated()
        result['peak_reserved'] = torch.cuda.max_memory_reserved()
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            pid, used = (part.strip() for part in line.split(','))
            if int(pid) == os.getpid():
                result['process_mib'] = int(used)
    else:
        result['cuda_initialized'] = torch.cuda.is_initialized()
    controller.gui.close()
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    if args.mode == 'shared':
        client.close()
    del app


if __name__ == '__main__':
    main()
