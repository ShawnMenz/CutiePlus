"""
Starts, stops and inspects the CutiePlus coordinator (and with it the GPU worker):

    python -m cutieplus.coordinator start --tasks workspace/cutie_tasks
    python -m cutieplus.coordinator status --tasks workspace/cutie_tasks
    python -m cutieplus.coordinator stop --tasks workspace/cutie_tasks

Settings come from cutie/config/gui_config.yaml (section service); --set key=value overrides one
(e.g. --set gpu.max_worker_gib=10).
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
from os import path

REPO = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


def load_config():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base='1.3.2', config_dir=path.join(REPO, 'cutie', 'config'),
                               job_name='coordinator'):
        return compose(config_name='gui_config')


def parse_value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def overrides_from(settings) -> dict:
    overrides: dict = {}
    for item in settings or ():
        key, _, value = item.partition('=')
        target = overrides
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parse_value(value)
    return overrides


def socket_for(args) -> str:
    from cutieplus.paths import socket_for as derive
    return derive(args.tasks, args.state_dir, args.socket)


def start(args) -> int:
    from cutieplus.coordinator.server import Server
    from cutieplus.coordinator.service import AlreadyRunning, Service, default_state_dir
    from cutieplus.gpu.hardware import NoGpuError
    from cutieplus.coordinator.worker_proxy import WorkerDied

    state_dir = path.abspath(args.state_dir or default_state_dir(args.tasks))
    os.makedirs(path.join(state_dir, 'logs'), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s coordinator %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr),
                                  logging.FileHandler(path.join(state_dir, 'logs',
                                                                'coordinator.log'))])
    overrides = overrides_from(args.set)
    if args.socket:
        overrides['socket'] = args.socket
    try:
        service = Service(load_config(), args.tasks, state_dir, overrides)
    except AlreadyRunning as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        service.start()
    except (WorkerDied, NoGpuError) as exc:
        print(f'The GPU worker could not start: {exc}', file=sys.stderr)
        service.stop()
        return 2
    server = Server(service, service.socket_path)
    server.start()
    stopped = threading.Event()
    service.on_shutdown = stopped.set
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    hardware = service.worker_hello.get('hardware', {})
    print(f'CutiePlus coordinator ready: {service.socket_path} (GPU worker on '
          f'{hardware.get("name")}, pid {service.worker.pid})', flush=True)
    stopped.wait()
    server.stop()
    service.stop()
    return 0


def request(args, op: str, **payload):
    import socket as socketlib
    from cutieplus.protocol import receive_message, send_message
    sock = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(socket_for(args))
    send_message(sock, {'op': op, 'args': payload})
    reply = receive_message(sock)
    sock.close()
    return reply


def status(args) -> int:
    try:
        reply = request(args, 'status')
    except OSError as exc:
        print(f'not running ({exc})')
        return 1
    print(json.dumps(reply['result'], indent=2, default=str))
    return 0


def stop(args) -> int:
    try:
        request(args, 'shutdown')
    except OSError as exc:
        print(f'not running ({exc})')
        return 1
    print('stopping')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m cutieplus.coordinator')
    parser.add_argument('command', choices=('start', 'status', 'stop'))
    parser.add_argument('--tasks', required=True, help='the task root (with task_set.json)')
    parser.add_argument('--state-dir', default=None, help='default: <task root>/.cutieplus')
    parser.add_argument('--socket', default=None, help='default: <state dir>/coordinator.sock')
    parser.add_argument('--set', action='append', help='a service setting, e.g. gpu.max_worker_gib=10')
    args = parser.parse_args(argv)
    args.tasks = path.abspath(args.tasks)
    return {'start': start, 'status': status, 'stop': stop}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
