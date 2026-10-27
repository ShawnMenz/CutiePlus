"""
The GPU worker process, as the cutieplus-gpu-worker service runs it (as the desktop account):

    python -m cutieplus.gpu_worker --config /etc/cutieplus/service.toml

1. Takes the GPU lock (gpu_worker/lock.py) before anything touches CUDA; exits with 75 when
   another process holds it (systemd starts it again later).
2. Loads the models (or the fake engine of the tests: --engine fake).
3. Connects to the service's worker socket, says who it is, and carries out the service's
   commands one at a time. When the service goes away it connects again, keeping the models;
   each new connection starts from a clean slate (reset).
"""
import argparse
import logging
import os
import socket
import sys
import time
import traceback
import uuid

from cutieplus.cooperation.config import SERVICE_CONFIG, ServiceConfig
from cutieplus.cooperation.protocol import ProtocolError, error, ok, receive_message, send_message
from cutieplus.gpu_worker.engine import OPS, CommandError
from cutieplus.gpu_worker.lock import GpuBusy, acquire_gpu_lock

log = logging.getLogger('cutieplus.gpu_worker')
EX_TEMPFAIL = 75
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gui_config():
    """The labeling window's configuration (models, weights, memory defaults)."""
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base='1.3.2', config_dir=os.path.join(REPO, 'cutie',
                                                                             'config'),
                               job_name='gpu_worker'):
        return compose(config_name='gui_config')


def build_engine(kind: str, config: ServiceConfig):
    if kind == 'fake':
        from cutieplus.gpu_worker.fake import FakeEngine
        return FakeEngine.from_environment(config.workspace_root, config.jobs_dir)
    from cutieplus.gpu_worker.engine import TorchEngine
    return TorchEngine(gui_config(), config.workspace_root, config.jobs_dir)


def serve(sock: socket.socket, engine) -> None:
    """The service's commands, until the connection ends."""
    while True:
        try:
            request = receive_message(sock)
        except (ConnectionError, OSError, ProtocolError):
            return
        op = request.get('op')
        if op not in OPS:
            send_message(sock, error('invalid', f'no command {op!r}'))
            continue
        try:
            reply = ok(getattr(engine, op)(**(request.get('args') or {})))
        except CommandError as exc:
            reply = error(exc.kind, str(exc))
        except (TypeError, KeyError, ValueError) as exc:
            reply = error('invalid', f'{op}: {exc}')
        except Exception as exc:  # reported; the worker stays up
            log.error('%s failed:\n%s', op, traceback.format_exc())
            reply = error('failed', f'{op} failed: {type(exc).__name__}: {exc}')
        try:
            send_message(sock, reply)
        except OSError:
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m cutieplus.gpu_worker')
    parser.add_argument('--config', default=SERVICE_CONFIG)
    parser.add_argument('--engine', choices=['torch', 'fake'], default='torch')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    config = ServiceConfig.load(args.config)
    try:
        lock = acquire_gpu_lock(config.gpu_lock)  # noqa: F841 (held for the process's life)
    except GpuBusy as exc:
        log.error('%s; not starting', exc)
        return EX_TEMPFAIL
    engine = build_engine(args.engine, config)
    instance_id = str(uuid.uuid4())
    hello = engine.hello()
    log.info('ready: %s on %s', hello['engine'], hello['device_name'])
    while True:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(config.worker_socket)
        except OSError:
            sock.close()
            time.sleep(1)
            continue
        try:
            send_message(sock, {'instance_id': instance_id, 'pid': os.getpid(), 'hello': hello})
            accepted = receive_message(sock)
            if not accepted.get('ok'):
                log.warning('the service refused this worker: %s', accepted.get('error'))
                time.sleep(5)
                continue
            log.info('connected as generation %s', accepted['result']['generation'])
            engine.reset()
            serve(sock, engine)
            log.info('the service closed the connection')
        except (ConnectionError, OSError, ProtocolError) as exc:
            log.warning('lost the service: %s', exc)
        finally:
            sock.close()
        time.sleep(0.5)


if __name__ == '__main__':
    sys.exit(main())
