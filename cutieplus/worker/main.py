"""
The GPU worker process. The coordinator starts it with the spawn method and talks to it through a
duplex pipe, one command at a time:

    request  {'seq': n, 'op': name, 'args': {...}}
    reply    {'seq': n, 'ok': True, 'result': {...}, 'seconds': s}
             {'seq': n, 'ok': False, 'error': {'kind': ..., 'message': ...}, 'seconds': s}

The first message the worker sends is {'event': 'ready', ...} (hardware, memory profile, model
identity) or {'event': 'failed', 'error': ...}. The worker exits when the pipe closes, so a
coordinator that dies takes its worker with it.
"""
import logging
import os
import sys
import time
import traceback
from multiprocessing.connection import Connection
from typing import Optional

OPS = {'hello', 'status', 'ritm_open', 'ritm_click', 'ritm_undo', 'ritm_replay', 'ritm_close',
       'prop_start', 'prop_step', 'prop_release', 'relieve_memory'}
TEST_OPS = {'test_inject', 'test_set_memory', 'test_crash'}


def worker_main(conn: Connection, cfg_container: dict, service: dict,
                log_path: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format=f'%(asctime)s worker[{os.getpid()}] %(levelname)s %(message)s')
    log = logging.getLogger('cutieplus.worker')
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(cfg_container)
        if service.get('engine') == 'fake':
            from cutieplus.worker.fake import FakeWorkerEngine
            engine = FakeWorkerEngine(cfg, service)
        else:
            from cutieplus.worker.engine import TorchWorkerEngine
            engine = TorchWorkerEngine(cfg, service)
        hello = engine.hello()
        conn.send(dict(hello, event='ready', pid=os.getpid()))
        log.info('ready on %s', hello['hardware'].get('name'))
    except Exception as exc:  # reported to the coordinator, which shows it
        log.exception('worker start failed')
        conn.send({'event': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                   'traceback': traceback.format_exc()})
        return

    allowed = OPS | (TEST_OPS if service.get('allow_test_commands') else set())
    from cutieplus.worker.engine import CommandError
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            log.info('the coordinator is gone; exiting')
            break
        seq = message.get('seq')
        op = message.get('op')
        if op == 'shutdown':
            conn.send({'seq': seq, 'ok': True, 'result': {}, 'seconds': 0.0})
            break
        start = time.perf_counter()
        try:
            if op not in allowed:
                raise CommandError('invalid', f'unknown command {op!r}')
            result = getattr(engine, op)(**message.get('args', {}))
            reply = {'seq': seq, 'ok': True, 'result': result}
        except CommandError as exc:
            reply = {'seq': seq, 'ok': False,
                     'error': dict(exc.details, kind=exc.kind, message=str(exc))}
        except Exception as exc:
            log.exception('command %s failed', op)
            reply = {'seq': seq, 'ok': False,
                     'error': {'kind': 'internal', 'message': f'{type(exc).__name__}: {exc}'}}
        reply['seconds'] = time.perf_counter() - start
        try:
            conn.send(reply)
        except (BrokenPipeError, OSError):
            break
    try:
        engine.close()
    except Exception:
        log.exception('closing the engine')
