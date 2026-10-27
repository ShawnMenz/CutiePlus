"""
The cooperation service, as its systemd unit runs it (deploy/cutieplus-service.service), as the
cutieplus account:

    python -m cutieplus.cooperation serve --config /etc/cutieplus/service.toml

It answers the windows on the app socket, runs the GPU queue (the worker connects to the worker
socket), keeps the system in step with the database through the root helper (the reconciler, every
few seconds and at once after a change), and backs the database up on schedule, until SIGTERM.
The maintenance commands are cutieplus.cooperation.admin_cli (cutieplus-admin).
"""
import argparse
import logging
import signal
import sys
import threading

from cutieplus.cooperation.config import SERVICE_CONFIG, ServiceConfig

log = logging.getLogger('cutieplus.cooperation')
BACKUP_CHECK_S = 600


def backup_loop(service, stop: threading.Event) -> None:
    """A backup whenever the newest is older than backup_interval_hours."""
    import os
    import time
    from cutieplus.cooperation.backups import backup_files, backup_now
    while not stop.wait(BACKUP_CHECK_S if backup_files(service.config.backups_dir) else 5):
        files = backup_files(service.config.backups_dir)
        hours = service.setting('backup_interval_hours')
        if not files or time.time() - os.path.getmtime(files[-1]) > hours * 3600:
            try:
                log.info('backed up to %s', backup_now(service))
            except Exception:
                log.exception('the backup failed')


def serve(config: ServiceConfig) -> None:
    from cutieplus.cooperation.app import build_service
    from cutieplus.cooperation.client import ServiceClient
    from cutieplus.cooperation.server import Server
    helper = ServiceClient(config.helper_socket, timeout=120, name='the root helper')
    service = build_service(config, helper=helper, gpu=True)
    if service.restored is not None:
        log.warning('the database was restored: %s', service.restored)
    # every local process may connect; every request but sign-in needs a session
    server = Server(service.handle, config.app_socket, mode=0o666, name='app')
    server.start()
    service.gpu.start()
    stop = threading.Event()
    threading.Thread(target=service.reconciler.loop, args=(stop, ), name='reconciler',
                     daemon=True).start()
    threading.Thread(target=backup_loop, args=(service, stop), name='backups',
                     daemon=True).start()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    log.info('the cooperation service is ready (schema %d)', service.db.schema_version())
    stop.wait()
    server.stop()
    service.gpu.stop()
    service.db.close()
    log.info('stopped')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m cutieplus.cooperation')
    parser.add_argument('command', choices=['serve'])
    parser.add_argument('--config', default=SERVICE_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    serve(ServiceConfig.load(args.config))
    return 0


if __name__ == '__main__':
    sys.exit(main())
