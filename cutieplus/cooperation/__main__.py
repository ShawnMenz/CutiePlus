"""
The cooperation service, as its systemd unit runs it (deploy/cutieplus-service.service), as the
cutieplus account:

    python -m cutieplus.cooperation serve --config /etc/cutieplus/service.toml

It answers the windows on the app socket until SIGTERM. The maintenance commands are
cutieplus.cooperation.admin_cli (cutieplus-admin).
"""
import argparse
import logging
import signal
import sys
import threading

from cutieplus.cooperation.config import SERVICE_CONFIG, ServiceConfig

log = logging.getLogger('cutieplus.cooperation')


def serve(config: ServiceConfig) -> None:
    from cutieplus.cooperation.app import build_service
    from cutieplus.cooperation.server import Server
    service = build_service(config)
    # every local process may connect; every request but sign-in needs a session
    server = Server(service.handle, config.app_socket, mode=0o666, name='app')
    server.start()
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    log.info('the cooperation service is ready (schema %d)', service.db.schema_version())
    stop.wait()
    server.stop()
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
