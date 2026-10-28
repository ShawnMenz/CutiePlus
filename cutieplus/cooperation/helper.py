"""
cutieplus-helper: the root part of the cooperation module, on the standard library only. It
answers the cooperation service (the cutieplus account, checked by the kernel's peer
credentials) and nobody else, with a fixed set of operations, each with checked arguments:

    publish_ssh_policy   render the annotation sshd's AuthorizedKeysFile from key entries (no
                         caller text reaches the file), write it atomically, read it back
    write_vnc_password   a desktop's VNC password file, made by tigervncpasswd -f
    configure_desktop    a desktop's environment file (the task root it opens)
    remove_desktop       a desktop's files, once it is stopped and released
    desktop_start/stop   systemctl start/stop cutieplus-desktop@<N>.service, N in the managed range
    desktop_status       the state and main process of those units
    restart_gpu_worker   systemctl restart of the GPU worker's unit (when it stops answering)
    ssh_connections      the live connections the annotation sshd accepted, from its journal
    disconnect           end one of those connections: that sshd process only, by its identity

There is no shell, no file path, unit name or command line from the caller, and no general
systemctl. Run:

    python3 -I /opt/cutieplus/current/cutieplus/cooperation/helper.py \
        --config /etc/cutieplus/service.toml
"""
if __name__ == '__main__' and __package__ in (None, ''):
    # run as a file (python3 -I helper.py): the release folder holds the package
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))))

import argparse
import json
import logging
import os
import pwd
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from os import path
from typing import Dict, List, Optional

from cutieplus.cooperation import procinfo, sshpolicy
from cutieplus.cooperation.config import SERVICE_CONFIG, ServiceConfig
from cutieplus.cooperation.procinfo import Peer, ProcessIdentity
from cutieplus.cooperation.protocol import error, ok

log = logging.getLogger('cutieplus.helper')

SYSTEMCTL = '/usr/bin/systemctl'
JOURNALCTL = '/usr/bin/journalctl'
VNCPASSWD = '/usr/bin/tigervncpasswd'
VNC_PASSWORD = re.compile(r'[A-Za-z0-9]{6,8}')  # VNC uses 8 characters at most
SAFE_PATH = re.compile(r'/[A-Za-z0-9._/-]+')
ACCEPTED = re.compile(r'Accepted publickey for (?P<user>\S+) from (?P<address>\S+) port '
                      r'(?P<port>\d+) ssh2: (?P<type>\S+) (?P<fingerprint>SHA256:[A-Za-z0-9+/]+)')
UNIT_STATES = {'active': 'running', 'reloading': 'running', 'activating': 'starting',
               'deactivating': 'stopping', 'inactive': 'stopped', 'failed': 'error'}


class Refusal(Exception):

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class System:
    """What the helper does to the system; the tests use a fake one."""

    def run(self, argv: List[str], stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
        return subprocess.run(argv, input=stdin, capture_output=True, timeout=120,
                              env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C.UTF-8'})

    def write_atomic(self, file_path: str, data: bytes, mode: int) -> None:
        folder = path.dirname(file_path)
        os.makedirs(folder, mode=0o755, exist_ok=True)
        temporary = f'{file_path}.{os.getpid()}.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                             mode)
        try:
            os.fchmod(descriptor, mode)
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, file_path)
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def read(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            return f.read()

    def remove(self, file_path: str) -> None:
        if path.lexists(file_path):
            os.remove(file_path)

    def kill(self, pid: int, number: int) -> None:
        os.kill(pid, number)

    def identity(self, pid: int) -> Optional[ProcessIdentity]:
        return procinfo.process_identity(pid)

    def unit_of(self, pid: int) -> Optional[str]:
        return procinfo.system_unit_of(pid)

    def journal(self, unit: str, follow: bool):
        """The journal entries of the unit (this boot), as dicts; with follow, forever."""
        argv = [JOURNALCTL, '-o', 'json', f'_SYSTEMD_UNIT={unit}', '--no-pager']
        argv += ['-f', '-n', '0'] if follow else ['-b']
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for line in process.stdout:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
        finally:
            process.kill()
            process.wait()


@dataclass
class Connection:
    identity: ProcessIdentity
    address: str
    port: int
    fingerprint: str
    accepted_at_us: int  # CLOCK_REALTIME microseconds, from the journal

    def as_dict(self) -> dict:
        return {'identity': self.identity.as_dict(), 'address': self.address, 'port': self.port,
                'fingerprint': self.fingerprint, 'accepted_at_us': self.accepted_at_us}


class ConnectionTracker:
    """The connections the annotation sshd accepted, from its journal entries. Only entries
    journald itself marks as coming from that unit's root process count (the fields beginning
    with "_" are written by journald, not by the logging process)."""

    def __init__(self, system: System, unit: str, user: str):
        self.system = system
        self.unit = unit
        self.user = user
        self.lock = threading.Lock()
        self.connections: Dict[int, Connection] = {}

    def entry(self, entry: dict) -> None:
        if entry.get('_SYSTEMD_UNIT') != self.unit or entry.get('_UID') != '0':
            return
        match = ACCEPTED.fullmatch(str(entry.get('MESSAGE', '')).strip())
        if match is None or match.group('user') != self.user:
            return
        try:
            pid = int(entry['_PID'])
            monotonic_us = int(entry.get('_SOURCE_MONOTONIC_TIMESTAMP')
                               or entry['__MONOTONIC_TIMESTAMP'])
            realtime_us = int(entry.get('_SOURCE_REALTIME_TIMESTAMP')
                              or entry['__REALTIME_TIMESTAMP'])
        except (KeyError, ValueError):
            return
        identity = self.system.identity(pid)
        # the process must have started before the entry: else its PID was reused since
        if identity is None or identity.start_ticks / os.sysconf('SC_CLK_TCK') > \
                monotonic_us / 1e6 + 1:
            return
        with self.lock:
            self.connections[pid] = Connection(identity, match.group('address'),
                                               int(match.group('port')),
                                               match.group('fingerprint'), realtime_us)

    def alive(self) -> List[Connection]:
        with self.lock:
            for pid, connection in list(self.connections.items()):
                if self.system.identity(pid) != connection.identity:
                    del self.connections[pid]
            return list(self.connections.values())

    def find(self, identity: ProcessIdentity) -> Optional[Connection]:
        return next((connection for connection in self.alive()
                     if connection.identity == identity), None)

    def follow(self) -> None:
        for entry in self.system.journal(self.unit, follow=False):
            self.entry(entry)
        while True:
            try:
                for entry in self.system.journal(self.unit, follow=True):
                    self.entry(entry)
            except Exception:  # journalctl ended: start it again
                log.exception('following the journal failed')
            time.sleep(2)


class Helper:

    def __init__(self, config: ServiceConfig, system: Optional[System] = None,
                 service_uid: Optional[int] = None):
        self.config = config
        self.system = system or System()
        self.service_uid = service_uid if service_uid is not None else \
            pwd.getpwnam(config.service_user).pw_uid
        self.tracker = ConnectionTracker(self.system, config.sshd_unit, config.desktop_user)
        self.lock = threading.Lock()  # one operation at a time
        self.operations = {
            'hello': self.hello, 'publish_ssh_policy': self.publish_ssh_policy,
            'write_vnc_password': self.write_vnc_password,
            'configure_desktop': self.configure_desktop, 'remove_desktop': self.remove_desktop,
            'desktop_start': self.desktop_start, 'desktop_stop': self.desktop_stop,
            'desktop_status': self.desktop_status, 'ssh_connections': self.ssh_connections,
            'disconnect': self.disconnect, 'restart_gpu_worker': self.restart_gpu_worker,
        }

    def handle(self, request: dict, peer: Peer) -> dict:
        if peer.uid != self.service_uid:
            return error('forbidden', 'the helper answers the cooperation service only')
        operation = self.operations.get(request.get('op'))
        args = request.get('args')
        if operation is None or not isinstance(args, dict):
            return error('bad_request', f'no operation {request.get("op")!r}')
        try:
            with self.lock:
                return ok(operation(**args))
        except TypeError as exc:
            return error('bad_request', str(exc))
        except (Refusal, sshpolicy.KeyError_) as exc:
            return error(getattr(exc, 'code', 'invalid'), str(exc))
        except (OSError, subprocess.SubprocessError) as exc:
            log.exception('%s failed', request.get('op'))
            return error('failed', f'{request.get("op")} failed: {exc}')

    # checks
    def display(self, value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or \
                value not in self.config.displays():
            raise Refusal('invalid', f'display {value!r} is outside the managed range')
        return value

    def unit(self, display: int) -> str:
        return self.config.desktop_unit_of(self.display(display))

    # operations
    def hello(self) -> dict:
        return {'version': 1, 'displays': [self.config.first_display, self.config.last_display],
                'authorized_keys': self.config.authorized_keys}

    def publish_ssh_policy(self, revision, entries) -> dict:
        """The annotation sshd's policy and port 22's jump file, from the same entries: the jump
        file is written last, so a key it lets in finds its line in the policy."""
        if not isinstance(entries, list):
            raise Refusal('invalid', 'entries is a list')
        entries = [sshpolicy.PolicyEntry.from_dict(entry) for entry in entries]
        ports = self.config.vnc_ports
        texts = (sshpolicy.render_policy(revision, entries, ports),
                 sshpolicy.render_jump_policy(revision, entries, ports, self.config.ssh_port))
        for file_path, text in zip((self.config.authorized_keys, self.config.jump_keys), texts):
            data = text.encode('utf-8')
            self.system.write_atomic(file_path, data, 0o644)
            if self.system.read(file_path) != data:
                raise Refusal('failed', f'{file_path} reads back differently')
        return {'revision': revision, 'checksum': sshpolicy.policies_checksum(*texts),
                'keys': len(entries)}

    def write_vnc_password(self, display, password) -> dict:
        display = self.display(display)
        if not isinstance(password, str) or not VNC_PASSWORD.fullmatch(password):
            raise Refusal('invalid', 'a VNC password has 6-8 letters or digits')
        result = self.system.run([VNCPASSWD, '-f'], stdin=password.encode('ascii') + b'\n')
        if result.returncode != 0 or not result.stdout:
            raise Refusal('failed', 'tigervncpasswd failed')
        file_path = self.config.vnc_password_file(display)
        self.system.write_atomic(file_path, result.stdout, 0o600)
        return {'file': file_path}

    def configure_desktop(self, display, tasks_dir) -> dict:
        display = self.display(display)
        if not isinstance(tasks_dir, str) or not SAFE_PATH.fullmatch(tasks_dir) or \
                not self.config.within_workspace(tasks_dir):
            raise Refusal('invalid', 'the task folder must lie in the workspace')
        file_path = path.join(self.config.desktops_dir, str(display), 'env')
        text = f'CUTIEPLUS_TASKS={path.normpath(tasks_dir)}\nCUTIEPLUS_DISPLAY={display}\n'
        self.system.write_atomic(file_path, text.encode('ascii'), 0o644)
        return {'file': file_path}

    def remove_desktop(self, display) -> dict:
        display = self.display(display)
        if self.desktop_status([display])[str(display)]['state'] not in ('stopped', 'error'):
            raise Refusal('invalid', f'desktop :{display} is still running')
        for name in ('vncpasswd', 'env'):
            self.system.remove(path.join(self.config.desktops_dir, str(display), name))
        return {'display': display}

    def desktop_start(self, display) -> dict:
        unit = self.unit(display)
        for name in ('vncpasswd', 'env'):
            if not path.exists(path.join(self.config.desktops_dir, str(display), name)):
                raise Refusal('invalid', f'desktop :{display} has no {name} yet')
        result = self.system.run([SYSTEMCTL, 'start', '--no-block', unit])
        if result.returncode != 0:
            raise Refusal('failed', result.stderr.decode('utf-8', 'replace').strip())
        return self.desktop_status([display])[str(display)]

    def desktop_stop(self, display) -> dict:
        unit = self.unit(display)
        result = self.system.run([SYSTEMCTL, 'stop', '--no-block', unit])
        if result.returncode != 0:
            raise Refusal('failed', result.stderr.decode('utf-8', 'replace').strip())
        return self.desktop_status([display])[str(display)]

    def desktop_status(self, displays) -> dict:
        if not isinstance(displays, list):
            raise Refusal('invalid', 'displays is a list')
        states = {}
        for display in displays:
            unit = self.unit(display)
            result = self.system.run([SYSTEMCTL, 'show', unit, '--property=ActiveState',
                                      '--property=MainPID'])
            values = dict(line.split('=', 1) for line in
                          result.stdout.decode('utf-8', 'replace').splitlines() if '=' in line)
            pid = int(values.get('MainPID') or 0)
            identity = self.system.identity(pid) if pid else None
            states[str(display)] = {
                'state': UNIT_STATES.get(values.get('ActiveState', ''), 'error'),
                'identity': identity.as_dict() if identity else None}
        return states

    def restart_gpu_worker(self) -> dict:
        """The GPU worker stopped answering: systemd stops its whole process group, then starts
        it again (the new one takes the GPU lock only once the old processes are gone)."""
        result = self.system.run([SYSTEMCTL, 'restart', '--no-block', self.config.worker_unit])
        if result.returncode != 0:
            raise Refusal('failed', result.stderr.decode('utf-8', 'replace').strip())
        return {'unit': self.config.worker_unit}

    def ssh_connections(self) -> list:
        return [connection.as_dict() for connection in self.tracker.alive()]

    def disconnect(self, identity) -> dict:
        """Ends one connection the annotation sshd accepted: refused for any other process."""
        try:
            identity = ProcessIdentity.from_dict(identity)
        except (KeyError, TypeError, ValueError):
            raise Refusal('invalid', 'identity is {pid, start_ticks, boot_id}') from None
        if identity is None or self.tracker.find(identity) is None:
            return {'closed': True, 'already': True}  # gone already, or never one of them
        if self.system.unit_of(identity.pid) != self.config.sshd_unit:
            raise Refusal('invalid', 'not a process of the annotation sshd')
        self.system.kill(identity.pid, signal.SIGTERM)
        for _ in range(50):
            if self.system.identity(identity.pid) != identity:
                return {'closed': True, 'already': False}
            time.sleep(0.1)
        self.system.kill(identity.pid, signal.SIGKILL)
        return {'closed': self.system.identity(identity.pid) != identity, 'already': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='cutieplus-helper')
    parser.add_argument('--config', default=SERVICE_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    if os.geteuid() != 0:
        log.error('the helper runs as root')
        return 1
    config = ServiceConfig.load(args.config)
    helper = Helper(config)
    threading.Thread(target=helper.tracker.follow, name='journal', daemon=True).start()
    from cutieplus.cooperation.server import Server
    server = Server(helper.handle, config.helper_socket, mode=0o660, name='helper')
    server.start()
    os.chown(config.helper_socket, 0, pwd.getpwnam(config.service_user).pw_gid)
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    stop.wait()
    server.stop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
