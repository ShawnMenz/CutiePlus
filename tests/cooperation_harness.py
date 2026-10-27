"""
The cooperation service in the test process: a temporary database, a clock the test moves, a fast
password hash, and peers whose systemd unit the test decides (a remote desktop :N, or the
console). Requests go through Service.handle, as the socket server would pass them.
"""
import os
from typing import Optional

from cutieplus.cooperation.app import build_service
from cutieplus.cooperation.auth import Passwords, password_hasher
from cutieplus.cooperation.client import ClientError
from cutieplus.cooperation.clock import FakeClock
from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.db import Database
from cutieplus.cooperation.procinfo import Peer

PASSWORD = 'correct horse battery'


class Harness:

    def __init__(self, tmp_path, **config):
        state = tmp_path / 'state'
        state.mkdir(exist_ok=True)
        workspace = tmp_path / 'workspace'
        workspace.mkdir(exist_ok=True)
        self.config = ServiceConfig(state_dir=str(state), workspace_root=str(workspace),
                                    verify_peers=False, **config).check()
        self.clock = FakeClock()
        self.units = {}  # pid -> systemd unit
        self.next_pid = 1000
        self.console = self.peer()
        self.service = build_service(self.config, Database(self.config.db_path, self.clock),
                                     self.clock, Passwords(password_hasher(fast=True)),
                                     self.units.get)
        self.db = self.service.db

    def peer(self, display: Optional[int] = None) -> Peer:
        self.next_pid += 1
        if display is not None:
            self.units[self.next_pid] = self.config.desktop_unit_of(display)
        return Peer(self.next_pid, os.getuid(), os.getgid())

    def raw(self, op: str, token: Optional[str] = None, peer: Optional[Peer] = None,
            request_id: Optional[str] = None, **args) -> dict:
        request = {'op': op, 'args': args}
        if token is not None:
            request['token'] = token
        if request_id is not None:
            request['request_id'] = request_id
        return self.service.handle(request, peer or self.console)

    def call(self, op: str, token: Optional[str] = None, peer: Optional[Peer] = None, **args):
        """The result; raises ClientError with the refusal's code."""
        reply = self.raw(op, token, peer, **args)
        if not reply['ok']:
            details = dict(reply['error'])
            raise ClientError(details.pop('code'), details.pop('message'), details)
        return reply['result']

    def code(self, op: str, token: Optional[str] = None, peer: Optional[Peer] = None,
             **args) -> str:
        """'ok', or the refusal's code."""
        reply = self.raw(op, token, peer, **args)
        return 'ok' if reply['ok'] else reply['error']['code']

    # users
    def owner(self, login: str = 'owner', peer: Optional[Peer] = None) -> str:
        """Bootstraps the owner and returns a session token of theirs."""
        from cutieplus.cooperation.admin_cli import bootstrap_owner
        setup = bootstrap_owner(self.service, login, login.title())
        return self.call('setup_credential', peer=peer, login_name=login, setup_token=setup,
                         new_password=PASSWORD)['token']

    def add(self, manager: str, login: str, role: Optional[str] = None,
            peer: Optional[Peer] = None) -> dict:
        """Creates a user through the API and sets up their password: {'id', 'token', ...}."""
        args = {'login_name': login, 'display_name': login.title()}
        if role is not None:
            args['role'] = role
        created = self.call('create_user', manager, **args)
        session = self.call('setup_credential', peer=peer, login_name=login,
                            setup_token=created['setup_token'], new_password=PASSWORD)
        return {'id': created['user']['id'], 'token': session['token'], 'created': created}

    def login(self, login: str, peer: Optional[Peer] = None) -> str:
        return self.call('login', peer=peer, login_name=login, password=PASSWORD)['token']

    def add_slot(self, user_id: str, display: int) -> str:
        """Puts the user's desktop at that display (a new user gets the lowest free one)."""
        from cutieplus.cooperation.clock import new_id
        now = self.service.now()
        with self.db.transaction() as cursor:
            rows = cursor.execute('SELECT id, display_number FROM desktop_slots WHERE user_id = ?',
                                  (user_id, )).fetchall()
            if rows and rows[0]['display_number'] == display:
                return rows[0]['id']
            slot_id = rows[0]['id'] if rows else new_id()
            cursor.execute('DELETE FROM desktop_slots WHERE id = ?', (slot_id, ))
            cursor.execute(
                'INSERT INTO desktop_slots (id, user_id, display_number, vnc_port, '
                'workspace_path, service_name, desired_state, observed_state, created_at, '
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, 'running', 'running', ?, ?)",
                (slot_id, user_id, display, self.config.vnc_port_of(display),
                 self.config.default_tasks_dir, self.config.desktop_unit_of(display), now, now))
        return slot_id


# the root helper, with a fake system: units, processes and sshd's journal entries are simulated;
# files (the policy, VNC passwords, environment files) are really written, below the test's folder

import base64
import struct
import subprocess

from cutieplus.cooperation.helper import SYSTEMCTL, VNCPASSWD, Helper, System
from cutieplus.cooperation.procinfo import ProcessIdentity

BOOT = 'fake-boot'


def ed25519_key(seed: int) -> str:
    """A well-formed ed25519 public key line."""
    blob = b''.join(struct.pack('>I', len(part)) + part
                    for part in (b'ssh-ed25519', bytes([seed % 256]) * 32))
    return f'ssh-ed25519 {base64.b64encode(blob).decode()} someone@laptop'


class FakeSystem(System):

    def __init__(self):
        self.units = {}  # unit -> pid (0: stopped)
        self.processes = {}  # pid -> (identity, unit)
        self.next_pid = 20000
        self.failing = {}  # 'start' | 'stop' | 'write' | 'vncpasswd' -> message
        self.killed = []
        self.commands = []

    def spawn(self, unit: str) -> ProcessIdentity:
        self.next_pid += 1
        identity = ProcessIdentity(self.next_pid, 100, BOOT)
        self.processes[self.next_pid] = (identity, unit)
        return identity

    def run(self, argv, stdin=None):
        self.commands.append(list(argv))
        if argv[0] == VNCPASSWD:
            if 'vncpasswd' in self.failing:
                return subprocess.CompletedProcess(argv, 1, b'', b'failed')
            return subprocess.CompletedProcess(argv, 0, b'OBFUSCATED:' + stdin.strip(), b'')
        assert argv[0] == SYSTEMCTL
        action, unit = argv[1], argv[-1] if argv[1] != 'show' else argv[2]
        if action in ('start', 'stop') and action in self.failing:
            return subprocess.CompletedProcess(argv, 1, b'', self.failing[action].encode())
        if action == 'start' and not self.units.get(unit):
            self.units[unit] = self.spawn(unit).pid
        elif action == 'stop' and self.units.get(unit):
            self.processes.pop(self.units[unit], None)
            self.units[unit] = 0
        pid = self.units.get(unit, 0)
        state = 'active' if pid else 'inactive'
        return subprocess.CompletedProcess(argv, 0,
                                           f'ActiveState={state}\nMainPID={pid}\n'.encode(), b'')

    def write_atomic(self, file_path, data, mode):
        if 'write' in self.failing and file_path.endswith('authorized_keys'):
            raise OSError(self.failing['write'])
        super().write_atomic(file_path, data, mode)

    def kill(self, pid, number):
        self.killed.append(pid)
        self.processes.pop(pid, None)

    def identity(self, pid):
        entry = self.processes.get(pid)
        return entry[0] if entry else None

    def unit_of(self, pid):
        entry = self.processes.get(pid)
        return entry[1] if entry else None

    def journal(self, unit, follow):
        return iter(())


class InProcessHelper:
    """The helper's client, calling it directly as the service account would."""

    def __init__(self, helper: Helper):
        self.helper = helper
        self.down = False

    def call(self, op: str, **args):
        from cutieplus.cooperation.client import ServiceUnavailable
        if self.down:
            raise ServiceUnavailable('the helper is not running')
        reply = self.helper.handle({'op': op, 'args': args},
                                   Peer(1, self.helper.service_uid, 0))
        if not reply['ok']:
            details = dict(reply['error'])
            raise ClientError(details.pop('code'), details.pop('message'), details)
        return reply['result']


class AccessHarness(Harness):
    """The service with the root helper (fake system) and its reconciler."""

    def __init__(self, tmp_path, **config):
        system_dir = tmp_path / 'system'
        system_dir.mkdir()
        super().__init__(tmp_path, authorized_keys=str(system_dir / 'ssh' / 'authorized_keys'),
                         desktops_dir=str(system_dir / 'desktops'), **config)
        self.system = FakeSystem()
        self.helper = Helper(self.config, self.system, service_uid=os.getuid())
        self.helper_client = InProcessHelper(self.helper)
        from cutieplus.cooperation.operations import Reconciler
        self.service.reconciler = Reconciler(self.service, self.helper_client)

    def reconcile(self, passes: int = 2) -> None:
        for _ in range(passes):
            self.service.reconciler.run_once()

    def policy_text(self) -> str:
        with open(self.config.authorized_keys, encoding='utf-8') as f:
            return f.read()

    def connect(self, public_key: str, address: str = '203.0.113.7', port: int = 50022):
        """An SSH sign-in the annotation sshd accepted, as its journal records it."""
        from cutieplus.cooperation.sshpolicy import parse_public_key
        identity = self.system.spawn(self.config.sshd_unit)
        fingerprint = parse_public_key(public_key).fingerprint
        self.helper.tracker.entry({
            '_SYSTEMD_UNIT': self.config.sshd_unit, '_UID': '0', '_PID': str(identity.pid),
            '__MONOTONIC_TIMESTAMP': '10000000', '__REALTIME_TIMESTAMP': '1800000000000000',
            'MESSAGE': f'Accepted publickey for {self.config.desktop_user} from {address} port '
                       f'{port} ssh2: ED25519 {fingerprint}'})
        return identity

    def operations(self, kind: str) -> list:
        return self.db.query('SELECT * FROM management_operations WHERE operation_type = ? '
                             'ORDER BY created_at', (kind, ))

    def unit_running(self, display: int) -> bool:
        return bool(self.system.units.get(self.config.desktop_unit_of(display)))
