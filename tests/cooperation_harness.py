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
        """A desktop slot as the desktops part will allocate it."""
        from cutieplus.cooperation.clock import new_id
        slot_id = new_id()
        now = self.service.now()
        with self.db.transaction() as cursor:
            cursor.execute(
                'INSERT INTO desktop_slots (id, user_id, display_number, vnc_port, '
                'workspace_path, service_name, desired_state, observed_state, created_at, '
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, 'running', 'running', ?, ?)",
                (slot_id, user_id, display, self.config.vnc_port_of(display),
                 self.config.workspace_root, self.config.desktop_unit_of(display), now, now))
        return slot_id
