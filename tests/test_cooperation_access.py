"""
Remote access (cutieplus/cooperation/keys.py, desktops.py, operations.py, helper.py, sshpolicy.py):
keys checked and published as a restricted SSH policy, desktops started and stopped toward their
desired state, connections tracked from sshd's journal and cut when access ends, and operations
that say what is still to do until the system has done it. The root helper runs with a fake
system; the policy and VNC password files are really written.
"""
import base64
import os
import struct
import subprocess

import pytest

from cutieplus.cooperation import sshpolicy
from cutieplus.cooperation.procinfo import Peer, ProcessIdentity
from tests.cooperation_harness import AccessHarness, ed25519_key


@pytest.fixture
def access(tmp_path):
    return AccessHarness(tmp_path)


def onboard(access, owner, login='alice', role=None, seed=1):
    """A user with a desktop and a key, the system reconciled."""
    user = access.add(owner, login, role)
    key = ed25519_key(seed)
    added = access.call('add_key', owner, user_id=user['id'], public_key=key,
                        device_label=f'{login} laptop')
    access.reconcile()
    return dict(user, key=key, key_id=added['key']['id'])


def blob(*parts: bytes) -> str:
    return base64.b64encode(b''.join(struct.pack('>I', len(part)) + part
                                     for part in parts)).decode()


def test_keys_are_checked_and_fingerprinted_as_openssh_does(tmp_path):
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', 'alice@laptop', '-f',
                    str(tmp_path / 'id')], check=True)
    line = (tmp_path / 'id.pub').read_text()
    shown = subprocess.run(['ssh-keygen', '-l', '-f', str(tmp_path / 'id.pub')], check=True,
                           capture_output=True, text=True).stdout.split()[1]
    key = sshpolicy.parse_public_key(line)
    assert key.fingerprint == shown and key.canonical == ' '.join(line.split()[:2])
    refused = [
        (tmp_path / 'id').read_text(),  # the private key
        'command="/bin/sh" ' + line,  # options in front
        line + '\nssh-ed25519 AAAA second',  # two lines
        'ssh-dss AAAAB3NzaC1kc3MAAACBAP= x',  # an old type
        'ssh-ed25519 not-base64!',
        f'ssh-ed25519 {blob(b"ssh-rsa", b"x" * 32)}',  # the data names another type
        f'ssh-ed25519 {blob(b"ssh-ed25519", b"x" * 31)}',  # cut short
        f'ssh-rsa {blob(b"ssh-rsa", bytes([1, 0, 1]), b"\x01" * 256)}',  # RSA 2048
    ]
    for text in refused:
        with pytest.raises(sshpolicy.KeyError_):
            sshpolicy.parse_public_key(text)
    rsa = sshpolicy.parse_public_key(f'ssh-rsa {blob(b"ssh-rsa", bytes([1, 0, 1]), b"\xff" * 384)}')
    assert rsa.bits == 3072


def test_a_policy_line_forwards_to_its_own_port_only():
    key = sshpolicy.parse_public_key(ed25519_key(7))
    entry = sshpolicy.PolicyEntry('0b9c2a4e-8f1d-4c3b-9a7e-5d6f7a8b9c0d', key.canonical, 5902,
                                  '2027-06-30T23:59:59.999999Z')
    assert sshpolicy.render_line(entry, (5902, 5920)) == (
        'restrict,port-forwarding,permitopen="localhost:5902",permitopen="127.0.0.1:5902",'
        'command="/usr/sbin/nologin",expiry-time="202706302359Z" '
        f'{key.canonical} cutieplus:0b9c2a4e-8f1d-4c3b-9a7e-5d6f7a8b9c0d')
    for bad in (entry.__class__(entry.key_id, key.canonical, 22, None),  # not a VNC port
                entry.__class__('../etc', key.canonical, 5902, None),
                entry.__class__(entry.key_id, key.canonical + ' evil', 5902, None),
                entry.__class__(entry.key_id, 'restrict ' + key.canonical, 5902, None)):
        with pytest.raises(sshpolicy.KeyError_):
            sshpolicy.render_line(bad, (5902, 5920))


def test_onboarding_gives_a_desktop_and_publishes_the_key(access):
    owner = access.owner()
    created = access.call('create_user', owner, login_name='alice', display_name='Alice')
    card = created['desktop']
    assert (card['display'], card['vnc_port'], card['ssh_port']) == (2, 5902, 2222)
    assert len(card['vnc_password']) == 8 and '-p 2222' in card['ssh_command']
    access.reconcile()
    assert access.unit_running(2)
    password_file = access.config.vnc_password_file(2)
    assert open(password_file, 'rb').read() == b'OBFUSCATED:' + card['vnc_password'].encode()
    assert os.stat(password_file).st_mode & 0o777 == 0o600
    env = open(os.path.join(access.config.desktops_dir, '2', 'env')).read()
    assert f'CUTIEPLUS_TASKS={access.config.default_tasks_dir}' in env
    assert [op['status'] for op in access.operations('provision_user')] == ['succeeded']
    # the VNC password is in no table
    assert card['vnc_password'] not in repr([access.db.query(f'SELECT * FROM {table}') for table
                                             in ('desktop_slots', 'management_operations',
                                                 'audit_events')])

    user_id = created['user']['id']
    added = access.call('add_key', owner, user_id=user_id, public_key=ed25519_key(1),
                        device_label='Laptop', expires_at='2027-06-30')
    assert added['key']['fingerprint_sha256'].startswith('SHA256:')
    assert access.operations('add_key')[0]['status'] == 'pending'
    access.reconcile()
    text = access.policy_text()
    assert 'permitopen="localhost:5902"' in text and 'expiry-time="202706302359Z"' in text
    assert 'someone@laptop' not in text  # the key's own comment is dropped
    assert access.operations('add_key')[0]['status'] == 'succeeded'
    state = access.db.query_one('SELECT * FROM policy_state')
    assert state['applied_revision'] == state['desired_revision'] > 0
    assert state['applied_snapshot_checksum'] == sshpolicy.checksum(text)
    details = access.call('get_user', owner, user_id=user_id)
    assert details['desktop']['observed_state'] == 'running'
    assert [key['device_label'] for key in details['keys']] == ['Laptop']


def test_a_key_is_one_users_and_a_revoked_key_never_returns(access):
    owner = access.owner()
    alice = onboard(access, owner)
    bob = access.add(owner, 'bob')
    assert access.code('add_key', owner, user_id=bob['id'], public_key=alice['key'],
                       device_label='x') == 'conflict'
    access.call('revoke_key', owner, key_id=alice['key_id'])
    assert access.code('add_key', owner, user_id=alice['id'], public_key=alice['key'],
                       device_label='again') == 'conflict'
    # who may: supervisors for annotators only, annotators for nobody, themselves included
    supervisor = access.add(owner, 'sam', 'supervisor')
    assert access.code('add_key', supervisor['token'], user_id=bob['id'],
                       public_key=ed25519_key(2), device_label='Bob') == 'ok'
    assert access.code('add_key', supervisor['token'], user_id=supervisor['id'],
                       public_key=ed25519_key(3), device_label='mine') == 'forbidden'
    assert access.code('add_key', bob['token'], user_id=bob['id'], public_key=ed25519_key(4),
                       device_label='mine') == 'forbidden'
    assert access.code('add_key', owner, user_id=bob['id'], public_key='-----BEGIN OPENSSH '
                       'PRIVATE KEY-----', device_label='x') == 'invalid'
    # a revocation is checked against the owner of the key as they are now
    sams_key = access.call('add_key', owner, user_id=supervisor['id'], public_key=ed25519_key(5),
                           device_label='Sam')['key']['id']
    other = access.add(owner, 'sue', 'supervisor')
    assert access.code('revoke_key', other['token'], key_id=sams_key) == 'forbidden'


def test_revoking_a_key_blocks_it_and_cuts_its_connections(access):
    owner = access.owner()
    alice = onboard(access, owner)
    second = access.call('add_key', owner, user_id=alice['id'], public_key=ed25519_key(9),
                         device_label='Desktop PC')['key']['id']
    access.reconcile()
    connection = access.connect(alice['key'])
    other = access.connect(ed25519_key(9))
    access.reconcile()
    rows = access.db.query("SELECT * FROM ssh_connections WHERE status = 'active'")
    assert len(rows) == 2 and {row['user_id'] for row in rows} == {alice['id']}
    connected = access.db.query("SELECT * FROM audit_events WHERE action = 'ssh_connected'")
    assert {row['actor_source'] for row in connected} == {'verified_sshd'}

    access.call('revoke_key', owner, key_id=alice['key_id'])
    access.reconcile()
    text = access.policy_text()
    assert alice['key_id'] not in text and second in text  # the other key still works
    assert connection.pid in access.system.killed and other.pid not in access.system.killed
    closed = access.db.query("SELECT * FROM ssh_connections WHERE status = 'closed'")
    assert [row['disconnect_reason'] for row in closed] == ['key revoked']
    assert access.operations('revoke_key')[0]['status'] == 'succeeded'
    assert access.unit_running(2)  # the desktop and its work stay


def test_a_failed_publication_is_shown_and_cuts_nothing_yet(access):
    owner = access.owner()
    alice = onboard(access, owner)
    access.connect(alice['key'])
    access.reconcile()
    access.system.failing['write'] = 'read-only file system'
    access.call('revoke_key', owner, key_id=alice['key_id'])
    access.reconcile()
    operation = access.operations('revoke_key')[0]
    assert operation['status'] == 'failed' and 'read-only' in operation['error_summary']
    assert operation['current_step'] == 'publishing the SSH policy'
    assert 'read-only' in access.db.query_one('SELECT * FROM policy_state')['last_apply_error']
    # the old policy still lets the key in: cutting its connection now would only see it back
    assert access.system.killed == []
    del access.system.failing['write']
    access.reconcile()
    assert access.operations('revoke_key')[0]['status'] == 'succeeded'
    assert alice['key_id'] not in access.policy_text() and access.system.killed


def test_suspending_stops_access_and_the_desktop_and_reactivating_brings_them_back(access):
    owner = access.owner()
    alice = onboard(access, owner)
    connection = access.connect(alice['key'])
    access.reconcile()
    access.call('suspend_user', owner, user_id=alice['id'])
    access.reconcile()
    assert alice['key_id'] not in access.policy_text()
    assert connection.pid in access.system.killed and not access.unit_running(2)
    assert access.operations('suspend_user')[0]['status'] == 'succeeded'
    access.call('reactivate_user', owner, user_id=alice['id'])
    access.reconcile()
    assert alice['key_id'] in access.policy_text() and access.unit_running(2)
    assert access.operations('reactivate_user')[0]['status'] == 'succeeded'


def test_the_newest_request_wins_over_one_not_yet_applied(access):
    owner = access.owner()
    alice = onboard(access, owner)
    access.call('suspend_user', owner, user_id=alice['id'])
    access.call('reactivate_user', owner, user_id=alice['id'])  # before any pass
    access.reconcile()
    assert [op['status'] for op in access.operations('suspend_user')] == ['superseded']
    assert access.operations('reactivate_user')[0]['status'] == 'succeeded'
    assert access.unit_running(2) and not any(
        command[1] == 'stop' for command in access.system.commands if len(command) > 1)


def test_removing_revokes_access_keeps_the_records_and_frees_the_desktop_last(access):
    owner = access.owner()
    alice = onboard(access, owner)
    connection = access.connect(alice['key'])
    access.reconcile()
    access.call('remove_user', owner, user_id=alice['id'])
    # until her desktop is released, its display is not given again
    assert access.call('create_user', owner, login_name='bob',
                       display_name='Bob')['desktop']['display'] == 3
    access.reconcile()
    assert connection.pid in access.system.killed and not access.unit_running(2)
    assert access.db.query_one('SELECT * FROM desktop_slots WHERE user_id = ?',
                               (alice['id'], )) is None
    assert not os.path.exists(access.config.vnc_password_file(2))
    assert access.operations('remove_user')[0]['status'] == 'succeeded'
    keys = access.db.query('SELECT status FROM ssh_keys WHERE user_id = ?', (alice['id'], ))
    assert [key['status'] for key in keys] == ['revoked']  # kept, revoked
    assert access.call('create_user', owner, login_name='carol',
                       display_name='Carol')['desktop']['display'] == 2


def test_disconnecting_keeps_the_key_and_the_desktop(access):
    owner = access.owner()
    alice = onboard(access, owner)
    connection = access.connect(alice['key'])
    access.reconcile()
    row = access.db.query_one("SELECT * FROM ssh_connections WHERE status = 'active'")
    access.call('disconnect', owner, connection_id=row['id'])
    access.reconcile()
    assert connection.pid in access.system.killed
    closed = access.db.query_one('SELECT * FROM ssh_connections WHERE id = ?', (row['id'], ))
    assert closed['disconnect_reason'] == 'disconnected by owner'
    assert alice['key_id'] in access.policy_text() and access.unit_running(2)


def test_an_expired_key_leaves_the_policy_and_its_connections_end(access):
    owner = access.owner()
    user = access.add(owner, 'alice')
    key = ed25519_key(3)
    access.call('add_key', owner, user_id=user['id'], public_key=key, device_label='Laptop',
                expires_at=access.clock.iso_in(days=1))
    access.reconcile()
    connection = access.connect(key)
    access.reconcile()
    access.clock.advance(days=1, seconds=1)
    access.reconcile()
    assert 'cutieplus:' not in access.policy_text()
    assert connection.pid in access.system.killed
    reason = access.db.query_one('SELECT disconnect_reason FROM ssh_connections')
    assert reason['disconnect_reason'] == 'key expired'


def test_a_desktop_that_fails_to_start_is_reported(access):
    owner = access.owner()
    access.system.failing['start'] = 'Xtigervnc: port 5902 in use'
    user = access.add(owner, 'alice')
    access.reconcile()
    operation = access.operations('provision_user')[0]
    assert operation['status'] == 'failed' and 'port 5902 in use' in operation['error_summary']
    slot = access.call('get_user', owner, user_id=user['id'])['desktop']
    assert 'port 5902 in use' in slot['last_error']
    del access.system.failing['start']
    access.reconcile()
    assert access.operations('provision_user')[0]['status'] == 'succeeded'


def test_a_new_vnc_password_restarts_the_desktop(access):
    owner = access.owner()
    alice = onboard(access, owner)
    before = access.system.units[access.config.desktop_unit_of(2)]
    reset = access.call('reset_vnc_password', owner, user_id=alice['id'])
    access.reconcile(passes=3)
    password = reset['desktop']['vnc_password']
    assert open(access.config.vnc_password_file(2), 'rb').read().endswith(password.encode())
    assert access.system.units[access.config.desktop_unit_of(2)] not in (0, before)
    assert {op['status'] for op in access.operations('restart_desktop')} == {'succeeded'}


def test_the_helper_answers_the_service_only_and_checks_every_argument(access):
    helper = access.helper
    stranger = Peer(1, os.getuid() + 1, 0)
    assert helper.handle({'op': 'hello', 'args': {}}, stranger)['error']['code'] == 'forbidden'
    me = Peer(1, os.getuid(), 0)

    def code(op, **args):
        reply = helper.handle({'op': op, 'args': args}, me)
        return 'ok' if reply['ok'] else reply['error']['code']

    assert code('desktop_start', display=1) == 'invalid'  # outside :2-:20
    assert code('desktop_start', display='2; reboot') == 'invalid'
    assert code('configure_desktop', display=2, tasks_dir='/etc') == 'invalid'
    assert code('configure_desktop', display=2,
                tasks_dir=access.config.workspace_root + '/../../etc') == 'invalid'
    assert code('write_vnc_password', display=2, password='a b c d e') == 'invalid'
    assert code('shell', command='id') == 'bad_request'
    assert code('desktop_stop', display=2, unit='ssh.service') == 'bad_request'
    # a process the annotation sshd did not accept is never signalled
    victim = access.system.spawn('ssh.service')
    assert helper.handle({'op': 'disconnect', 'args': {'identity': victim.as_dict()}},
                         me)['result'] == {'closed': True, 'already': True}
    assert victim.pid not in access.system.killed
    # nor one that took over the PID of a connection
    connection = access.connect(ed25519_key(1))
    access.system.processes[connection.pid] = (ProcessIdentity(connection.pid, 999, 'fake-boot'),
                                               access.config.sshd_unit)
    helper.handle({'op': 'disconnect', 'args': {'identity': connection.as_dict()}}, me)
    assert connection.pid not in access.system.killed


def test_journal_entries_count_only_from_the_annotation_sshd_as_root(access):
    tracker = access.helper.tracker
    identity = access.system.spawn(access.config.sshd_unit)
    message = (f'Accepted publickey for zexuan from 198.51.100.4 port 4000 ssh2: ED25519 '
               f'{sshpolicy.parse_public_key(ed25519_key(1)).fingerprint}')
    base = {'_PID': str(identity.pid), '__MONOTONIC_TIMESTAMP': '10000000',
            '__REALTIME_TIMESTAMP': '1', 'MESSAGE': message}
    tracker.entry(dict(base, _SYSTEMD_UNIT='ssh.service', _UID='0'))  # the owner's sshd
    tracker.entry(dict(base, _SYSTEMD_UNIT=access.config.sshd_unit, _UID='1000'))  # not root
    tracker.entry(dict(base, _SYSTEMD_UNIT=access.config.sshd_unit, _UID='0',
                       MESSAGE=message.replace('zexuan', 'root')))
    assert tracker.alive() == []
    tracker.entry(dict(base, _SYSTEMD_UNIT=access.config.sshd_unit, _UID='0'))
    assert [connection.address for connection in tracker.alive()] == ['198.51.100.4']


def test_a_removed_users_desktop_is_kept_while_the_policy_may_still_admit_them(access):
    owner = access.owner()
    alice = onboard(access, owner)
    access.system.failing['write'] = 'disk full'
    access.call('remove_user', owner, user_id=alice['id'])
    access.reconcile()
    # her key may still be in the live file, forwarding to port 5902: :2 is not given again
    assert not access.unit_running(2)
    assert access.db.query_one('SELECT * FROM desktop_slots WHERE user_id = ?', (alice['id'], ))
    assert access.call('create_user', owner, login_name='bob',
                       display_name='Bob')['desktop']['display'] == 3
    del access.system.failing['write']
    access.reconcile()
    assert access.db.query_one('SELECT * FROM desktop_slots WHERE user_id = ?',
                               (alice['id'], )) is None


def test_a_request_id_is_its_callers_own(access):
    owner = access.owner()
    supervisor = access.add(owner, 'sam', 'supervisor')
    alice = access.add(owner, 'alice')
    first = access.raw('add_key', owner, request_id='r1', user_id=alice['id'],
                       public_key=ed25519_key(1), device_label='Laptop')
    again = access.raw('add_key', owner, request_id='r1', user_id=alice['id'],
                       public_key=ed25519_key(1), device_label='Laptop')
    assert first['ok'] and again['result']['repeated']  # applied once
    # another caller with the same id gets their own request, not the first one's operation
    theirs = access.raw('add_key', supervisor['token'], request_id='r1', user_id=alice['id'],
                        public_key=ed25519_key(2), device_label='Desktop')
    assert theirs['ok'] and 'repeated' not in theirs['result']
    # nor does an unauthorized caller learn anything from it
    bob = access.add(owner, 'bob')
    refused = access.raw('add_key', bob['token'], request_id='r1', user_id=alice['id'],
                         public_key=ed25519_key(3), device_label='x')
    assert refused['error']['code'] == 'forbidden'
