"""
Management operations whose effect spans the database and the operating system, and the
reconciler that carries them out.

A request (suspend a user, revoke a key, stop a desktop) changes the desired state in the database
and records a management_operations row in the same transaction. The reconciler then brings the
system to the desired state as it is NOW, through the root helper: it never replays an old
imperative step, so a suspension followed by a reactivation before either was applied ends with a
running desktop, and the suspension is marked superseded. An operation is succeeded only once
everything it needs has been observed (the policy file written and read back, the connections
gone, the desktop's unit stopped); until then the dialog shows it running, or failed with the
helper's error while the reconciler keeps trying.

Each pass of the reconciler:
1. publishes the SSH policy when its desired revision is ahead of the applied one, or when what was
   applied no longer matches the database (a key expired);
2. records the connections the annotation sshd accepted, and closes those that ended;
3. once the policy is current, cuts the connections of revoked or expired keys and of users who
   may not sign in, and those a manager asked to disconnect;
4. writes pending VNC passwords and starts or stops each desktop toward its desired state;
5. releases the desktops of removed users once they are stopped and unconnected;
6. deletes the users an admin deleted for good, once nothing of theirs is left running;
7. settles the operations.
"""
import json
import logging
import os
import secrets
import threading
import time
from typing import Dict, List, Optional

from cutieplus.cooperation import audit, policy
from cutieplus.cooperation.client import ClientError
from cutieplus.cooperation.clock import new_id
from cutieplus.cooperation.errors import LIMIT, Refused
from cutieplus.cooperation.procinfo import ProcessIdentity

log = logging.getLogger(__name__)

PENDING, RUNNING, SUCCEEDED, FAILED, SUPERSEDED = ('pending', 'running', 'succeeded', 'failed',
                                                   'superseded')
OPEN = (PENDING, RUNNING, FAILED)
# a newer operation of a group supersedes the open ones of the same group on the same user
GROUPS = {
    'provision_user': 'lifecycle', 'suspend_user': 'lifecycle', 'reactivate_user': 'lifecycle',
    'remove_user': 'lifecycle', 'delete_user': 'lifecycle', 'start_desktop': 'desktop',
    'stop_desktop': 'desktop',
    'restart_desktop': 'desktop',
}
SUPERSEDES = {'lifecycle': ('lifecycle', 'desktop'), 'desktop': ('desktop', )}
VNC_ALPHABET = 'abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'
DESKTOP_RETRY_S = 60  # a desktop that failed to start is tried again after this


def new_vnc_password() -> str:
    return ''.join(secrets.choice(VNC_ALPHABET) for _ in range(8))


def start_operation(service, cursor, operation_type: str, actor_id: Optional[str],
                    target_user_id: Optional[str] = None,
                    target_resource_id: Optional[str] = None, request_id: Optional[str] = None,
                    desired_revision: Optional[int] = None,
                    parameters: Optional[dict] = None) -> str:
    now = service.now()
    group = GROUPS.get(operation_type)
    if group is not None and target_user_id is not None:
        older = [row for row in cursor.execute(
            f"SELECT id, operation_type FROM management_operations WHERE target_user_id = ? "
            f"AND status IN ({','.join('?' * len(OPEN))})", (target_user_id, *OPEN)).fetchall()
            if GROUPS.get(row['operation_type']) in SUPERSEDES[group]]
        for row in older:
            cursor.execute("UPDATE management_operations SET status = 'superseded', "
                           'updated_at = ?, completed_at = ?, current_step = NULL WHERE id = ?',
                           (now, now, row['id']))
    operation_id = new_id()
    cursor.execute(
        'INSERT INTO management_operations (id, operation_type, actor_user_id, target_user_id, '
        'target_resource_id, request_id, desired_revision, validated_parameters_json, status, '
        'current_step, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (operation_id, operation_type, actor_id, target_user_id, target_resource_id,
         scoped_request_id(actor_id, request_id),
         desired_revision, json.dumps(parameters or {}, sort_keys=True), PENDING, 'waiting', now,
         now))
    service.reconcile_soon()
    return operation_id


def scoped_request_id(actor_id: Optional[str], request_id: Optional[str]) -> Optional[str]:
    """A request id is the caller's own: two users choosing the same one never meet."""
    if request_id is None:
        return None
    return f'{actor_id or "system"}/{request_id}'


def repeated_operation(cursor, actor_id: str, request_id: Optional[str],
                       operation_type: str) -> Optional[dict]:
    """The operation this caller's request with this id already made (a retried request is
    applied once). Only asked once the request itself is authorized."""
    if request_id is None:
        return None
    rows = cursor.execute('SELECT * FROM management_operations WHERE request_id = ? AND '
                          'actor_user_id = ? AND operation_type = ?',
                          (scoped_request_id(actor_id, request_id), actor_id,
                           operation_type)).fetchall()
    return rows[0] if rows else None


def public_operation(row: dict) -> dict:
    return {key: row[key] for key in ('id', 'operation_type', 'actor_user_id', 'target_user_id',
                                      'target_resource_id', 'status', 'current_step',
                                      'created_at', 'updated_at', 'completed_at',
                                      'error_summary')}


# the lifecycle of a user: what the rest of the system follows

def allocate_desktop(service, cursor, user: dict) -> dict:
    """The lowest free display of the managed range; its VNC password waits in memory for the
    reconciler (it is never stored in the database). Returns what the caller must see once."""
    used = {row['display_number'] for row in cursor.execute(
        'SELECT display_number FROM desktop_slots').fetchall()}
    free = [display for display in service.config.displays() if display not in used]
    if not free:
        raise Refused(LIMIT, f'every remote desktop (:{service.config.first_display}-'
                      f':{service.config.last_display}) is taken')
    display = free[0]
    slot_id = new_id()
    now = service.now()
    cursor.execute(
        'INSERT INTO desktop_slots (id, user_id, display_number, vnc_port, workspace_path, '
        'service_name, desired_state, observed_state, created_at, updated_at) '
        "VALUES (?, ?, ?, ?, ?, ?, 'running', 'stopped', ?, ?)",
        (slot_id, user['id'], display, service.config.vnc_port_of(display),
         service.config.default_tasks_dir, service.config.desktop_unit_of(display), now, now))
    password = new_vnc_password()
    # for the reconciler, once the transaction is committed (Service.after_commit)
    service.after_commit(lambda: service.pending_vnc_passwords.__setitem__(slot_id, password))
    return desktop_card(service, display, password)


def desktop_card(service, display: int, password: Optional[str] = None) -> dict:
    """How the user reaches their desktop: shown to the manager to hand on."""
    config = service.config
    port = config.vnc_port_of(display)
    # through port 22 (the jump), to the annotation sshd on this computer, then the VNC port,
    # forwarded from the same port on their computer: a VNC server there (5900, 5901) or another
    # desktop's tunnel does not collide; a port in use there ends ssh with the error, rather than
    # leaving it connected without the tunnel
    jump = config.public_host + ('' if config.jump_port == 22 else f':{config.jump_port}')
    card = {'display': display, 'vnc_port': port, 'ssh_port': config.ssh_port,
            'ssh_command': f'ssh -N -o ExitOnForwardFailure=yes -J {config.jump_user}@{jump} '
                           f'-p {config.ssh_port} -L {port}:localhost:{port} '
                           f'{config.desktop_user}@localhost',
            'vnc_address': f'localhost:{port}'}
    if password is not None:
        card['vnc_password'] = password
    return card


def on_user_change(service, cursor, before: Optional[dict], after: dict, actor_id, now,
                   result: dict) -> None:
    """A user hook (Service.user_hooks): desktops, keys and the SSH policy follow the user."""
    slot = (cursor.execute('SELECT * FROM desktop_slots WHERE user_id = ?',
                           (after['id'], )).fetchall() or [None])[0]
    if before is None:
        if after['role'] in ('annotator', 'supervisor'):
            result['desktop'] = allocate_desktop(service, cursor, after)
            start_operation(service, cursor, 'provision_user', actor_id, after['id'])
        return
    if before['status'] == after['status']:
        return
    if after['status'] == 'suspended':
        if slot is not None:
            cursor.execute("UPDATE desktop_slots SET desired_state = 'stopped', updated_at = ? "
                           'WHERE id = ?', (now, slot['id']))
        start_operation(service, cursor, 'suspend_user', actor_id, after['id'],
                        desired_revision=policy.bump(cursor))
    elif after['status'] == 'archived':
        cursor.execute("UPDATE ssh_keys SET status = 'revoked', revoked_by = ?, revoked_at = ? "
                       "WHERE user_id = ? AND status != 'revoked'", (actor_id, now, after['id']))
        if slot is not None:
            cursor.execute("UPDATE desktop_slots SET desired_state = 'stopped', updated_at = ? "
                           'WHERE id = ?', (now, slot['id']))
        start_operation(service, cursor, 'remove_user', actor_id, after['id'],
                        desired_revision=policy.bump(cursor))
    elif before['status'] == 'suspended':
        if slot is not None:
            cursor.execute("UPDATE desktop_slots SET desired_state = 'running', updated_at = ? "
                           'WHERE id = ?', (now, slot['id']))
        start_operation(service, cursor, 'reactivate_user', actor_id, after['id'],
                        desired_revision=policy.bump(cursor))
    for hook in service.suspension_hooks if after['status'] in ('suspended', 'archived') else ():
        hook(cursor, after, actor_id, now)


def purge_user(service, cursor, operation: dict) -> List[str]:
    """Deletes a removed user and what is theirs: their GPU jobs, connections, keys, sessions and
    operations. The audit log keeps every entry (those of what they did keep their login name),
    and other users' records that name them (who created a user or a key, revoked a key,
    cancelled a job, ran an operation, changed a setting) keep the rest and name nobody there.
    Returns the files of their jobs, to remove once this is committed."""
    user_id = operation['target_user_id']
    user = cursor.execute('SELECT * FROM users WHERE id = ?', (user_id, )).fetchall()[0]
    jobs = cursor.execute('SELECT id, input_reference FROM gpu_jobs WHERE owner_user_id = ?',
                          (user_id, )).fetchall()
    results = service.config.results_dir
    files = [os.path.join(results, f'{job["id"]}{suffix}') for job in jobs
             for suffix in ('.jsonl', '.json')]
    files += [job['input_reference'] for job in jobs  # a retry shares its original's snapshot
              if os.path.dirname(job['input_reference']) == service.config.snapshots_dir]
    theirs = 'SELECT id FROM gpu_jobs WHERE owner_user_id = ?'
    for sql in (f'UPDATE gpu_state SET active_job_id = NULL WHERE active_job_id IN ({theirs})',
                f'UPDATE gpu_jobs SET retry_of_job_id = NULL WHERE retry_of_job_id IN ({theirs})',
                'UPDATE gpu_jobs SET cancelled_by = NULL WHERE cancelled_by = ?',
                'DELETE FROM gpu_jobs WHERE owner_user_id = ?',
                'DELETE FROM ssh_connections WHERE user_id = ?',
                'UPDATE ssh_keys SET created_by = NULL WHERE created_by = ?',
                'UPDATE ssh_keys SET revoked_by = NULL WHERE revoked_by = ?',
                'DELETE FROM ssh_keys WHERE user_id = ?',
                'DELETE FROM app_sessions WHERE user_id = ?',
                'UPDATE management_operations SET actor_user_id = NULL WHERE actor_user_id = ?',
                'UPDATE users SET created_by = NULL WHERE created_by = ?',
                'UPDATE system_settings SET updated_by = NULL WHERE updated_by = ?'):
        cursor.execute(sql, (user_id, ))
    cursor.execute('DELETE FROM management_operations WHERE target_user_id = ? AND id != ?',
                   (user_id, operation['id']))
    cursor.execute('UPDATE management_operations SET target_user_id = NULL WHERE id = ?',
                   (operation['id'], ))
    cursor.execute('UPDATE audit_events SET actor_login_name = ?, actor_user_id = NULL WHERE '
                   'actor_user_id = ?', (user['login_name'], user_id))
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id, ))
    audit.record(cursor, service.now(), 'purge_user', audit.SUCCEEDED, None, audit.SYSTEM, 'user',
                 user_id, operation_id=operation['id'],
                 before={key: user[key] for key in ('login_name', 'display_name', 'role')})
    return files


class Reconciler:

    def __init__(self, service, helper):
        """helper: has call(op, **args) and raises ClientError (a ServiceClient on the helper's
        socket, or the tests' in-process helper)."""
        self.service = service
        self.helper = helper
        self.db = service.db
        self.errors: Dict[str, str] = {}  # this pass's: 'policy', 'desktop:<slot>', ...
        self.configured = set()  # displays whose environment file this process wrote
        self.retry_at: Dict[str, float] = {}
        self.unknown: List[dict] = []  # connections with a key the database does not know
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.first_pass = True

    def call(self, op: str, **args):
        return self.helper.call(op, **args)

    # one pass
    def run_once(self) -> None:
        with self.lock:
            self.errors = {}
            for step in (self.publish_policy, self.sync_connections, self.cut_connections,
                         self.sync_desktops, self.release_desktops, self.purge_users):
                try:
                    step()
                except ClientError as exc:
                    self.errors.setdefault(step.__name__, f'{exc.code}: {exc}')
                    log.warning('%s: %s', step.__name__, exc)
            self.settle_operations()
            self.first_pass = False

    def loop(self, stop: threading.Event, every_s: float = 5.0) -> None:
        while not stop.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception('a reconciliation pass failed')
            self.wake.wait(every_s)
            self.wake.clear()

    # 1. the SSH policy
    def publish_policy(self) -> None:
        state = self.db.query_one('SELECT * FROM policy_state WHERE id = 1')
        now = self.service.now()
        ports, annotation_port = self.service.config.vnc_ports, self.service.config.ssh_port
        revision, entries = policy.snapshot(self.db, now)
        if revision == state['applied_revision'] and not self.first_pass:
            if policy.expected_checksum(revision, entries, ports, annotation_port) == \
                    state['applied_snapshot_checksum']:
                return
            # what was applied no longer matches (a key expired): a new revision
            with self.db.transaction() as cursor:
                policy.bump(cursor)
            revision, entries = policy.snapshot(self.db, now)
        try:
            result = self.call('publish_ssh_policy', revision=revision,
                               entries=[entry.as_dict() for entry in entries])
            if result['checksum'] != policy.expected_checksum(revision, entries, ports,
                                                              annotation_port):
                raise ClientError('failed', 'the published policy differs from the database')
        except ClientError as exc:
            self.errors['policy'] = str(exc)
            with self.db.transaction() as cursor:
                cursor.execute('UPDATE policy_state SET last_apply_error = ? WHERE id = 1',
                               (str(exc), ))
            return
        with self.db.transaction() as cursor:
            cursor.execute('UPDATE policy_state SET applied_revision = ?, '
                           'applied_snapshot_checksum = ?, last_applied_at = ?, '
                           'last_apply_error = NULL WHERE id = 1',
                           (revision, result['checksum'], now))
            audit.record(cursor, now, 'publish_ssh_policy', audit.SUCCEEDED, None, audit.SYSTEM,
                         'policy', str(revision), after={'keys': len(entries)})

    def policy_current(self) -> bool:
        state = self.db.query_one('SELECT * FROM policy_state WHERE id = 1')
        return state['applied_revision'] == state['desired_revision']

    # 2. connections
    def sync_connections(self) -> None:
        alive = self.call('ssh_connections')
        now = self.service.now()
        known = {row['server_process_identity']: row for row in self.db.query(
            "SELECT * FROM ssh_connections WHERE status IN ('active', 'disconnecting')")}
        seen = set()
        with self.db.transaction() as cursor:
            for connection in alive:
                identity = json.dumps(connection['identity'], sort_keys=True)
                seen.add(identity)
                if identity in known:
                    continue
                key = (cursor.execute('SELECT * FROM ssh_keys WHERE fingerprint_sha256 = ?',
                                      (connection['fingerprint'], )).fetchall() or [None])[0]
                if key is None:
                    # not a key of the policy: no record to attach it to; cut it
                    audit.record(cursor, now, 'ssh_unknown_key', audit.REFUSED, None, audit.SSHD,
                                 'connection', None, after=connection)
                    self.unknown.append(connection['identity'])
                    continue
                slot = (cursor.execute('SELECT id FROM desktop_slots WHERE user_id = ?',
                                       (key['user_id'], )).fetchall() or [{'id': None}])[0]
                connection_id = new_id()
                cursor.execute(
                    'INSERT INTO ssh_connections (id, user_id, ssh_key_id, desktop_slot_id, '
                    'source_address, source_port, ssh_entrypoint, connected_at, status, '
                    "server_process_identity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)",
                    (connection_id, key['user_id'], key['id'], slot['id'], connection['address'],
                     connection['port'], f'annotation-sshd:{self.service.config.ssh_port}', now,
                     identity))
                cursor.execute('UPDATE ssh_keys SET last_authenticated_at = ? WHERE id = ?',
                               (now, key['id']))
                audit.record(cursor, now, 'ssh_connected', audit.SUCCEEDED, key['user_id'],
                             audit.SSHD, 'connection', connection_id,
                             after={'address': connection['address'],
                                    'fingerprint': connection['fingerprint']})
            for identity, row in known.items():
                if identity not in seen:
                    cursor.execute("UPDATE ssh_connections SET status = 'closed', "
                                   'disconnected_at = ?, disconnect_reason = '
                                   "coalesce(disconnect_reason, 'ended') WHERE id = ?",
                                   (now, row['id']))
                    audit.record(cursor, now, 'ssh_disconnected', audit.SUCCEEDED, row['user_id'],
                                 audit.SYSTEM, 'connection', row['id'])

    # 3. cutting connections
    def cut_connections(self) -> None:
        now = self.service.now()
        for identity in self.unknown:
            self.call('disconnect', identity=identity)
        self.unknown = []
        rows = self.db.query(
            "SELECT c.*, k.status AS key_status, k.expires_at, u.status AS user_status "
            "FROM ssh_connections c JOIN ssh_keys k ON k.id = c.ssh_key_id "
            "JOIN users u ON u.id = c.user_id WHERE c.status IN ('active', 'disconnecting')")
        current = self.policy_current()
        for row in rows:
            reason = None
            if row['status'] == 'disconnecting':
                reason = row['disconnect_reason'] or 'disconnected by a manager'
            elif current and row['key_status'] == 'revoked':
                reason = 'key revoked'
            elif current and row['user_status'] not in ('pending', 'active'):
                reason = f'user {row["user_status"]}'
            elif current and row['expires_at'] is not None and row['expires_at'] <= now:
                reason = 'key expired'
            if reason is None:
                continue
            try:
                closed = self.call('disconnect', identity=json.loads(
                    row['server_process_identity']))['closed']
            except ClientError as exc:
                self.errors[f'connection:{row["id"]}'] = str(exc)
                continue
            if closed:
                with self.db.transaction() as cursor:
                    cursor.execute("UPDATE ssh_connections SET status = 'closed', "
                                   'disconnected_at = ?, disconnect_reason = ? WHERE id = ?',
                                   (now, reason, row['id']))
                    audit.record(cursor, now, 'ssh_disconnect', audit.SUCCEEDED, None,
                                 audit.SYSTEM, 'connection', row['id'], after={'reason': reason})

    # 4. desktops
    def sync_desktops(self) -> None:
        slots = self.db.query('SELECT * FROM desktop_slots ORDER BY display_number')
        if not slots:
            return
        states = self.call('desktop_status', displays=[slot['display_number'] for slot in slots])
        now = self.service.now()
        for slot in slots:
            display = slot['display_number']
            observed = states[str(display)]
            error = None
            try:
                password = self.service.pending_vnc_passwords.get(slot['id'])
                if password is not None:
                    written = self.call('write_vnc_password', display=display, password=password)
                    with self.db.transaction() as cursor:
                        cursor.execute('UPDATE desktop_slots SET vnc_password_file_reference = ?,'
                                       ' updated_at = ? WHERE id = ?',
                                       (written['file'], now, slot['id']))
                    self.service.pending_vnc_passwords.pop(slot['id'], None)
                    slot['vnc_password_file_reference'] = written['file']
                if display not in self.configured:
                    self.call('configure_desktop', display=display,
                              tasks_dir=slot['workspace_path'])
                    self.configured.add(display)
                state = observed['state']
                restart = self.restart_wanted(slot)
                if slot['desired_state'] == 'running' and state in ('stopped', 'error'):
                    if slot['vnc_password_file_reference'] is None:
                        error = 'no VNC password: reset it'
                    elif time.monotonic() >= self.retry_at.get(slot['id'], 0):
                        self.retry_at[slot['id']] = time.monotonic() + (
                            DESKTOP_RETRY_S if state == 'error' else 0)
                        observed = self.call('desktop_start', display=display)
                elif (slot['desired_state'] == 'stopped' or restart) and \
                        state in ('running', 'starting'):
                    observed = self.call('desktop_stop', display=display)
                if observed['state'] == 'error':
                    error = error or 'the desktop failed (see journalctl -u ' \
                        f'{slot["service_name"]})'
            except ClientError as exc:
                error = str(exc)
            if error:
                self.errors[f'desktop:{slot["id"]}'] = error
            # a process the slot has not seen before: a new generation of the desktop
            old_identity = ProcessIdentity.from_dict(json.loads(slot['process_identity'])) \
                if slot['process_identity'] else None
            new_process = observed['identity'] is not None and \
                ProcessIdentity.from_dict(observed['identity']) != old_identity
            with self.db.transaction() as cursor:
                cursor.execute(
                    'UPDATE desktop_slots SET observed_state = ?, process_identity = ?, '
                    'runtime_generation = runtime_generation + ?, last_heartbeat_at = ?, '
                    'last_error = ?, updated_at = ? WHERE id = ?',
                    (observed['state'], json.dumps(observed['identity'], sort_keys=True)
                     if observed['identity'] else None, 1 if new_process else 0, now, error,
                     now, slot['id']))

    def restart_wanted(self, slot: dict) -> bool:
        """An open restart_desktop operation whose desktop still runs the process it named."""
        for row in self.db.query(
                "SELECT * FROM management_operations WHERE operation_type = 'restart_desktop' "
                f"AND target_user_id = ? AND status IN ({','.join('?' * len(OPEN))})",
                (slot['user_id'], *OPEN)):
            if json.loads(row['validated_parameters_json'])['generation'] == \
                    slot['runtime_generation']:
                return True
        return False

    # 5. releasing desktops of removed users
    def release_desktops(self) -> None:
        """Only once the published policy is current: until then the file may still let a key
        of the removed user reach this display's port, which its next user would get."""
        if not self.policy_current():
            return
        rows = self.db.query(
            "SELECT s.* FROM desktop_slots s JOIN users u ON u.id = s.user_id "
            "WHERE u.status = 'archived' AND s.observed_state IN ('stopped', 'error') AND NOT "
            "EXISTS (SELECT 1 FROM ssh_connections c WHERE c.user_id = u.id AND c.status IN "
            "('active', 'disconnecting'))")
        for slot in rows:
            self.call('remove_desktop', display=slot['display_number'])
            self.configured.discard(slot['display_number'])
            with self.db.transaction() as cursor:
                cursor.execute('DELETE FROM desktop_slots WHERE id = ?', (slot['id'], ))
                audit.record(cursor, self.service.now(), 'release_desktop', audit.SUCCEEDED, None,
                             audit.SYSTEM, 'desktop', str(slot['display_number']),
                             after={'user_id': slot['user_id']})

    # 6. deleting users for good
    def purge_users(self) -> None:
        """Once the policy without their keys is published, their connections are closed, their
        desktop is released and their GPU jobs have ended."""
        if not self.policy_current():
            return
        rows = self.db.query(
            "SELECT m.* FROM management_operations m JOIN users u ON u.id = m.target_user_id "
            "WHERE m.operation_type = 'delete_user' AND m.status IN "
            f"({','.join('?' * len(OPEN))}) AND u.status = 'archived' AND NOT EXISTS "
            "(SELECT 1 FROM desktop_slots s WHERE s.user_id = u.id) AND NOT EXISTS "
            "(SELECT 1 FROM ssh_connections c WHERE c.user_id = u.id AND c.status IN "
            "('active', 'disconnecting')) AND NOT EXISTS (SELECT 1 FROM gpu_jobs j WHERE "
            "j.owner_user_id = u.id AND j.status IN ('queued', 'running', 'cancel_requested'))",
            OPEN)
        for row in rows:
            try:
                with self.db.transaction() as cursor:
                    files = purge_user(self.service, cursor, row)
            except Exception:  # the other operations still settle
                log.exception('deleting user %s failed', row['target_user_id'])
                self.errors['purge'] = 'the account could not be deleted: see the service log'
                continue
            for file_path in files:
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass

    # 7. operations
    def settle_operations(self) -> None:
        now = self.service.now()
        state = self.db.query_one('SELECT * FROM policy_state WHERE id = 1')
        rows = self.db.query('SELECT * FROM management_operations WHERE status IN '
                             f"({','.join('?' * len(OPEN))}) ORDER BY created_at", OPEN)
        with self.db.transaction() as cursor:
            for row in rows:
                status, step, error = self.progress(cursor, row, state)
                cursor.execute(
                    'UPDATE management_operations SET status = ?, current_step = ?, '
                    'error_summary = ?, updated_at = ?, completed_at = ? WHERE id = ?',
                    (status, step, error, now, now if status in (SUCCEEDED, SUPERSEDED) else None,
                     row['id']))

    def progress(self, cursor, row: dict, state: dict):
        """(status, the step it waits for, the error holding it up) of an open operation."""
        kind = row['operation_type']
        user_id = row['target_user_id']
        parameters = json.loads(row['validated_parameters_json'])
        one = lambda sql, *bindings: (cursor.execute(sql, bindings).fetchall() or [None])[0]
        user = one('SELECT * FROM users WHERE id = ?', user_id) if user_id else None
        slot = one('SELECT * FROM desktop_slots WHERE user_id = ?', user_id) if user_id else None
        waiting = []  # (step, error) still to do, in order

        def need_policy():
            if row['desired_revision'] is not None and \
                    state['applied_revision'] < row['desired_revision']:
                waiting.append(('publishing the SSH policy', self.errors.get('policy')))

        def need_no_connections(where: str, *bindings):
            if one(f"SELECT 1 AS x FROM ssh_connections WHERE {where} AND status IN "
                   "('active', 'disconnecting')", *bindings):
                waiting.append(('closing SSH connections', next(
                    (error for key, error in self.errors.items()
                     if key.startswith('connection:')), None)))

        def need_desktop(wanted: str):
            if slot is None:
                return
            if slot['vnc_password_file_reference'] is None and wanted == 'running':
                waiting.append(('writing the VNC password', self.errors.get(
                    f'desktop:{slot["id"]}')))
            elif slot['observed_state'] != wanted:
                waiting.append((f'{"starting" if wanted == "running" else "stopping"} the '
                                'desktop', self.errors.get(f'desktop:{slot["id"]}')))

        if kind in ('suspend_user', 'reactivate_user'):
            wanted_status = ('suspended', ) if kind == 'suspend_user' else ('pending', 'active')
            if user['status'] not in wanted_status:
                return SUPERSEDED, None, None
        if kind == 'provision_user':
            need_desktop('running')
        elif kind == 'suspend_user':
            need_policy()
            need_no_connections('user_id = ?', user_id)
            need_desktop('stopped')
        elif kind == 'reactivate_user':
            need_policy()
            need_desktop('running')
        elif kind == 'remove_user':
            need_policy()
            need_no_connections('user_id = ?', user_id)
            if slot is not None:
                need_desktop('stopped')
                waiting.append(('releasing the desktop', None))
        elif kind == 'delete_user':
            if user is None:  # deleted: purge_user
                return SUCCEEDED, None, None
            need_policy()
            need_no_connections('user_id = ?', user_id)
            if slot is not None:
                need_desktop('stopped')
                waiting.append(('releasing the desktop', None))
            if one("SELECT 1 AS x FROM gpu_jobs WHERE owner_user_id = ? AND status IN "
                   "('queued', 'running', 'cancel_requested')", user_id):
                waiting.append(('ending their GPU jobs', None))
            waiting.append(('deleting the account', self.errors.get('purge')))
        elif kind == 'add_key':
            need_policy()
        elif kind == 'revoke_key':
            need_policy()
            need_no_connections('ssh_key_id = ?', row['target_resource_id'])
        elif kind in ('start_desktop', 'stop_desktop'):
            wanted = 'running' if kind == 'start_desktop' else 'stopped'
            if slot is None or slot['desired_state'] != wanted:
                return SUPERSEDED, None, None
            need_desktop(wanted)
        elif kind == 'restart_desktop':
            if slot is None:
                return SUPERSEDED, None, None
            if slot['runtime_generation'] <= parameters['generation'] or \
                    slot['observed_state'] != 'running':
                waiting.append(('restarting the desktop', self.errors.get(
                    f'desktop:{slot["id"]}')))
        elif kind == 'reset_vnc_password':
            if slot is not None and slot['id'] in self.service.pending_vnc_passwords:
                waiting.append(('writing the VNC password', self.errors.get(
                    f'desktop:{slot["id"]}')))
        elif kind == 'disconnect':
            need_no_connections('id = ?', row['target_resource_id'])
        if not waiting:
            return SUCCEEDED, None, None
        step, error = waiting[0]
        return (FAILED if error else RUNNING), step, error
