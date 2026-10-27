"""
Users' SSH public keys and their connections (Manage Annotators: keys, Disconnect). A key's
fingerprint is computed here from the key itself, and belongs to one user only, for good: a revoked
key never comes back (send a new one). Private keys are refused on sight; the comment of a key is
dropped. Adding or revoking a key changes the SSH policy, which the reconciler publishes; revoking
also cuts the key's tracked connections, once the policy no longer has it.
"""
from datetime import datetime, timedelta, timezone

from cutieplus.cooperation import audit, authz, policy, validate
from cutieplus.cooperation.api import Context, op
from cutieplus.cooperation.clock import iso, new_id
from cutieplus.cooperation.errors import CONFLICT, INVALID, NOT_FOUND, Refused
from cutieplus.cooperation.operations import public_operation, repeated_operation, start_operation
from cutieplus.cooperation.service import Service
from cutieplus.cooperation.sshpolicy import KeyError_, parse_public_key
from cutieplus.cooperation.users import load_user, user_target

MAX_LABEL = 60
MAX_KEY_YEARS = 5


def public_key_row(row: dict) -> dict:
    return {key: row[key] for key in ('id', 'user_id', 'device_label', 'fingerprint_sha256',
                                      'status', 'created_at', 'expires_at', 'revoked_at',
                                      'last_authenticated_at')} | {
        'key_type': row['public_key'].split()[0]}


def expiry(service: Service, value):
    """None, a date (the end of that day, UTC) or a UTC time; in the future, within 5 years."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise Refused(INVALID, 'expires_at is a date like 2027-06-30')
    try:
        if len(value) == 10:
            moment = datetime.strptime(value, '%Y-%m-%d').replace(
                tzinfo=timezone.utc) + timedelta(days=1) - timedelta(microseconds=1)
        else:
            moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if moment.tzinfo is None:
                raise ValueError(value)
    except ValueError:
        raise Refused(INVALID, 'expires_at is a date like 2027-06-30') from None
    now = service.clock.now()
    if not now < moment <= now + timedelta(days=366 * MAX_KEY_YEARS):
        raise Refused(INVALID, f'a key expires in the future, within {MAX_KEY_YEARS} years')
    return iso(moment)


@op()
def add_key(service: Service, ctx: Context, user_id, public_key, device_label,
            expires_at=None) -> dict:
    """Registers a user's public key (the .pub file's line). Supervisors for annotators, admins
    for annotators, supervisors and themselves."""
    repeated = repeated_operation(service, ctx.request_id)
    if repeated is not None:
        return {'operation': public_operation(repeated), 'repeated': True}
    try:
        key = parse_public_key(public_key)
    except KeyError_ as exc:
        raise Refused(INVALID, str(exc)) from None
    label = validate.text(device_label, 'device_label', MAX_LABEL)
    expires = expiry(service, expires_at)
    now = service.now()
    with service.db.transaction() as cursor:
        target = user_target(service, ctx, cursor, user_id, authz.MANAGE_KEYS)
        if target['status'] == 'archived':
            raise Refused(INVALID, 'a removed user gets no keys')
        holder = cursor.execute('SELECT user_id, status FROM ssh_keys WHERE '
                                'fingerprint_sha256 = ?', (key.fingerprint, )).fetchall()
        if holder:
            if holder[0]['user_id'] != target['id']:
                raise Refused(CONFLICT, 'this key belongs to another user')
            raise Refused(CONFLICT, 'this key is registered already' if holder[0]['status'] !=
                          'revoked' else 'this key was revoked: it cannot come back; send a new '
                          'one')
        key_id = new_id()
        cursor.execute(
            'INSERT INTO ssh_keys (id, user_id, public_key, fingerprint_sha256, device_label, '
            "status, created_by, created_at, expires_at) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
            (key_id, target['id'], key.canonical, key.fingerprint, label, ctx.actor.user_id, now,
             expires))
        operation_id = start_operation(service, cursor, 'add_key', ctx.actor.user_id,
                                       target['id'], key_id, ctx.request_id, policy.bump(cursor))
        row = cursor.execute('SELECT * FROM ssh_keys WHERE id = ?', (key_id, )).fetchall()[0]
        audit.record(cursor, now, 'add_key', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='ssh_key', target_id=key_id, request_id=ctx.request_id,
                     operation_id=operation_id, after=public_key_row(row))
    return {'key': public_key_row(row), 'operation_id': operation_id}


def load_key(cursor, key_id) -> dict:
    validate.identifier(key_id, 'key_id')
    rows = cursor.execute('SELECT * FROM ssh_keys WHERE id = ?', (key_id, )).fetchall()
    if not rows:
        raise Refused(NOT_FOUND, 'no such key')
    return rows[0]


@op()
def revoke_key(service: Service, ctx: Context, key_id) -> dict:
    """Blocks the key at the next policy publication and cuts its tracked connections; the user's
    other keys keep working. Checked against the role of the key's owner as it is now."""
    repeated = repeated_operation(service, ctx.request_id)
    if repeated is not None:
        return {'operation': public_operation(repeated), 'repeated': True}
    now = service.now()
    with service.db.transaction() as cursor:
        key = load_key(cursor, key_id)
        owner = load_user(cursor, key['user_id'])
        authz.check_user_action(ctx.actor, owner, authz.MANAGE_KEYS, service.owner_id())
        if key['status'] == 'revoked':
            raise Refused(INVALID, 'the key is revoked already')
        cursor.execute("UPDATE ssh_keys SET status = 'revoked', revoked_by = ?, revoked_at = ? "
                       'WHERE id = ?', (ctx.actor.user_id, now, key['id']))
        operation_id = start_operation(service, cursor, 'revoke_key', ctx.actor.user_id,
                                       owner['id'], key['id'], ctx.request_id,
                                       policy.bump(cursor))
        audit.record(cursor, now, 'revoke_key', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='ssh_key', target_id=key['id'], request_id=ctx.request_id,
                     operation_id=operation_id, before=public_key_row(key),
                     after={'status': 'revoked'})
    return {'key_id': key['id'], 'operation_id': operation_id}


@op()
def disconnect(service: Service, ctx: Context, connection_id) -> dict:
    """Ends one SSH connection; the user's desktop and work stay, and their key still works."""
    validate.identifier(connection_id, 'connection_id')
    now = service.now()
    with service.db.transaction() as cursor:
        rows = cursor.execute('SELECT * FROM ssh_connections WHERE id = ?',
                              (connection_id, )).fetchall()
        if not rows:
            raise Refused(NOT_FOUND, 'no such connection')
        connection = rows[0]
        owner = load_user(cursor, connection['user_id'])
        authz.check_user_action(ctx.actor, owner, authz.MANAGE_SESSIONS, service.owner_id())
        if connection['status'] == 'closed':
            raise Refused(INVALID, 'the connection is closed already')
        cursor.execute("UPDATE ssh_connections SET status = 'disconnecting', "
                       'disconnect_reason = ? WHERE id = ?',
                       (f'disconnected by {ctx.actor.login_name}', connection['id']))
        operation_id = start_operation(service, cursor, 'disconnect', ctx.actor.user_id,
                                       owner['id'], connection['id'], ctx.request_id)
        audit.record(cursor, now, 'disconnect', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='connection', target_id=connection['id'],
                     request_id=ctx.request_id, operation_id=operation_id)
    return {'connection_id': connection['id'], 'operation_id': operation_id}


def key_details(service: Service, user: dict, details: dict) -> None:
    db = service.db
    details['keys'] = [public_key_row(row) for row in db.query(
        'SELECT * FROM ssh_keys WHERE user_id = ? ORDER BY created_at', (user['id'], ))]
    details['connections'] = db.query(
        'SELECT id, ssh_key_id, source_address, source_port, connected_at, disconnected_at, '
        'status, disconnect_reason FROM ssh_connections WHERE user_id = ? '
        "ORDER BY status = 'closed', connected_at DESC LIMIT 20", (user['id'], ))
    details['operations'] = [public_operation(row) for row in db.query(
        'SELECT * FROM management_operations WHERE target_user_id = ? ORDER BY created_at DESC '
        'LIMIT 20', (user['id'], ))]
