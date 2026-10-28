"""
Managing application users (the Manage Annotators dialog). Every operation loads its target inside
its write transaction and asks the policy (authz) about that row, so the check and the change see
the same state: a Supervisor acting on someone who has just become a Supervisor is refused.

Status:  pending (no password yet: setup token) -> active <-> suspended;  any -> archived (Remove).
Every change of role, status or credentials raises the user's authz_version and ends their
sessions: they sign in again, with what they may do now. Removed users stay in the database, so
their jobs and audit records keep naming them, until an admin deletes them for good; their
annotation files are never touched.
"""
from datetime import timedelta
from typing import Optional

from cutieplus.cooperation import audit, authz, settings, validate
from cutieplus.cooperation.api import Context, op
from cutieplus.cooperation.clock import iso, new_id
from cutieplus.cooperation.errors import (CONFLICT, FORBIDDEN, INVALID, NOT_FOUND,
                                          UNEXPECTED_FIELD, Refused)
from cutieplus.cooperation.operations import start_operation
from cutieplus.cooperation.service import Service, confirm_password, public_user

MAX_DISPLAY_NAME = 80
MAX_CONTACT = 200


def load_user(cursor, user_id) -> dict:
    validate.identifier(user_id, 'user_id')
    rows = cursor.execute('SELECT * FROM users WHERE id = ?', (user_id, )).fetchall()
    if not rows:
        raise Refused(NOT_FOUND, 'no such user')
    return rows[0]


def user_target(service: Service, ctx: Context, cursor, user_id, action: str) -> dict:
    target = load_user(cursor, user_id)
    authz.check_user_action(ctx.actor, target, action, service.owner_id())
    return target


def change_user(service: Service, ctx: Context, cursor, before: dict, action: str,
                result: Optional[dict] = None, end_sessions: bool = True, **columns) -> dict:
    """Applies columns to the user (raising authz_version when end_sessions), runs the hooks of
    the rest of the service, and audits the change. Returns the new row."""
    now = service.now()
    assignments = ', '.join(f'{name} = ?' for name in columns)
    extra = ', authz_version = authz_version + 1' if end_sessions else ''
    cursor.execute(f'UPDATE users SET {assignments}{extra}, row_version = row_version + 1, '
                   'updated_at = ? WHERE id = ?', (*columns.values(), now, before['id']))
    if end_sessions:
        service.revoke_sessions(cursor, before['id'])
    after = load_user(cursor, before['id'])
    service.run_user_hooks(cursor, before, after, ctx.actor.user_id, now, result)
    audit.record(cursor, now, action, audit.SUCCEEDED, ctx.actor.user_id, target_type='user',
                 target_id=before['id'], request_id=ctx.request_id,
                 before=public_user(before), after=public_user(after))
    return after


@op()
def list_users(service: Service, ctx: Context, query='', include_archived=False) -> list:
    """An admin sees everyone; a supervisor the annotators; an annotator nobody (refused)."""
    authz.require_manager(ctx.actor)
    query = validate.text(query, 'query', 80, required=False) or ''
    where, bindings = ['1'], []
    if ctx.actor.role == authz.SUPERVISOR:
        where.append("u.role = 'annotator'")
    if not include_archived:
        where.append("u.status != 'archived'")
    if query:
        where.append("(u.login_name LIKE ? ESCAPE '\\' OR u.display_name LIKE ? ESCAPE '\\' OR "
                     "coalesce(u.contact, '') LIKE ? ESCAPE '\\')")
        pattern = '%' + query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
        bindings += [pattern] * 3
    now = service.clock.now()
    idle_cutoff = iso(now - timedelta(minutes=service.setting('session_idle_minutes')))
    rows = service.db.query(
        'SELECT u.*, s.display_number AS desktop_display, s.observed_state AS desktop_state, '
        's.desired_state AS desktop_desired, s.last_error AS desktop_error, '
        "(SELECT count(*) FROM ssh_connections c WHERE c.user_id = u.id AND c.status IN "
        "('active', 'disconnecting')) AS ssh_connections, "
        '(SELECT count(*) FROM app_sessions a WHERE a.user_id = u.id AND a.revoked_at IS NULL '
        'AND a.expires_at > ? AND a.last_seen_at > ?) AS app_sessions, '
        "(SELECT count(*) FROM management_operations m WHERE m.target_user_id = u.id AND "
        "m.status IN ('pending', 'running')) AS operations_running, "
        "(SELECT count(*) FROM management_operations m WHERE m.target_user_id = u.id AND "
        "m.status = 'failed') AS operations_failed "
        f'FROM users u LEFT JOIN desktop_slots s ON s.user_id = u.id '
        f'WHERE {" AND ".join(where)} ORDER BY u.role, u.login_name',
        [iso(now), idle_cutoff] + bindings)
    owner = service.owner_id()
    summary = ('desktop_display', 'desktop_state', 'desktop_desired', 'desktop_error',
               'ssh_connections', 'app_sessions', 'operations_running', 'operations_failed')
    return [dict(public_user(row), is_owner=row['id'] == owner,
                 **{key: row[key] for key in summary}) for row in rows]


@op()
def get_user(service: Service, ctx: Context, user_id) -> dict:
    with service.db.lock:
        target = user_target(service, ctx, service.db.conn.cursor(), user_id, authz.VIEW)
    sessions = service.db.query(
        'SELECT id, desktop_slot_id, issued_at, expires_at, last_seen_at FROM app_sessions '
        'WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ? ORDER BY issued_at',
        (target['id'], service.now()))
    details = dict(public_user(target), is_owner=target['id'] == service.owner_id(),
                   sessions=sessions)
    if authz.is_manager(ctx.actor):
        details['recent_activity'] = audit.events(service.db, 'e.target_id = ?',
                                                  (target['id'], ), limit=20)
    for extend in service.user_details:
        extend(target, details)
    return details


@op()
def create_user(service: Service, ctx: Context, login_name, display_name, contact=None,
                role=None) -> dict:
    """A supervisor creates annotators only, and may not name a role at all; an admin creates an
    annotator or a supervisor. The reply holds the one-time setup token: it is shown once."""
    authz.require_manager(ctx.actor)
    if ctx.actor.role != authz.ADMIN:
        if role is not None:
            raise Refused(UNEXPECTED_FIELD, 'a supervisor creates annotators: no role field',
                          fields=['role'])
        role = authz.ANNOTATOR
    elif role is None:
        role = authz.ANNOTATOR
    elif role not in authz.ASSIGNABLE_ROLES:
        raise Refused(INVALID, f'role is one of {", ".join(authz.ASSIGNABLE_ROLES)}')
    name = validate.new_login_name(login_name)
    display_name = validate.text(display_name, 'display_name', MAX_DISPLAY_NAME)
    contact = validate.text(contact, 'contact', MAX_CONTACT, required=False)
    now = service.now()
    result = {}
    with service.db.transaction() as cursor:
        if cursor.execute('SELECT 1 FROM users WHERE login_name = ?', (name, )).fetchall():
            raise Refused(CONFLICT, f'{name} is taken (also by removed users)')
        user_id = new_id()
        cursor.execute(
            'INSERT INTO users (id, login_name, display_name, contact, role, status, '
            'credential_setup_required, created_by, created_at, updated_at) '
            "VALUES (?, ?, ?, ?, ?, 'pending', 1, ?, ?, ?)",
            (user_id, name, display_name, contact, role, ctx.actor.user_id, now, now))
        token = service.setup_token(cursor, user_id)
        user = load_user(cursor, user_id)
        service.run_user_hooks(cursor, None, user, ctx.actor.user_id, now, result)
        audit.record(cursor, now, 'create_user', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='user', target_id=user_id, request_id=ctx.request_id,
                     after=public_user(user))
    return dict(result, user=public_user(service.user_row(user_id)), setup_token=token,
                setup_token_expires_at=user['setup_token_expires_at'])


@op()
def update_user(service: Service, ctx: Context, user_id, row_version, display_name=None,
                contact=None) -> dict:
    """Profile fields only; row_version: the one the caller saw (a change meanwhile: conflict)."""
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.EDIT_PROFILE)
        if before['status'] == 'archived':
            raise Refused(INVALID, 'a removed user is not edited')
        if row_version != before['row_version']:
            raise Refused(CONFLICT, 'the user changed meanwhile; refresh')
        columns = {}
        if display_name is not None:
            columns['display_name'] = validate.text(display_name, 'display_name',
                                                    MAX_DISPLAY_NAME)
        if contact is not None:
            columns['contact'] = validate.text(contact, 'contact', MAX_CONTACT, required=False)
        if not columns:
            return public_user(before)
        after = change_user(service, ctx, cursor, before, 'update_user', end_sessions=False,
                            **columns)
    return public_user(after)


@op()
def suspend_user(service: Service, ctx: Context, user_id) -> dict:
    """Blocks the user's sign-in and ends their sessions; the rest of the service follows (their
    SSH access, desktop and queued jobs)."""
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.SUSPEND)
        if before['status'] not in ('pending', 'active'):
            raise Refused(INVALID, f'the user is {before["status"]}')
        after = change_user(service, ctx, cursor, before, 'suspend_user', status='suspended',
                            suspended_at=service.now())
    return public_user(after)


@op()
def reactivate_user(service: Service, ctx: Context, user_id) -> dict:
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.REACTIVATE)
        if before['status'] != 'suspended':
            raise Refused(INVALID, f'the user is {before["status"]}, not suspended')
        status = 'pending' if before['credential_setup_required'] else 'active'
        after = change_user(service, ctx, cursor, before, 'reactivate_user', status=status,
                            suspended_at=None)
    return public_user(after)


@op()
def remove_user(service: Service, ctx: Context, user_id) -> dict:
    """Revokes every access and archives the user. Their annotation files, jobs and audit
    records are kept."""
    service.require_fresh(ctx)
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.REMOVE)
        if before['status'] == 'archived':
            raise Refused(INVALID, 'the user is already removed')
        after = change_user(service, ctx, cursor, before, 'remove_user', status='archived',
                            archived_at=service.now(), setup_token_hash=None,
                            setup_token_expires_at=None)
    return public_user(after)


@op()
def delete_user(service: Service, ctx: Context, user_id, password) -> dict:
    """Deletes a user for good: an admin, never on themselves or the owner, with their password
    every time. Their access ends at once, as with Remove; once their desktop is released, the
    reconciler deletes their account, keys, sessions and GPU job records (operations.purge_user)
    and their login name is free again. The audit log and their annotation files are kept."""
    with service.db.lock:  # refused before the password is looked at
        user_target(service, ctx, service.db.conn.cursor(), user_id, authz.DELETE)
    confirm_password(service, ctx, password, 'delete_user')
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.DELETE)
        if cursor.execute("SELECT 1 FROM management_operations WHERE operation_type = "
                          "'delete_user' AND target_user_id = ? AND status IN "
                          "('pending', 'running', 'failed')", (before['id'], )).fetchall():
            raise Refused(INVALID, 'the user is being deleted already')
        after = before
        if before['status'] != 'archived':
            after = change_user(service, ctx, cursor, before, 'remove_user', status='archived',
                                archived_at=service.now(), setup_token_hash=None,
                                setup_token_expires_at=None)
        revision = cursor.execute('SELECT desired_revision FROM policy_state WHERE id = 1'
                                  ).fetchall()[0]['desired_revision']
        operation_id = start_operation(service, cursor, 'delete_user', ctx.actor.user_id,
                                       before['id'], request_id=ctx.request_id,
                                       desired_revision=revision,
                                       parameters={'login_name': before['login_name']})
        audit.record(cursor, service.now(), 'delete_user', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='user', target_id=before['id'], request_id=ctx.request_id,
                     operation_id=operation_id, before=public_user(before))
    return {'user': public_user(after), 'operation_id': operation_id}


@op()
def reset_credential(service: Service, ctx: Context, user_id) -> dict:
    """Clears the password and issues a new setup token (shown once); the user's sessions end."""
    result = {}
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.RESET_CREDENTIAL)
        if before['status'] not in ('pending', 'active'):
            raise Refused(INVALID, f'the user is {before["status"]}')
        token = service.setup_token(cursor, before['id'])
        after = change_user(service, ctx, cursor, before, 'reset_credential', result,
                            password_hash=None, credential_setup_required=1)
    return dict(result, user=public_user(after), setup_token=token,
                setup_token_expires_at=after['setup_token_expires_at'])


@op()
def set_role(service: Service, ctx: Context, user_id, role) -> dict:
    """Admin only: makes an annotator a supervisor, or a supervisor an annotator. The user signs
    in again; an annotator's desktop keeps running."""
    authz.require_admin(ctx.actor)
    if role not in authz.ASSIGNABLE_ROLES:
        raise Refused(INVALID, f'role is one of {", ".join(authz.ASSIGNABLE_ROLES)}')
    service.require_fresh(ctx)
    with service.db.transaction() as cursor:
        before = user_target(service, ctx, cursor, user_id, authz.SET_ROLE)
        if before['status'] == 'archived':
            raise Refused(INVALID, 'a removed user has no role to change')
        if before['role'] == role:
            return public_user(before)
        after = change_user(service, ctx, cursor, before, 'set_role', role=role)
    return public_user(after)


@op(name='revoke_sessions')
def end_sessions(service: Service, ctx: Context, user_id) -> dict:
    """Signs the user out of every window (your own, or a user you manage)."""
    with service.db.transaction() as cursor:
        target = user_target(service, ctx, cursor, user_id, authz.MANAGE_SESSIONS)
        ended = service.revoke_sessions(cursor, target['id'],
                                        keep=ctx.session['id'] if target['id'] ==
                                        ctx.actor.user_id else None)
        audit.record(cursor, service.now(), 'revoke_sessions', audit.SUCCEEDED,
                     ctx.actor.user_id, target_type='user', target_id=target['id'],
                     request_id=ctx.request_id, after={'ended': ended})
    return {'ended': ended}


@op()
def list_audit(service: Service, ctx: Context, user_id=None, limit=100) -> list:
    """An admin: every event (or one user's); a supervisor: the events about annotators and their
    own actions."""
    authz.require_manager(ctx.actor)
    limit = validate.integer(limit, 'limit', 1, 1000)
    where, bindings = [], []
    if user_id is not None:
        with service.db.lock:
            target = user_target(service, ctx, service.db.conn.cursor(), user_id, authz.VIEW)
        where.append('(e.target_id = ? OR e.actor_user_id = ?)')
        bindings += [target['id'], target['id']]
    if ctx.actor.role != authz.ADMIN:
        where.append("((e.target_type = 'user' AND e.target_id IN "
                     "(SELECT id FROM users WHERE role = 'annotator')) OR e.actor_user_id = ?)")
        bindings.append(ctx.actor.user_id)
    return audit.events(service.db, ' AND '.join(where) or '1', bindings, limit)


@op()
def get_settings(service: Service, ctx: Context) -> dict:
    authz.require_admin(ctx.actor)
    owner = service.owner_id()
    owner_row = service.db.query_one('SELECT login_name FROM users WHERE id = ?', (owner, ))
    return {'editable': settings.all_settings(service.db),
            'fixed': {'owner': owner_row['login_name'] if owner_row else None,
                      'workspace_root': service.config.workspace_root,
                      'displays': [service.config.first_display, service.config.last_display],
                      'ssh_port': service.config.ssh_port, 'gpu_jobs_at_once': 1}}


@op()
def update_setting(service: Service, ctx: Context, key, value) -> dict:
    authz.require_admin(ctx.actor)
    service.require_fresh(ctx)
    if key == settings.OWNER_USER_ID:
        raise Refused(FORBIDDEN, 'the owner is changed only by the maintenance commands')
    value = settings.validated(key, value)
    with service.db.transaction() as cursor:
        before = settings.read_setting(service.db, key)
        settings.write_setting(cursor, key, value, ctx.actor.user_id, service.now())
        audit.record(cursor, service.now(), 'update_setting', audit.SUCCEEDED,
                     ctx.actor.user_id, target_type='setting', target_id=key,
                     request_id=ctx.request_id, before={'value': before},
                     after={'value': value})
    return {'key': key, 'value': value}
