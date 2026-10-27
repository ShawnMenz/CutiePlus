"""
Remote desktops (desktop_slots): one per annotator or supervisor, given at creation (or to anyone
with assign_desktop), at a display and VNC port that stay theirs. A desktop runs while its user may
sign in and stops when they are suspended or removed; disconnecting SSH leaves it running, so a
user reconnects to the same desktop and work. The requests here change the desired state; the
reconciler starts and stops the units through the root helper.

A VNC password is generated here, shown once in the reply, and handed to the helper by the
reconciler from memory: the database keeps only the reference to its file.
"""
import json

from cutieplus.cooperation import audit, authz
from cutieplus.cooperation.api import Context, op
from cutieplus.cooperation.errors import CONFLICT, INVALID, NOT_FOUND, Refused
from cutieplus.cooperation.operations import (allocate_desktop, desktop_card, new_vnc_password,
                                              start_operation)
from cutieplus.cooperation.service import Service
from cutieplus.cooperation.users import user_target


def slot_of(cursor, user_id: str):
    rows = cursor.execute('SELECT * FROM desktop_slots WHERE user_id = ?', (user_id, )).fetchall()
    return rows[0] if rows else None


def public_slot(row: dict) -> dict:
    return {key: row[key] for key in ('id', 'display_number', 'vnc_port', 'workspace_path',
                                      'service_name', 'desired_state', 'observed_state',
                                      'runtime_generation', 'last_heartbeat_at', 'last_error',
                                      'updated_at')} | {
        'vnc_password_set': row['vnc_password_file_reference'] is not None}


def managed_slot(service: Service, ctx: Context, cursor, user_id) -> tuple:
    target = user_target(service, ctx, cursor, user_id, authz.MANAGE_DESKTOP)
    slot = slot_of(cursor, target['id'])
    if slot is None:
        raise Refused(NOT_FOUND, f'{target["login_name"]} has no desktop')
    return target, slot


@op()
def assign_desktop(service: Service, ctx: Context, user_id) -> dict:
    """A desktop for a user who has none (an admin for themselves, say)."""
    with service.db.transaction() as cursor:
        target = user_target(service, ctx, cursor, user_id, authz.MANAGE_DESKTOP)
        if target['status'] not in ('pending', 'active'):
            raise Refused(INVALID, f'the user is {target["status"]}')
        if slot_of(cursor, target['id']) is not None:
            raise Refused(CONFLICT, f'{target["login_name"]} has a desktop already')
        card = allocate_desktop(service, cursor, target)
        operation_id = start_operation(service, cursor, 'provision_user', ctx.actor.user_id,
                                       target['id'], request_id=ctx.request_id)
        audit.record(cursor, service.now(), 'assign_desktop', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='desktop', target_id=str(card['display']),
                     request_id=ctx.request_id, operation_id=operation_id,
                     after={'user_id': target['id']})
    return {'desktop': card, 'operation_id': operation_id}


def _set_desired(service: Service, ctx: Context, user_id, wanted: str, kind: str) -> dict:
    with service.db.transaction() as cursor:
        target, slot = managed_slot(service, ctx, cursor, user_id)
        if wanted == 'running' and target['status'] not in ('pending', 'active'):
            raise Refused(INVALID, f'the user is {target["status"]}: reactivate them first')
        cursor.execute('UPDATE desktop_slots SET desired_state = ?, updated_at = ? WHERE id = ?',
                       (wanted, service.now(), slot['id']))
        operation_id = start_operation(service, cursor, kind, ctx.actor.user_id, target['id'],
                                       slot['id'], ctx.request_id)
        audit.record(cursor, service.now(), kind, audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='desktop', target_id=str(slot['display_number']),
                     request_id=ctx.request_id, operation_id=operation_id,
                     before={'desired_state': slot['desired_state']},
                     after={'desired_state': wanted})
    return {'operation_id': operation_id}


@op()
def start_desktop(service: Service, ctx: Context, user_id) -> dict:
    return _set_desired(service, ctx, user_id, 'running', 'start_desktop')


@op()
def stop_desktop(service: Service, ctx: Context, user_id) -> dict:
    """Stops the desktop: its window saves its pending work and closes."""
    return _set_desired(service, ctx, user_id, 'stopped', 'stop_desktop')


@op()
def restart_desktop(service: Service, ctx: Context, user_id) -> dict:
    with service.db.transaction() as cursor:
        target, slot = managed_slot(service, ctx, cursor, user_id)
        if target['status'] not in ('pending', 'active'):
            raise Refused(INVALID, f'the user is {target["status"]}')
        cursor.execute("UPDATE desktop_slots SET desired_state = 'running', updated_at = ? "
                       'WHERE id = ?', (service.now(), slot['id']))
        operation_id = start_operation(service, cursor, 'restart_desktop', ctx.actor.user_id,
                                       target['id'], slot['id'], ctx.request_id,
                                       parameters={'generation': slot['runtime_generation']})
        audit.record(cursor, service.now(), 'restart_desktop', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='desktop', target_id=str(slot['display_number']),
                     request_id=ctx.request_id, operation_id=operation_id)
    return {'operation_id': operation_id}


@op()
def reset_vnc_password(service: Service, ctx: Context, user_id) -> dict:
    """A new VNC password (shown once). The desktop restarts to use it: its window saves its
    pending work and closes."""
    password = new_vnc_password()
    with service.db.transaction() as cursor:
        target, slot = managed_slot(service, ctx, cursor, user_id)
        operation_id = start_operation(service, cursor, 'reset_vnc_password', ctx.actor.user_id,
                                       target['id'], slot['id'], ctx.request_id)
        if slot['observed_state'] in ('running', 'starting'):
            start_operation(service, cursor, 'restart_desktop', ctx.actor.user_id, target['id'],
                            slot['id'], parameters={'generation': slot['runtime_generation']})
        audit.record(cursor, service.now(), 'reset_vnc_password', audit.SUCCEEDED,
                     ctx.actor.user_id, target_type='desktop',
                     target_id=str(slot['display_number']), request_id=ctx.request_id,
                     operation_id=operation_id)
    service.pending_vnc_passwords[slot['id']] = password  # once committed
    service.reconcile_soon()
    return {'desktop': desktop_card(service, slot['display_number'], password),
            'operation_id': operation_id}


def desktop_details(service: Service, user: dict, details: dict) -> None:
    row = service.db.query_one('SELECT * FROM desktop_slots WHERE user_id = ?', (user['id'], ))
    details['desktop'] = None if row is None else dict(
        public_slot(row), card=desktop_card(service, row['display_number']),
        process=json.loads(row['process_identity']) if row['process_identity'] else None)
