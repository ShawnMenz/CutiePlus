"""
The cooperation service: every request of a window passes through Service.handle, which

1. works out where it comes from (Origin) from the kernel's word on the peer process: the managed
   desktop whose systemd service holds it, or the console;
2. for every operation but hello, login and setup_credential, authenticates the session token:
   the session must be unexpired, unrevoked, from the same origin it was issued to, and its user
   active, set up and of the same authorization version as when it was issued, so a session issued
   before a role change, a suspension or a credential reset is refused at once;
3. runs the operation (cutieplus.cooperation.api), which loads its target as the database has it
   now, asks the policy (authz), applies the change in one short transaction and audits it.

GUI buttons only reflect the policy; a request sent directly is checked the same way.
"""
import logging
import re
import threading
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

from cutieplus.cooperation import api, audit, auth, settings, validate
from cutieplus.cooperation.api import Context, Origin, op
from cutieplus.cooperation.authz import Actor
from cutieplus.cooperation.clock import Clock, iso, new_id, parse
from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.db import Database
from cutieplus.cooperation.errors import (BAD_REQUEST, FORBIDDEN, INVALID, INVALID_CREDENTIALS,
                                          NOT_FOUND, OWNER_PROTECTED, REAUTH_REQUIRED,
                                          SESSION_EXPIRED, SESSION_INVALID, SESSION_REVOKED,
                                          SESSION_STALE, SETUP_REQUIRED, UNEXPECTED_FIELD,
                                          WRONG_DESKTOP, Refused)
from cutieplus.cooperation.procinfo import Peer, system_unit_of
from cutieplus.cooperation.protocol import PROTOCOL_VERSION, error, ok

log = logging.getLogger(__name__)

SERVICE_VERSION = '1'
LAST_SEEN_EVERY_S = 60  # a session's last_seen_at is written at most this often
# a user's change the other parts of the service follow (desktops, keys, jobs), in the change's
# transaction: hook(cursor, before, after, actor_id, now, result); before is None for a new user;
# result is the reply's dict, for what the caller must see once (a desktop's VNC password)
UserHook = Callable[..., None]


def public_user(row: dict) -> dict:
    return {key: row[key] for key in ('id', 'login_name', 'display_name', 'contact', 'role',
                                      'status', 'credential_setup_required', 'row_version',
                                      'created_at', 'updated_at', 'last_login_at',
                                      'suspended_at', 'archived_at')}


class Service:

    def __init__(self, config: ServiceConfig, db: Database, clock: Optional[Clock] = None,
                 passwords: Optional[auth.Passwords] = None,
                 unit_resolver: Callable[[int], Optional[str]] = system_unit_of):
        self.config = config
        self.db = db
        self.clock = clock or db.clock
        self.passwords = passwords or auth.Passwords()
        self.unit_resolver = unit_resolver
        self.limiter = auth.LoginLimiter()
        self.user_hooks: List[UserHook] = []
        # what other parts add to get_user's reply: extend(user_row, details)
        self.user_details: List[Callable[[dict, dict], None]] = []
        # a user suspended or removed: hook(cursor, user, actor_id, now) (their GPU jobs)
        self.suspension_hooks: List[Callable[..., None]] = []
        # VNC passwords to write, by desktop slot: in memory only, never in the database
        self.pending_vnc_passwords = {}
        self.reconciler = None
        self.lock = threading.Lock()
        pattern = re.escape(config.desktop_unit).replace(re.escape('{display}'), r'(\d+)')
        self.desktop_unit_pattern = re.compile(pattern)

    def after_commit(self, callback: Callable[[], None]) -> None:
        self.db.after_commit(callback)

    def reconcile_soon(self) -> None:
        """A change for the reconciler: its next pass starts now rather than in a few seconds."""
        if self.reconciler is not None:
            self.reconciler.wake.set()

    # time
    def now(self) -> str:
        return iso(self.clock.now())

    def setting(self, key: str):
        return settings.read_setting(self.db, key)

    def owner_id(self) -> Optional[str]:
        return settings.read_setting(self.db, settings.OWNER_USER_ID)

    # requests
    def handle(self, request: dict, peer: Peer) -> dict:
        operation = api.OPS.get(request.get('op'))
        if operation is None:
            return error(BAD_REQUEST, f'no operation {request.get("op")!r}')
        args = request.get('args') or {}
        if not isinstance(args, dict):
            return error(BAD_REQUEST, 'args must be an object')
        unexpected = sorted(set(args) - operation.parameters)
        if unexpected:
            return error(UNEXPECTED_FIELD, f'{operation.name} takes no {", ".join(unexpected)}',
                         fields=unexpected)
        missing = sorted(operation.required - set(args))
        if missing:
            return error(BAD_REQUEST, f'{operation.name} needs {", ".join(missing)}')
        request_id = request.get('request_id')
        if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 64):
            return error(BAD_REQUEST, 'request_id is text of up to 64 characters')
        ctx = Context(peer, self.origin_of(peer), request_id)
        try:
            if not operation.public:
                ctx.actor, ctx.session = self.authenticate(request.get('token'), ctx.origin)
            return ok(operation.function(self, ctx, **args))
        except Refused as exc:
            if ctx.actor is not None and exc.code in (FORBIDDEN, OWNER_PROTECTED):
                with self.db.transaction() as cursor:
                    audit.record(cursor, self.now(), operation.name, audit.REFUSED,
                                 ctx.actor.user_id, request_id=request_id, error_code=exc.code,
                                 after={'args': sorted(args)})
            return error(exc.code, exc.message, **exc.details)

    def origin_of(self, peer: Peer) -> Origin:
        unit = self.unit_resolver(peer.pid)
        match = self.desktop_unit_pattern.fullmatch(unit) if unit else None
        if match is None:
            return Origin()
        display = int(match.group(1))
        slot = self.db.query_one('SELECT * FROM desktop_slots WHERE display_number = ?',
                                 (display, ))
        return Origin(display, slot)

    # sessions
    def authenticate(self, token, origin: Origin) -> Tuple[Actor, dict]:
        if not isinstance(token, str) or not token:
            raise Refused(SESSION_INVALID, 'sign in first')
        session = self.db.query_one('SELECT * FROM app_sessions WHERE token_hash = ?',
                                    (auth.token_hash(token), ))
        if session is None:
            raise Refused(SESSION_INVALID, 'the session is not known; sign in again')
        now = self.clock.now()
        if session['revoked_at'] is not None:
            raise Refused(SESSION_REVOKED, 'the session was ended; sign in again')
        idle = timedelta(minutes=self.setting('session_idle_minutes'))
        if parse(session['expires_at']) <= now or parse(session['last_seen_at']) + idle <= now:
            raise Refused(SESSION_EXPIRED, 'the session expired; sign in again')
        if session['desktop_slot_id'] != origin.slot_id:
            raise Refused(SESSION_INVALID, 'the session belongs to another desktop')
        user = self.db.query_one('SELECT * FROM users WHERE id = ?', (session['user_id'], ))
        if user['status'] != 'active':
            raise Refused(SESSION_REVOKED, f'the account is {user["status"]}')
        if user['credential_setup_required']:
            raise Refused(SETUP_REQUIRED, 'set up your password with your setup token')
        if user['authz_version'] != session['authz_version_at_issue']:
            raise Refused(SESSION_STALE, 'your permissions changed; sign in again')
        if (now - parse(session['last_seen_at'])).total_seconds() >= LAST_SEEN_EVERY_S:
            with self.db.transaction() as cursor:
                cursor.execute('UPDATE app_sessions SET last_seen_at = ? WHERE id = ?',
                               (iso(now), session['id']))
        actor = Actor(user['id'], user['login_name'], user['display_name'], user['role'],
                      session['id'], session['desktop_slot_id'], user['id'] == self.owner_id())
        return actor, session

    def require_fresh(self, ctx: Context) -> None:
        """For role changes, Remove and the settings: the password entered recently."""
        moment = parse(ctx.session['last_reauthenticated_at'] or ctx.session['issued_at'])
        if moment + timedelta(minutes=self.setting('reauth_minutes')) < self.clock.now():
            raise Refused(REAUTH_REQUIRED, 'enter your password again first')

    def issue_session(self, cursor, user: dict, origin: Origin) -> dict:
        token = auth.new_session_token()
        now = self.clock.now()
        expires = now + timedelta(hours=self.setting('session_max_hours'))
        session_id = new_id()
        cursor.execute(
            'INSERT INTO app_sessions (id, user_id, token_hash, desktop_slot_id, issued_at, '
            'expires_at, last_seen_at, authz_version_at_issue, last_reauthenticated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (session_id, user['id'], auth.token_hash(token), origin.slot_id, iso(now),
             iso(expires), iso(now), user['authz_version'], iso(now)))
        cursor.execute('UPDATE users SET last_login_at = ? WHERE id = ?', (iso(now), user['id']))
        return {'token': token, 'session_id': session_id, 'expires_at': iso(expires)}

    def revoke_sessions(self, cursor, user_id: str, keep: Optional[str] = None) -> int:
        cursor.execute('UPDATE app_sessions SET revoked_at = ? WHERE user_id = ? AND '
                       'revoked_at IS NULL AND id IS NOT ?', (self.now(), user_id, keep))
        return self.db.conn.changes()

    @staticmethod
    def signs_in_here(user: Optional[dict], origin: Origin) -> bool:
        """A remote desktop is its user's: nobody else signs in there, and a name nobody has is
        nobody's. The console takes any name (an unknown one fails at its password)."""
        if not origin.is_desktop:
            return True
        return user is not None and origin.slot is not None and origin.slot['user_id'] == user['id']

    def user_row(self, user_id: str) -> dict:
        row = self.db.query_one('SELECT * FROM users WHERE id = ?', (user_id, ))
        if row is None:
            raise Refused(NOT_FOUND, 'no such user')
        return row

    def run_user_hooks(self, cursor, before: Optional[dict], after: dict, actor_id, now,
                       result: Optional[dict] = None) -> None:
        for hook in self.user_hooks:
            hook(cursor, before, after, actor_id, now, {} if result is None else result)

    def setup_token(self, cursor, user_id: str) -> str:
        """A new one-time setup token for the user (the previous one stops working)."""
        token = auth.new_setup_token()
        cursor.execute('UPDATE users SET setup_token_hash = ?, setup_token_expires_at = ? '
                       'WHERE id = ?', (auth.token_hash(auth.normalize_setup_token(token)),
                                        self.clock.iso_in(hours=self.setting('setup_token_hours')),
                                        user_id))
        return token


# sign-in and the caller's own session

@op(public=True)
def hello(service: Service, ctx: Context) -> dict:
    return {'service_version': SERVICE_VERSION, 'protocol_version': PROTOCOL_VERSION,
            'server_time': service.now(), 'owner_configured': service.owner_id() is not None,
            'origin': {'display': ctx.origin.display, 'assigned': ctx.origin.slot is not None}}


def _sign_in_user(service: Service, ctx: Context, action: str, login_name,
                  refusal: str) -> Tuple[str, Optional[dict]]:
    """(the normalized name, its user or None); a malformed name is nobody, as an unknown one.
    At a desktop, any name but its own user's (known or not) is refused here, before any password
    or token is looked at, with the same words as a wrong one: it tells nothing, and counts
    against the desktop only."""
    try:
        name = validate.login_name(login_name)
        user = service.db.query_one('SELECT * FROM users WHERE login_name = ?', (name, ))
    except Refused:
        name, user = '?', None
    service.limiter.check_origin(ctx.origin.key)
    if not service.signs_in_here(user, ctx.origin):
        _failed_sign_in(service, ctx, action, name, user, WRONG_DESKTOP)
        raise Refused(INVALID_CREDENTIALS, refusal)
    service.limiter.check(name, ctx.origin.key)
    return name, user


def _failed_sign_in(service: Service, ctx: Context, action: str, name: str,
                    user: Optional[dict], code: str = INVALID_CREDENTIALS) -> None:
    service.limiter.failed_sign_in(name, ctx.origin.key, code != WRONG_DESKTOP)
    with service.db.transaction() as cursor:
        audit.record(cursor, service.now(), action, audit.FAILED, None, audit.SYSTEM, 'user',
                     user['id'] if user else None, ctx.request_id, error_code=code,
                     after={'login_name': name, 'origin': ctx.origin.key})


WRONG_PASSWORD = 'wrong login name or password (first time? use your setup token)'
WRONG_TOKEN = 'wrong login name or setup token, or the token expired: ask for a new one'


@op(public=True)
def login(service: Service, ctx: Context, login_name, password) -> dict:
    name, user = _sign_in_user(service, ctx, 'login', login_name, WRONG_PASSWORD)
    usable = user is not None and user['status'] == 'active' and \
        not user['credential_setup_required']
    if not service.passwords.verify(user['password_hash'] if usable else None, password):
        _failed_sign_in(service, ctx, 'login', name, user)
        raise Refused(INVALID_CREDENTIALS, WRONG_PASSWORD)
    service.limiter.succeeded(name, ctx.origin.key)
    with service.db.transaction() as cursor:
        issued = service.issue_session(cursor, user, ctx.origin)
        audit.record(cursor, service.now(), 'login', audit.SUCCEEDED, user['id'],
                     target_type='session', target_id=issued['session_id'],
                     request_id=ctx.request_id, after={'origin': ctx.origin.key})
    return dict(issued, user=public_user(service.user_row(user['id'])))


@op(public=True)
def setup_credential(service: Service, ctx: Context, login_name, setup_token,
                     new_password) -> dict:
    """The one-time setup (or reset) of a user's password with the token a supervisor or admin
    handed them. It sets that user's password only: never a role or anything else."""
    name, user = _sign_in_user(service, ctx, 'setup_credential', login_name, WRONG_TOKEN)
    digest = auth.token_hash(auth.normalize_setup_token(setup_token))
    now = service.now()
    valid = (user is not None and user['status'] in ('pending', 'active')
             and auth.same_hash(user['setup_token_hash'], digest)
             and user['setup_token_expires_at'] > now)
    if not valid:
        _failed_sign_in(service, ctx, 'setup_credential', name, user)
        raise Refused(INVALID_CREDENTIALS, WRONG_TOKEN)
    password_hash = service.passwords.hash(service.passwords.check_policy(new_password, name))
    with service.db.transaction() as cursor:
        # consumed atomically: a second use of the same token finds it gone
        cursor.execute(
            "UPDATE users SET password_hash = ?, credential_setup_required = 0, "
            "setup_token_hash = NULL, setup_token_expires_at = NULL, "
            "status = CASE status WHEN 'pending' THEN 'active' ELSE status END, "
            "authz_version = authz_version + 1, row_version = row_version + 1, updated_at = ? "
            "WHERE id = ? AND setup_token_hash = ? AND setup_token_expires_at > ? "
            "AND status IN ('pending', 'active')", (password_hash, now, user['id'], digest, now))
        if service.db.conn.changes() != 1:
            raise Refused(INVALID_CREDENTIALS, 'the setup token was used meanwhile')
        service.revoke_sessions(cursor, user['id'])
        after = cursor.execute('SELECT * FROM users WHERE id = ?', (user['id'], )).fetchall()[0]
        service.run_user_hooks(cursor, user, after, user['id'], now)
        audit.record(cursor, now, 'setup_credential', audit.SUCCEEDED, user['id'],
                     target_type='user', target_id=user['id'], request_id=ctx.request_id,
                     before={'status': user['status']}, after={'status': after['status']})
        issued = service.issue_session(cursor, after, ctx.origin)
    service.limiter.succeeded(name, ctx.origin.key)
    return dict(issued, user=public_user(service.user_row(user['id'])))


def permissions(actor: Actor) -> dict:
    return {'manage_annotators': actor.role in ('admin', 'supervisor'),
            'manage_supervisors': actor.role == 'admin',
            'settings': actor.role == 'admin', 'audit_all': actor.role == 'admin',
            'all_jobs': actor.role == 'admin'}


@op()
def whoami(service: Service, ctx: Context) -> dict:
    return {'user': public_user(service.user_row(ctx.actor.user_id)), 'is_owner':
            ctx.actor.is_owner, 'permissions': permissions(ctx.actor),
            'session': {'id': ctx.session['id'], 'expires_at': ctx.session['expires_at'],
                        'origin': ctx.origin.key}}


@op()
def logout(service: Service, ctx: Context) -> None:
    with service.db.transaction() as cursor:
        cursor.execute('UPDATE app_sessions SET revoked_at = ? WHERE id = ?',
                       (service.now(), ctx.session['id']))
        audit.record(cursor, service.now(), 'logout', audit.SUCCEEDED, ctx.actor.user_id,
                     target_type='session', target_id=ctx.session['id'])


@op()
def reauthenticate(service: Service, ctx: Context, password) -> dict:
    user = service.user_row(ctx.actor.user_id)
    service.limiter.check(user['login_name'], ctx.origin.key)
    if not service.passwords.verify(user['password_hash'], password):
        _failed_sign_in(service, ctx, 'reauthenticate', user['login_name'], user)
        raise Refused(INVALID_CREDENTIALS, 'wrong password')
    service.limiter.succeeded(user['login_name'], ctx.origin.key)
    with service.db.transaction() as cursor:
        cursor.execute('UPDATE app_sessions SET last_reauthenticated_at = ? WHERE id = ?',
                       (service.now(), ctx.session['id']))
    return {'reauthenticated_at': service.now()}


@op()
def change_password(service: Service, ctx: Context, current_password, new_password) -> None:
    """Your own password; your other sessions end, this one stays."""
    user = service.user_row(ctx.actor.user_id)
    service.limiter.check(user['login_name'], ctx.origin.key)
    if not service.passwords.verify(user['password_hash'], current_password):
        _failed_sign_in(service, ctx, 'change_password', user['login_name'], user)
        raise Refused(INVALID_CREDENTIALS, 'wrong password')
    if new_password == current_password:
        raise Refused(INVALID, 'the new password is the old one')
    password_hash = service.passwords.hash(
        service.passwords.check_policy(new_password, user['login_name']))
    with service.db.transaction() as cursor:
        cursor.execute('UPDATE users SET password_hash = ?, authz_version = authz_version + 1, '
                       'row_version = row_version + 1, updated_at = ? WHERE id = ?',
                       (password_hash, service.now(), user['id']))
        service.revoke_sessions(cursor, user['id'], keep=ctx.session['id'])
        cursor.execute('UPDATE app_sessions SET authz_version_at_issue = '
                       '(SELECT authz_version FROM users WHERE id = ?) WHERE id = ?',
                       (user['id'], ctx.session['id']))
        audit.record(cursor, service.now(), 'change_password', audit.SUCCEEDED, user['id'],
                     target_type='user', target_id=user['id'], request_id=ctx.request_id)
