"""
cutieplus-admin: the protected maintenance commands, run by root on the host (sudo). They are the
only way to create the owner, recover the owner's access or hand the ownership to someone else; no
request of the application can. Each drops root for the service account before it opens the
database, so the files it writes stay the service's.

    sudo cutieplus-admin bootstrap-owner --login <name> --name "<display name>"
    sudo cutieplus-admin owner-reset          # a new setup token for the owner
    sudo cutieplus-admin owner-transfer --login <name>
    sudo cutieplus-admin backup
    sudo cutieplus-admin status
"""
import argparse
import json
import os
import pwd
import shutil
import subprocess
import sys
from typing import Optional

from cutieplus.cooperation import audit, settings, validate
from cutieplus.cooperation.clock import new_id
from cutieplus.cooperation.config import SERVICE_CONFIG, ServiceConfig
from cutieplus.cooperation.errors import Refused
from cutieplus.cooperation.service import Service

SERVICE_ACCOUNT = 'cutieplus'


class AdminError(Exception):
    pass


def bootstrap_owner(service: Service, login_name: str, display_name: str) -> str:
    """Creates the owner, an admin, and returns their one-time setup token. Refused once an owner
    exists."""
    if service.owner_id() is not None:
        raise AdminError('the owner exists already; use owner-reset or owner-transfer')
    try:
        name = validate.login_name(login_name)
        display_name = validate.text(display_name, 'display name', 80)
    except Refused as exc:
        raise AdminError(exc.message) from None
    now = service.now()
    with service.db.transaction() as cursor:
        if cursor.execute('SELECT 1 FROM users WHERE login_name = ?', (name, )).fetchall():
            raise AdminError(f'{name} is taken')
        user_id = new_id()
        cursor.execute(
            'INSERT INTO users (id, login_name, display_name, role, status, '
            "credential_setup_required, created_at, updated_at) VALUES (?, ?, ?, 'admin', "
            "'pending', 1, ?, ?)", (user_id, name, display_name, now, now))
        token = service.setup_token(cursor, user_id)
        settings.write_setting(cursor, settings.OWNER_USER_ID, user_id, None, now)
        audit.record(cursor, now, 'bootstrap_owner', audit.SUCCEEDED, None, audit.SYSTEM, 'user',
                     user_id, after={'login_name': name, 'role': 'admin'})
    return token


def _owner(service: Service) -> dict:
    owner_id = service.owner_id()
    if owner_id is None:
        raise AdminError('there is no owner yet: run bootstrap-owner')
    return service.user_row(owner_id)


def owner_reset(service: Service) -> str:
    """The owner's password is cleared and their sessions end; returns a new setup token."""
    owner = _owner(service)
    now = service.now()
    with service.db.transaction() as cursor:
        token = service.setup_token(cursor, owner['id'])
        cursor.execute("UPDATE users SET password_hash = NULL, credential_setup_required = 1, "
                       "status = 'active', role = 'admin', authz_version = authz_version + 1, "
                       "row_version = row_version + 1, updated_at = ? WHERE id = ?",
                       (now, owner['id']))
        service.revoke_sessions(cursor, owner['id'])
        audit.record(cursor, now, 'owner_reset', audit.SUCCEEDED, None, audit.SYSTEM, 'user',
                     owner['id'])
    return token


def owner_transfer(service: Service, login_name: str) -> None:
    """Hands the ownership to an active user, who becomes the admin; the previous owner becomes a
    supervisor. Both sign in again."""
    owner = _owner(service)
    name = validate.login_name(login_name)
    target = service.db.query_one('SELECT * FROM users WHERE login_name = ?', (name, ))
    if target is None or target['status'] != 'active':
        raise AdminError(f'{name} is not an active user')
    if target['id'] == owner['id']:
        raise AdminError(f'{name} is the owner already')
    now = service.now()
    with service.db.transaction() as cursor:
        for user_id, role in ((target['id'], 'admin'), (owner['id'], 'supervisor')):
            cursor.execute('UPDATE users SET role = ?, authz_version = authz_version + 1, '
                           'row_version = row_version + 1, updated_at = ? WHERE id = ?',
                           (role, now, user_id))
            service.revoke_sessions(cursor, user_id)
        settings.write_setting(cursor, settings.OWNER_USER_ID, target['id'], None, now)
        audit.record(cursor, now, 'owner_transfer', audit.SUCCEEDED, None, audit.SYSTEM, 'user',
                     target['id'], before={'owner': owner['login_name']},
                     after={'owner': target['login_name']})


def status(service: Service) -> dict:
    db = service.db
    owner = service.owner_id()
    return {
        'schema_version': db.schema_version(),
        'owner': (db.query_one('SELECT login_name FROM users WHERE id = ?', (owner, )) or
                  {}).get('login_name'),
        'users': {row['status']: row['n'] for row in db.query(
            'SELECT status, count(*) AS n FROM users GROUP BY status')},
        'policy': db.query_one('SELECT * FROM policy_state WHERE id = 1'),
        'gpu': db.query('SELECT * FROM gpu_state'),
        'jobs': {row['status']: row['n'] for row in db.query(
            'SELECT status, count(*) AS n FROM gpu_jobs GROUP BY status')},
        'pending_operations': db.query_one(
            "SELECT count(*) AS n FROM management_operations WHERE status IN "
            "('pending', 'running', 'failed')")['n'],
    }


def restore_file(config: ServiceConfig, backup: str, now: str) -> list:
    """Puts a backup in place of the database (the service is stopped) and marks it restored,
    so the next service reconciles it (restore.py). Returns the users, for review."""
    import apsw
    from cutieplus.cooperation.db import Database
    from cutieplus.cooperation.restore import mark_restored
    source = apsw.Connection(f'file:{backup}?immutable=1',
                             flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI)
    try:
        check = source.execute('PRAGMA integrity_check').fetchall()[0][0]
    finally:
        source.close()
    if check != 'ok':
        raise AdminError(f'{backup} fails its integrity check: {check}')
    temporary = config.db_path + '.restoring'
    shutil.copyfile(backup, temporary)
    os.chmod(temporary, 0o600)
    for suffix in ('-wal', '-shm'):
        if os.path.exists(config.db_path + suffix):
            os.remove(config.db_path + suffix)
    os.replace(temporary, config.db_path)
    db = Database(config.db_path, require_patched_sqlite=config.require_patched_sqlite)
    try:
        mark_restored(db, backup, now)
        return db.query('SELECT login_name, role, status, updated_at FROM users ORDER BY '
                        'login_name')
    finally:
        db.close()


def drop_root(account: str = SERVICE_ACCOUNT) -> None:
    """Root becomes the service account (its group only), for good."""
    if os.geteuid() != 0:
        raise AdminError('run it as root: sudo cutieplus-admin ...')
    entry = pwd.getpwnam(account)
    os.setgroups([])
    os.setgid(entry.pw_gid)
    os.setuid(entry.pw_uid)
    os.umask(0o077)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog='cutieplus-admin', description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', default=SERVICE_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    bootstrap = commands.add_parser('bootstrap-owner', help='create the owner (an admin)')
    bootstrap.add_argument('--login', required=True)
    bootstrap.add_argument('--name', required=True, help='display name')
    commands.add_parser('owner-reset', help="a new setup token for the owner's password")
    transfer = commands.add_parser('owner-transfer', help='hand the ownership to another user')
    transfer.add_argument('--login', required=True)
    commands.add_parser('backup', help='back the database up now')
    restore = commands.add_parser('restore', help='put a backup in place of the database')
    restore.add_argument('backup')
    commands.add_parser('status', help='users, policy, GPU and jobs at a glance')
    args = parser.parse_args(argv)
    try:
        config = ServiceConfig.load(args.config)
        if args.command == 'restore':
            return restore_command(config, args.backup)
        drop_root()
        from cutieplus.cooperation.app import build_service
        service = build_service(config)
        if args.command == 'bootstrap-owner':
            token = bootstrap_owner(service, args.login, args.name)
            print(f'The owner {args.login} is created. Sign in to Cutie at the monitor with '
                  f'"First time" and this setup token (shown once, valid '
                  f'{service.setting("setup_token_hours")} h):\n\n    {token}\n')
        elif args.command == 'owner-reset':
            print(f'Setup token for the owner (shown once):\n\n    {owner_reset(service)}\n')
        elif args.command == 'owner-transfer':
            owner_transfer(service, args.login)
            print(f'{args.login} is the owner now; both of you sign in again.')
        elif args.command == 'backup':
            from cutieplus.cooperation.backups import backup_now
            print(backup_now(service))
        else:
            print(json.dumps(status(service), indent=2, default=str))
    except (AdminError, OSError, KeyError) as exc:
        print(f'cutieplus-admin: {exc}', file=sys.stderr)
        return 1
    return 0


def restore_command(config: ServiceConfig, backup: str) -> int:
    """As root: stops the service, restores as the service account (a child process that drops
    root), and starts the service again, which reconciles the restored database."""
    if os.geteuid() != 0:
        raise AdminError('run it as root: sudo cutieplus-admin restore <backup>')
    backup = os.path.abspath(backup)
    subprocess.run(['/usr/bin/systemctl', 'stop', 'cutieplus-service.service'], check=True)
    child = os.fork()
    if child == 0:
        code = 0
        try:
            drop_root()
            from cutieplus.cooperation.clock import Clock
            users = restore_file(config, backup, Clock().iso_now())
            print(f'Restored {backup}. Review these users: changes made after the backup are '
                  'not in it.')
            for user in users:
                print(f'  {user["login_name"]:<20} {user["role"]:<11} {user["status"]:<10} '
                      f'changed {user["updated_at"][:16]}')
        except Exception as exc:  # the parent says it failed
            print(f'cutieplus-admin: {exc}', file=sys.stderr)
            code = 1
        os._exit(code)
    _, status = os.waitpid(child, 0)
    subprocess.run(['/usr/bin/systemctl', 'start', 'cutieplus-service.service'], check=True)
    return 0 if os.waitstatus_to_exitcode(status) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
