"""
The cooperation database (cutieplus/cooperation/db.py): settings of every connection, versioned
migrations, constraints, rollback, and the online backup.
"""
import apsw
import pytest

from cutieplus.cooperation.db import Database, DatabaseError, migrations, sqlite_version
from cutieplus.cooperation.config import MIN_SQLITE

NOW = '2026-10-27T12:00:00.000000Z'


def add_user(cursor, user_id='u1', login='alice', role='annotator'):
    cursor.execute('INSERT INTO users (id, login_name, display_name, role, status, created_at, '
                   'updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                   (user_id, login, login.title(), role, 'pending', NOW, NOW))


def test_the_runtime_has_the_wal_fix_and_every_table(tmp_path):
    assert sqlite_version() >= MIN_SQLITE
    db = Database(str(tmp_path / 'c.sqlite3'))
    assert (tmp_path / 'c.sqlite3').stat().st_mode & 0o777 == 0o600
    assert (tmp_path / 'c.sqlite3-wal').stat().st_mode & 0o777 == 0o600
    assert db.query_one('PRAGMA journal_mode')['journal_mode'] == 'wal'
    assert db.query_one('PRAGMA foreign_keys')['foreign_keys'] == 1
    assert db.query_one('PRAGMA synchronous')['synchronous'] == 2  # FULL
    tables = {row['name'] for row in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'users', 'ssh_keys', 'app_sessions', 'desktop_slots', 'ssh_connections', 'gpu_jobs',
            'gpu_state', 'management_operations', 'policy_state', 'audit_events',
            'system_settings', 'schema_migrations'} <= tables
    assert db.schema_version() == len(migrations())


def test_constraints_hold_and_a_failed_transaction_leaves_nothing(tmp_path):
    db = Database(str(tmp_path / 'c.sqlite3'))
    with pytest.raises(apsw.ConstraintError):  # no such user: foreign keys are enforced
        with db.transaction() as cursor:
            cursor.execute("INSERT INTO ssh_keys (id, user_id, public_key, fingerprint_sha256, "
                           "device_label, status, created_at) VALUES ('k', 'nobody', 'x', 'f', "
                           "'laptop', 'active', ?)", (NOW, ))
    with pytest.raises(apsw.ConstraintError):  # an unknown role
        with db.transaction() as cursor:
            add_user(cursor, role='root')
    with pytest.raises(RuntimeError):
        with db.transaction() as cursor:
            add_user(cursor)
            raise RuntimeError('half way')
    assert db.query('SELECT * FROM users') == []


def test_a_changed_migration_stops_the_service_and_a_backup_restores(tmp_path):
    folder = tmp_path / 'migrations'
    folder.mkdir()
    (folder / '0001_first.sql').write_text('CREATE TABLE t (x INTEGER);')
    db_path = str(tmp_path / 'c.sqlite3')
    db = Database(db_path, migrations_dir=str(folder))
    with db.transaction() as cursor:
        cursor.execute('INSERT INTO t VALUES (1)')
    backup = db.backup(str(tmp_path / 'backup.sqlite3'))
    # a second migration: the database is backed up before it is applied
    (folder / '0002_second.sql').write_text('ALTER TABLE t ADD COLUMN y TEXT;')
    db.close()
    db = Database(db_path, migrations_dir=str(folder))
    assert db.schema_version() == 2 and (tmp_path / 'c.sqlite3.before-0002').exists()
    db.close()
    (folder / '0001_first.sql').write_text('CREATE TABLE t (x TEXT);')
    with pytest.raises(DatabaseError):
        Database(db_path, migrations_dir=str(folder))
    restored = apsw.Connection(backup)
    assert restored.execute('SELECT x FROM t').fetchall() == [(1, )]
