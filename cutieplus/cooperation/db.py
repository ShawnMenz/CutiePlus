"""
The cooperation database (SQLite through apsw, which bundles a current SQLite). One connection,
serialized by a lock: the service is the only writer. Every connection enforces foreign keys, waits
for a busy database, uses the write-ahead log and full synchronization (authentication and
authorization changes must survive a power cut). Transactions are short: none is held during GPU
inference, a desktop's start or any other long work.

Schema versions are the files migrations/NNNN_*.sql, applied in order, each in a transaction and
recorded in schema_migrations with its checksum; a recorded migration whose file changed stops the
service. A database with data is backed up before a migration is applied.
"""
import hashlib
import os
import re
import threading
from contextlib import contextmanager
from os import path
from typing import Callable, Iterator, List, Optional, Tuple

import apsw

from cutieplus.cooperation.clock import Clock, iso
from cutieplus.cooperation.config import MIN_SQLITE

MIGRATIONS_DIR = path.join(path.dirname(path.abspath(__file__)), 'migrations')
BUSY_TIMEOUT_MS = 5000


class DatabaseError(Exception):
    pass


def sqlite_version() -> Tuple[int, ...]:
    return tuple(int(part) for part in apsw.sqlite_lib_version().split('.'))


def migrations(folder: str = MIGRATIONS_DIR) -> List[Tuple[int, str, str, str]]:
    """(version, description, SQL, checksum) of every migration file, in order."""
    found = []
    for name in sorted(os.listdir(folder)):
        match = re.fullmatch(r'(\d{4})_(\w+)\.sql', name)
        if match is None:
            continue
        with open(path.join(folder, name), encoding='utf-8') as f:
            sql = f.read()
        found.append((int(match.group(1)), match.group(2).replace('_', ' '), sql,
                      hashlib.sha256(sql.encode('utf-8')).hexdigest()))
    return found


def _dict_rows(cursor, row):
    return {description[0]: value for description, value in zip(cursor.get_description(), row)}


class Database:

    def __init__(self, db_path: str, clock: Optional[Clock] = None,
                 require_patched_sqlite: bool = True, migrations_dir: str = MIGRATIONS_DIR):
        if require_patched_sqlite and sqlite_version() < MIN_SQLITE:
            raise DatabaseError(f'SQLite {apsw.sqlite_lib_version()} has the WAL-reset bug; '
                                f'{".".join(map(str, MIN_SQLITE))} or later is needed')
        self.path = db_path
        self.clock = clock or Clock()
        self.lock = threading.RLock()
        self.committed: List[Callable[[], None]] = []  # after_commit's, of the transaction
        if not path.exists(db_path):
            # private from the start; SQLite gives its -wal and -shm files the same mode
            os.close(os.open(db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
        self.conn = apsw.Connection(db_path)
        self.conn.set_busy_timeout(BUSY_TIMEOUT_MS)
        self.conn.row_trace = _dict_rows
        self.conn.execute('PRAGMA foreign_keys = ON')
        mode = self.conn.execute('PRAGMA journal_mode = WAL').fetchall()[0]['journal_mode']
        if mode != 'wal':
            raise DatabaseError(f'{db_path}: the write-ahead log is unavailable ({mode})')
        self.conn.execute('PRAGMA synchronous = FULL')
        self.migrate(migrations_dir)

    # transactions
    @contextmanager
    def transaction(self) -> Iterator[apsw.Cursor]:
        """A write transaction, taken at once (BEGIN IMMEDIATE): committed on success, rolled
        back on any exception."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('BEGIN IMMEDIATE')
            self.committed = []
            try:
                yield cursor
            except BaseException:
                cursor.execute('ROLLBACK')
                self.committed = []
                raise
            cursor.execute('COMMIT')
            callbacks, self.committed = self.committed, []
        for callback in callbacks:
            callback()

    def after_commit(self, callback: Callable[[], None]) -> None:
        """Runs callback once the current transaction is committed; never if it rolls back."""
        self.committed.append(callback)

    def query(self, sql: str, bindings=()) -> List[dict]:
        with self.lock:
            return self.conn.execute(sql, bindings).fetchall()

    def query_one(self, sql: str, bindings=()) -> Optional[dict]:
        rows = self.query(sql, bindings)
        return rows[0] if rows else None

    # schema versions
    def schema_version(self) -> int:
        row = self.query_one("SELECT name FROM sqlite_master WHERE name = 'schema_migrations'")
        if row is None:
            return 0
        return self.query_one('SELECT max(version) AS v FROM schema_migrations')['v'] or 0

    def migrate(self, folder: str = MIGRATIONS_DIR) -> None:
        with self.lock:
            self.conn.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER '
                              'PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL, '
                              'checksum TEXT NOT NULL)')
            applied = {row['version']: row['checksum']
                       for row in self.query('SELECT version, checksum FROM schema_migrations')}
            pending = []
            for version, description, sql, checksum in migrations(folder):
                if version in applied:
                    if applied[version] != checksum:
                        raise DatabaseError(f'migration {version} changed after it was applied')
                else:
                    pending.append((version, description, sql, checksum))
            if pending and applied:
                self.backup(f'{self.path}.before-{pending[0][0]:04d}')
            for version, description, sql, checksum in pending:
                with self.transaction() as cursor:
                    cursor.execute(sql)
                    cursor.execute('INSERT INTO schema_migrations VALUES (?, ?, ?, ?)',
                                   (version, description, iso(self.clock.now()), checksum))

    # backups (SQLite's online backup)
    def backup(self, destination: str) -> str:
        temporary = destination + '.partial'
        if path.exists(temporary):
            os.remove(temporary)
        with self.lock:
            target = apsw.Connection(temporary)
            try:
                with target.backup('main', self.conn, 'main') as copy:
                    while not copy.done:
                        copy.step(256)
                check = target.execute('PRAGMA integrity_check').fetchall()[0][0]
            finally:
                target.close()
        if check != 'ok':
            os.remove(temporary)
            raise DatabaseError(f'the backup failed its integrity check: {check}')
        os.replace(temporary, destination)
        return destination

    def close(self) -> None:
        with self.lock:
            self.conn.close()
