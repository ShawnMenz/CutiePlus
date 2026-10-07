"""
SQLite metadata of the shared service. One database per task root, on a local filesystem (SQLite
locking is unreliable over network filesystems). Write-ahead logging with full synchronization:
a committed transaction survives a crash of the process or of the machine.
"""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from os import path
from typing import Any, Iterable, Iterator, List, Optional

SCHEMA_VERSION = 1

NETWORK_FILESYSTEMS = {'nfs', 'nfs4', 'cifs', 'smb3', 'smbfs', 'sshfs', 'fuse.sshfs', 'ceph',
                       'glusterfs', '9p', 'afs', 'fuse.rclone', 'davfs', 'fuse.davfs', 'lustre',
                       'gpfs', 'beegfs'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,               -- '<split>/<sequence>'
    workspace TEXT NOT NULL UNIQUE,
    input_revision INTEGER NOT NULL,        -- human inputs of inference (masks, anchors)
    content_revision INTEGER NOT NULL,      -- the whole committed annotation state
    view_revision INTEGER NOT NULL,         -- content revision the legacy files show
    lease_epoch INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,                    -- JSON: prefill, intervals, description, progress
    view_files TEXT NOT NULL DEFAULT '{}',  -- JSON: [size, mtime_ns] of the legacy JSON files
    imported_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    task_id TEXT NOT NULL,
    frame TEXT NOT NULL,
    mask_object TEXT,                       -- SHA-256 of the mask PNG; NULL: no mask file
    anchor TEXT,                            -- JSON anchor entry; NULL: not an anchor
    view_stat TEXT,                         -- JSON [size, mtime_ns] of masks/<frame>.png
    updated_revision INTEGER NOT NULL,
    PRIMARY KEY (task_id, frame)
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    annotator_name TEXT NOT NULL,
    vnc_display TEXT,
    process_id INTEGER,
    host TEXT,
    created_at TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    status TEXT NOT NULL                    -- active | closed | expired
);
CREATE TABLE IF NOT EXISTS leases (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    acquired_at TEXT NOT NULL,
    renewed_at TEXT NOT NULL,
    status TEXT NOT NULL                    -- active | released | expired | lost
);
CREATE TABLE IF NOT EXISTS edits (
    edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    frame TEXT,
    op TEXT NOT NULL,
    request_id TEXT,
    session_id TEXT,
    author TEXT,
    at TEXT NOT NULL,
    before_object TEXT,
    after_object TEXT,
    before_anchor TEXT,
    after_anchor TEXT,
    input_revision INTEGER NOT NULL,
    content_revision INTEGER NOT NULL,
    job_id TEXT,
    attempt_id TEXT
);
CREATE INDEX IF NOT EXISTS edits_task ON edits (task_id, content_revision);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,
    op TEXT NOT NULL,
    created_at TEXT NOT NULL,
    response TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    snapshot TEXT NOT NULL,                 -- JSON: the immutable input snapshot
    effective TEXT,                         -- JSON: effective resource configuration per attempt
    committed_count INTEGER NOT NULL DEFAULT 0,
    last_committed_ti INTEGER,
    stop_reason TEXT,
    error TEXT,
    metrics TEXT
);
CREATE TABLE IF NOT EXISTS exports (
    export_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    session_id TEXT,
    content_revision INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,                   -- running | complete | failed | stale
    detail TEXT
);
CREATE TABLE IF NOT EXISTS worker_starts (
    start_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    generation INTEGER NOT NULL,
    pid INTEGER,
    hello TEXT NOT NULL                     -- JSON: hardware, memory profile, model identity
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class StorageError(Exception):
    pass


def filesystem_type(file_path: str) -> Optional[str]:
    """The type of the filesystem holding file_path (from /proc/mounts), or None."""
    target = path.realpath(file_path)
    best, best_type = '', None
    try:
        with open('/proc/mounts', encoding='utf-8') as mounts:
            for line in mounts:
                fields = line.split()
                if len(fields) < 3:
                    continue
                mount_point = fields[1].replace('\\040', ' ')
                if (target == mount_point or target.startswith(mount_point.rstrip('/') + '/')
                        or mount_point == '/') and len(mount_point) >= len(best):
                    best, best_type = mount_point, fields[2]
    except OSError:
        return None
    return best_type


def check_local_filesystem(directory: str) -> None:
    kind = filesystem_type(directory)
    if kind in NETWORK_FILESYSTEMS:
        raise StorageError(f'{directory} is on a network filesystem ({kind}); keep the service '
                           'database on a local disk (service.state_dir).')


def dumps(value: Any) -> Optional[str]:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(text: Optional[str]) -> Any:
    return None if text is None else json.loads(text)


class Database:

    def __init__(self, file_path: str, allow_network_filesystem: bool = False):
        directory = path.dirname(path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)
        if not allow_network_filesystem:
            check_local_filesystem(directory)
        self.file_path = file_path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(file_path, timeout=30, isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA synchronous=FULL')
            self.conn.execute('PRAGMA busy_timeout=30000')
            self.conn.executescript(SCHEMA)
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self.conn.execute("INSERT INTO meta VALUES ('schema_version', ?)",
                                  (str(SCHEMA_VERSION), ))
            elif int(row['value']) != SCHEMA_VERSION:
                raise StorageError(f'{file_path} has schema {row["value"]}, expected {SCHEMA_VERSION}')
        # tests set this to make the next commit fail, as a full disk would
        self.fail_next_commit: Optional[Exception] = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One short write transaction (BEGIN IMMEDIATE); rolled back on any error."""
        with self.lock:
            self.conn.execute('BEGIN IMMEDIATE')
            try:
                yield self.conn
                if self.fail_next_commit is not None:
                    error, self.fail_next_commit = self.fail_next_commit, None
                    raise error
                self.conn.execute('COMMIT')
            except BaseException:
                self.conn.execute('ROLLBACK')
                raise

    def query(self, sql: str, args: Iterable = ()) -> List[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    def query_one(self, sql: str, args: Iterable = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def close(self) -> None:
        with self.lock:
            self.conn.close()
