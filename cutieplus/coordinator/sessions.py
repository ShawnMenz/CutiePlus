"""
Labeling sessions: one per GUI process, identified by a UUID, with the annotator found from the
VNC display as before (gui/annotators.py), the process id and the time it was last heard from.
A session that sends no heartbeat for `timeout` seconds expires: its leases are released and
its jobs and interactions stopped (on_expire). Session ids coordinate operations between trusted
users of one Linux account; they are not authentication.
"""
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from cutieplus.storage.db import Database

log = logging.getLogger(__name__)


class SessionError(Exception):
    pass


def now_iso(timestamp: Optional[float] = None) -> str:
    moment = datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc)
    return moment.isoformat(timespec='microseconds')


def iso_timestamp(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


class SessionManager:

    # every request of a window counts as a heartbeat; last_seen is written at most this often
    WRITE_INTERVAL = 1.0

    def __init__(self, db: Database, timeout: float = 30.0,
                 on_expire: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.db = db
        self.timeout = timeout
        self.on_expire = on_expire
        self.clock = clock
        self.lock = threading.Lock()
        self.written: Dict[str, float] = {}  # active session -> when last_seen was last written

    def register(self, annotator_name: str, vnc_display: Optional[str], process_id: Optional[int],
                 host: Optional[str], session_id: Optional[str] = None) -> dict:
        """A new session, or the same one again after a reconnect (while it has not expired)."""
        now = now_iso(self.clock())
        with self.lock, self.db.transaction() as conn:
            if session_id is not None:
                row = conn.execute('SELECT * FROM sessions WHERE session_id = ?',
                                   (session_id, )).fetchone()
                if row is not None and row['status'] == 'active':
                    conn.execute('UPDATE sessions SET last_seen = ? WHERE session_id = ?',
                                 (now, session_id))
                    return {'session_id': session_id, 'resumed': True}
                if row is not None:
                    raise SessionError(f'session {session_id} is {row["status"]}; start a new one')
            session_id = session_id or uuid.uuid4().hex
            conn.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                         (session_id, annotator_name, vnc_display, process_id, host, now, now,
                          'active'))
        log.info('session %s: %s on %s (pid %s)', session_id, annotator_name, vnc_display, process_id)
        return {'session_id': session_id, 'resumed': False}

    def heartbeat(self, session_id: str) -> bool:
        """False when the session is not active (expired or closed). last_seen is written at most
        once per WRITE_INTERVAL, without waiting for the disk: a window polls a running
        propagation many times a second, and losing last_seen in a crash is harmless (a restarted
        service gives every session a full timeout)."""
        now = self.clock()
        with self.lock:
            last = self.written.get(session_id)
            if last is not None and now - last < self.WRITE_INTERVAL:
                return True
            with self.db.transaction(durable=False) as conn:
                row = conn.execute('SELECT status FROM sessions WHERE session_id = ?',
                                   (session_id, )).fetchone()
                if row is None or row['status'] != 'active':
                    self.written.pop(session_id, None)
                    return False
                conn.execute('UPDATE sessions SET last_seen = ? WHERE session_id = ?',
                             (now_iso(now), session_id))
            self.written[session_id] = now
            return True

    def get(self, session_id: str) -> Optional[dict]:
        row = self.db.query_one('SELECT * FROM sessions WHERE session_id = ?', (session_id, ))
        return dict(row) if row is not None else None

    def is_active(self, session_id: str) -> bool:
        session = self.get(session_id)
        return session is not None and session['status'] == 'active'

    def close(self, session_id: str) -> None:
        self._end(session_id, 'closed')

    def _end(self, session_id: str, status: str) -> bool:
        with self.lock, self.db.transaction() as conn:
            self.written.pop(session_id, None)
            changed = conn.execute("UPDATE sessions SET status = ? WHERE session_id = ? AND status = "
                                   "'active'", (status, session_id)).rowcount
        if changed and self.on_expire is not None:
            self.on_expire(session_id)
        return bool(changed)

    def expire_stale(self) -> List[str]:
        """Expires the sessions not heard from for `timeout` seconds."""
        limit = self.clock() - self.timeout
        stale = [row['session_id'] for row in
                 self.db.query("SELECT session_id, last_seen FROM sessions WHERE status = 'active'")
                 if iso_timestamp(row['last_seen']) < limit]
        expired = [session_id for session_id in stale if self._end(session_id, 'expired')]
        for session_id in expired:
            log.warning('session %s expired (no heartbeat for %.0f s)', session_id, self.timeout)
        return expired

    def grant_grace(self) -> int:
        """After a restart of the service: every active session gets a full timeout to reconnect."""
        with self.lock, self.db.transaction() as conn:
            return conn.execute("UPDATE sessions SET last_seen = ? WHERE status = 'active'",
                                (now_iso(self.clock()), )).rowcount

    def active(self) -> List[dict]:
        return [dict(row) for row in
                self.db.query("SELECT * FROM sessions WHERE status = 'active' ORDER BY created_at")]
