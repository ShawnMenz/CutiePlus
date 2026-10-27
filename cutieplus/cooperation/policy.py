"""
The SSH policy the database calls for: one entry per key that may sign in now (an active key, not
expired, of a pending or active user who has a desktop), forwarding to that user's VNC port only.

policy_state.desired_revision goes up with every change that may change it (bump); the reconciler
publishes the snapshot through the root helper and only then records it as applied, with the
checksum of what was written. A database change alone changes no one's SSH access.
"""
from typing import List, Tuple

from cutieplus.cooperation.sshpolicy import PolicyEntry, checksum, render_policy


def bump(cursor) -> int:
    """The policy must be published again; returns the new desired revision."""
    cursor.execute('UPDATE policy_state SET desired_revision = desired_revision + 1 WHERE id = 1')
    return cursor.execute('SELECT desired_revision FROM policy_state WHERE id = 1').fetchall()[0][
        'desired_revision']


def snapshot(db, now: str) -> Tuple[int, List[PolicyEntry]]:
    """(the desired revision, its entries), read together."""
    with db.lock:
        revision = db.query_one('SELECT desired_revision FROM policy_state WHERE id = 1')[
            'desired_revision']
        rows = db.query(
            "SELECT k.id, k.public_key, k.expires_at, s.vnc_port FROM ssh_keys k "
            "JOIN users u ON u.id = k.user_id JOIN desktop_slots s ON s.user_id = u.id "
            "WHERE k.status = 'active' AND u.status IN ('pending', 'active') "
            "AND (k.expires_at IS NULL OR k.expires_at > ?) ORDER BY k.id", (now, ))
    return revision, [PolicyEntry(row['id'], row['public_key'], row['vnc_port'],
                                  row['expires_at']) for row in rows]


def expected_checksum(revision: int, entries: List[PolicyEntry], ports) -> str:
    return checksum(render_policy(revision, entries, ports))
