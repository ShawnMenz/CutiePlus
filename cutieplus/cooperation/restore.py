"""
After a database backup is restored (cutieplus-admin restore), the database is older than the
system around it. The first service to open it reconciles, on the safe side:

- every application session ends (sign in again);
- jobs that were running are interrupted (their worker is long gone);
- a key the database calls active but the published SSH policy no longer holds was revoked after
  the backup: it is revoked again, never re-admitted;
- the SSH policy is published again from the restored database (a key added after the backup is
  not in it: add it again), and desktops are brought to the restored desired states.

Users changed after the backup (a suspension, say) come back as the backup had them: the restore
command lists the users for the admin to review.
"""
import json
import re
from typing import Optional, Set

from cutieplus.cooperation import audit, policy, settings

RESTORED_FROM = 'restored_from'  # set by the restore command; not an editable setting
POLICY_KEY = re.compile(r' cutieplus:([0-9a-f-]{36})$')


def published_key_ids(authorized_keys: str) -> Optional[Set[str]]:
    """The key ids the live policy file admits; None when it cannot be read."""
    try:
        with open(authorized_keys, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    return {match.group(1) for match in map(POLICY_KEY.search, lines) if match}


def mark_restored(db, source: str, now: str) -> None:
    with db.transaction() as cursor:
        settings.write_setting(cursor, RESTORED_FROM, {'source': source, 'at': now}, None, now)
        audit.record(cursor, now, 'restore', audit.SUCCEEDED, None, audit.SYSTEM, 'database',
                     source)


def reconcile_after_restore(service) -> Optional[dict]:
    """Runs once after a restore (then forgets it); None when the database was not restored."""
    db = service.db
    marker = db.query_one('SELECT value_json FROM system_settings WHERE key = ?',
                          (RESTORED_FROM, ))
    if marker is None:
        return None
    now = service.now()
    published = published_key_ids(service.config.authorized_keys)
    with db.transaction() as cursor:
        cursor.execute('UPDATE app_sessions SET revoked_at = ? WHERE revoked_at IS NULL', (now, ))
        sessions = db.conn.changes()
        cursor.execute("UPDATE gpu_jobs SET status = 'interrupted', finished_at = ?, "
                       "error_code = 'restored', error_summary = 'the database was restored' "
                       "WHERE status IN ('running', 'cancel_requested')", (now, ))
        jobs = db.conn.changes()
        revoked = []
        if published is not None:
            for key in cursor.execute("SELECT id, user_id FROM ssh_keys WHERE status = "
                                      "'active'").fetchall():
                if key['id'] not in published:
                    cursor.execute("UPDATE ssh_keys SET status = 'revoked', revoked_at = ? "
                                   'WHERE id = ?', (now, key['id']))
                    revoked.append(key['id'])
                    audit.record(cursor, now, 'restore_revoke_key', audit.SUCCEEDED, None,
                                 audit.SYSTEM, 'ssh_key', key['id'],
                                 after={'reason': 'not in the published policy at restore'})
        cursor.execute("UPDATE ssh_connections SET status = 'closed', disconnected_at = ?, "
                       "disconnect_reason = 'restored' WHERE status != 'closed'", (now, ))
        revision = policy.bump(cursor)
        cursor.execute('DELETE FROM system_settings WHERE key = ?', (RESTORED_FROM, ))
        summary = {'source': json.loads(marker['value_json']).get('source'),
                   'sessions_ended': sessions, 'jobs_interrupted': jobs,
                   'keys_revoked': revoked, 'policy_revision': revision,
                   'policy_file_read': published is not None}
        audit.record(cursor, now, 'restore_reconciled', audit.SUCCEEDED, None, audit.SYSTEM,
                     'database', None, after=summary)
    service.reconcile_soon()
    return summary
