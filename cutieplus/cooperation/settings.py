"""
The settings an Admin may change while the service runs (system_settings), each with its bounds.
Anything else in system_settings is kept by the service itself: the owner's id is set only by the
protected maintenance commands (admin_cli), never through the settings API. The GPU runs one job
at a time; that is not a setting.
"""
import json
from dataclasses import dataclass
from typing import Any, Dict

from cutieplus.cooperation.errors import INVALID, NOT_FOUND, Refused

OWNER_USER_ID = 'owner_user_id'  # not editable here


@dataclass(frozen=True)
class Setting:
    default: int
    low: int
    high: int
    description: str


SETTINGS: Dict[str, Setting] = {
    'session_idle_minutes': Setting(720, 5, 10080, 'a session unused this long ends'),
    'session_max_hours': Setting(168, 1, 720, 'every session ends this long after sign-in'),
    'reauth_minutes': Setting(10, 1, 120,
                              'role changes, Remove and settings need the password this recently'),
    'setup_token_hours': Setting(72, 1, 336, 'a setup or reset token expires after this'),
    'pending_jobs_per_user': Setting(3, 1, 50, 'propagations one user may have waiting'),
    'backup_interval_hours': Setting(24, 1, 168, 'how often the database is backed up'),
    'backup_keep': Setting(14, 1, 365, 'how many backups are kept'),
}


def read_setting(db, key: str) -> Any:
    row = db.query_one('SELECT value_json FROM system_settings WHERE key = ?', (key, ))
    if row is not None:
        return json.loads(row['value_json'])
    return SETTINGS[key].default if key in SETTINGS else None


def all_settings(db) -> Dict[str, dict]:
    return {key: {'value': read_setting(db, key), 'default': item.default, 'low': item.low,
                  'high': item.high, 'description': item.description}
            for key, item in SETTINGS.items()}


def validated(key: str, value) -> int:
    if key not in SETTINGS:
        raise Refused(NOT_FOUND, f'{key} is not a setting that can be changed')
    item = SETTINGS[key]
    if isinstance(value, bool) or not isinstance(value, int) or not item.low <= value <= item.high:
        raise Refused(INVALID, f'{key} is a whole number from {item.low} to {item.high}')
    return value


def write_setting(cursor, key: str, value, updated_by, now: str) -> None:
    cursor.execute('INSERT INTO system_settings (key, value_json, updated_by, updated_at) '
                   'VALUES (?, ?, ?, ?) ON CONFLICT (key) DO UPDATE SET '
                   'value_json = excluded.value_json, updated_by = excluded.updated_by, '
                   'updated_at = excluded.updated_at', (key, json.dumps(value), updated_by, now))
