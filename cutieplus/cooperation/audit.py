"""
The audit log (audit_events), written by the service in the same transaction as the change it
records. The actor is the authenticated user of the session (authenticated_app), sshd's verified
journal entry (verified_sshd) or the service itself (system); nothing a caller writes names an
actor. Secrets never enter it: password and token hashes are dropped from recorded states.
"""
import json
from typing import Any, List, Optional

SECRET_FIELDS = frozenset(('password_hash', 'setup_token_hash', 'token_hash', 'password',
                           'setup_token', 'token', 'vnc_password'))
APP, SSHD, SYSTEM = 'authenticated_app', 'verified_sshd', 'system'
SUCCEEDED, FAILED, REFUSED = 'succeeded', 'failed', 'refused'


def _state(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key not in SECRET_FIELDS}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def record(cursor, now: str, action: str, result: str = SUCCEEDED, actor_id: Optional[str] = None,
           source: str = APP, target_type: Optional[str] = None, target_id: Optional[str] = None,
           request_id: Optional[str] = None, operation_id: Optional[str] = None,
           before: Any = None, after: Any = None, error_code: Optional[str] = None) -> None:
    cursor.execute(
        'INSERT INTO audit_events (actor_user_id, actor_source, action, target_type, target_id, '
        'request_id, operation_id, before_state_json, after_state_json, result, error_code, '
        'created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (actor_id, source, action, target_type, target_id, request_id, operation_id,
         _state(before), _state(after), result, error_code, now))


def events(db, where: str = '1', bindings=(), limit: int = 100) -> List[dict]:
    rows = db.query(
        'SELECT e.*, a.login_name AS actor_login FROM audit_events e LEFT JOIN users a '
        f'ON a.id = e.actor_user_id WHERE {where} ORDER BY e.id DESC LIMIT ?',
        tuple(bindings) + (int(limit), ))
    for row in rows:
        for key in ('before_state_json', 'after_state_json'):
            text = row.pop(key)
            row[key[:-5]] = json.loads(text) if text else None
    return rows
