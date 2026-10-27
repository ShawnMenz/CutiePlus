"""
Backups of the cooperation database, through SQLite's online backup (Database.backup), into
<state_dir>/backups/cooperation-<UTC time>.sqlite3; the newest backup_keep are kept.
"""
import os
from os import path
from typing import List

from cutieplus.cooperation import audit

PREFIX, SUFFIX = 'cooperation-', '.sqlite3'


def backup_files(folder: str) -> List[str]:
    if not path.isdir(folder):
        return []
    return sorted(path.join(folder, name) for name in os.listdir(folder)
                  if name.startswith(PREFIX) and name.endswith(SUFFIX))


def backup_now(service) -> str:
    folder = service.config.backups_dir
    os.makedirs(folder, mode=0o700, exist_ok=True)
    stamp = service.now().replace(':', '').replace('-', '').replace('.', '')
    destination = service.db.backup(path.join(folder, f'{PREFIX}{stamp}{SUFFIX}'))
    keep = service.setting('backup_keep')
    for old in backup_files(folder)[:-keep]:
        os.remove(old)
    with service.db.transaction() as cursor:
        audit.record(cursor, service.now(), 'backup', audit.SUCCEEDED, None, audit.SYSTEM,
                     'database', path.basename(destination))
    return destination
