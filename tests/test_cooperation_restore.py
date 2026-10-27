"""
Backup and restore of the cooperation database (backups.py, admin_cli.restore_file, restore.py):
the backup is SQLite's online copy; a restored database ends every session, interrupts the jobs
that were running, revokes again any key revoked after the backup (the published policy no longer
had it), and publishes its own policy.
"""
import os

from cutieplus.cooperation.admin_cli import AdminError, restore_file
from cutieplus.cooperation.app import build_service
from cutieplus.cooperation.auth import Passwords, password_hasher
from cutieplus.cooperation.backups import backup_files, backup_now
from cutieplus.cooperation.db import Database
from cutieplus.cooperation.operations import Reconciler
from tests.cooperation_harness import AccessHarness, ed25519_key


def reopen(access: AccessHarness):
    """The service started again on the same database and system."""
    access.db.close()
    db = Database(access.config.db_path, access.clock)
    service = build_service(access.config, db, access.clock,
                            Passwords(password_hasher(fast=True)), access.units.get)
    service.reconciler = Reconciler(service, access.helper_client)
    access.service, access.db = service, db
    return service


def test_a_restored_database_is_reconciled_on_the_safe_side(tmp_path):
    access = AccessHarness(tmp_path)
    owner = access.owner()
    alice = access.add(owner, 'alice')
    bob = access.add(owner, 'bob')
    alices_key = access.call('add_key', owner, user_id=alice['id'], public_key=ed25519_key(1),
                             device_label='Laptop')['key']['id']
    access.call('add_key', owner, user_id=bob['id'], public_key=ed25519_key(2),
                device_label='Laptop')
    access.reconcile()
    # a running job at backup time
    with access.db.transaction() as cursor:
        session = cursor.execute('SELECT id FROM app_sessions LIMIT 1').fetchall()[0]['id']
        cursor.execute(
            "INSERT INTO gpu_jobs (id, owner_user_id, submitted_by_app_session_id, "
            "operation_type, workspace_id, input_reference, input_revision, "
            "annotation_or_anchor_revision, model_identifier, model_version, parameters_json, "
            "status, queue_sequence, idempotency_key, submitted_at) VALUES ('0b9c2a4e-8f1d-4c3b-"
            "9a7e-5d6f7a8b9c0d', ?, ?, 'propagate', 'w', 'x', 'y', 0, 'm', 'v', '{}', "
            "'running', 1, 'restore-test', ?)", (alice['id'], session, access.service.now()))
    backup = backup_now(access.service)
    assert backup_files(access.config.backups_dir) == [backup]

    # after the backup: alice's key is revoked (and published), bob gets a new key
    access.call('revoke_key', owner, key_id=alices_key)
    bobs_new = access.call('add_key', owner, user_id=bob['id'], public_key=ed25519_key(3),
                           device_label='Phone')['key']['id']
    access.reconcile()
    assert alices_key not in access.policy_text() and bobs_new in access.policy_text()

    access.db.close()
    users = restore_file(access.config, backup, access.service.now())
    assert [user['login_name'] for user in users] == ['alice', 'bob', 'owner']
    service = reopen(access)
    summary = service.restored
    assert summary['keys_revoked'] == [alices_key]  # revoked after the backup: stays revoked
    assert summary['jobs_interrupted'] == 1 and summary['sessions_ended'] >= 3
    assert access.code('whoami', owner) == 'session_revoked'
    access.reconcile()
    text = access.policy_text()
    assert alices_key not in text  # never re-admitted
    assert bobs_new not in text  # unknown to the restored database: added again by hand
    assert text.count('cutieplus:') == 1  # bob's first key
    assert service.restored is not None and reopen(access).restored is None  # once only


def test_a_damaged_backup_is_refused(tmp_path):
    access = AccessHarness(tmp_path)
    access.owner()
    backup = backup_now(access.service)
    with open(backup, 'r+b') as f:
        f.seek(100)  # the schema's own page, after the file header
        f.write(b'\xff' * 2000)
    access.db.close()
    before = open(access.config.db_path, 'rb').read()
    try:
        restore_file(access.config, backup, access.service.now())
        refused = False
    except (AdminError, Exception):
        refused = True
    assert refused
    assert open(access.config.db_path, 'rb').read() == before  # untouched


def test_backups_keep_the_newest_few(tmp_path):
    access = AccessHarness(tmp_path)
    access.owner()
    with access.db.transaction() as cursor:
        cursor.execute("INSERT INTO system_settings (key, value_json, updated_at) VALUES "
                       "('backup_keep', '2', ?)", (access.service.now(), ))
    made = []
    for _ in range(3):
        access.clock.advance(hours=1)
        made.append(backup_now(access.service))
    assert backup_files(access.config.backups_dir) == made[1:]
    assert os.stat(access.config.backups_dir).st_mode & 0o777 == 0o700
