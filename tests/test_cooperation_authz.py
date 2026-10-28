"""
Who may do what to whom (cutieplus/cooperation/authz.py, users.py), by direct requests to the
service: a window that hides a button proves nothing, so every case here is a request a modified
window could send.

The cast: the owner (admin), two supervisors, two annotators.
"""
import pytest

from tests.cooperation_harness import PASSWORD, Harness

OK, F, OP = 'ok', 'forbidden', 'owner_protected'
ACTIONS = ('view', 'edit', 'suspend', 'reactivate', 'remove', 'reset', 'set_role', 'sessions',
           'delete')

# (actor, target) -> the outcome of each action, in ACTIONS order
EXPECTED = {
    ('owner', 'owner'): (OK, OK, OP, F, OP, OP, OP, OK, OP),
    ('owner', 'sup1'): (OK, ) * 9,
    ('owner', 'ann1'): (OK, ) * 9,
    ('sup1', 'owner'): (F, F, OP, F, OP, OP, F, F, OP),
    ('sup1', 'sup1'): (OK, OK, F, F, F, F, F, OK, F),
    ('sup1', 'sup2'): (F, ) * 9,
    ('sup1', 'ann1'): (OK, OK, OK, OK, OK, OK, F, OK, F),
    ('ann1', 'owner'): (F, F, OP, F, OP, OP, F, F, OP),
    ('ann1', 'sup1'): (F, ) * 9,
    ('ann1', 'ann1'): (OK, OK, F, F, F, F, F, OK, F),
    ('ann1', 'ann2'): (F, ) * 9,
}
CASES = [(actor, target, action, outcomes[index]) for (actor, target), outcomes in
         EXPECTED.items() for index, action in enumerate(ACTIONS)]


class Cast:

    def __init__(self, tmp_path):
        self.harness = Harness(tmp_path)
        owner = self.harness.owner()
        self.tokens = {'owner': owner}
        self.ids = {'owner': self.harness.service.owner_id()}
        for name, role in (('sup1', 'supervisor'), ('sup2', 'supervisor'), ('ann1', None),
                           ('ann2', None)):
            user = self.harness.add(owner, name, role)
            self.tokens[name], self.ids[name] = user['token'], user['id']

    def request(self, actor: str, target: str, action: str) -> str:
        harness, token, user_id = self.harness, self.tokens[actor], self.ids[target]
        row = harness.service.user_row(user_id)
        if action == 'view':
            return harness.code('get_user', token, user_id=user_id)
        if action == 'edit':
            return harness.code('update_user', token, user_id=user_id,
                                row_version=row['row_version'], display_name='Renamed')
        if action == 'reactivate':
            if target not in (actor, 'owner'):  # suspended first, by the owner
                harness.call('suspend_user', self.tokens['owner'], user_id=user_id)
            return harness.code('reactivate_user', token, user_id=user_id)
        if action == 'set_role':
            role = 'annotator' if row['role'] != 'annotator' else 'supervisor'
            return harness.code('set_role', token, user_id=user_id, role=role)
        if action == 'delete':
            return harness.code('delete_user', token, user_id=user_id, password=PASSWORD)
        operation = {'suspend': 'suspend_user', 'remove': 'remove_user',
                     'reset': 'reset_credential', 'sessions': 'revoke_sessions'}[action]
        return harness.code(operation, token, user_id=user_id)


@pytest.mark.parametrize('actor, target, action, expected', CASES)
def test_each_role_acts_on_whom_it_manages(tmp_path, actor, target, action, expected):
    cast = Cast(tmp_path)
    before = cast.harness.service.user_row(cast.ids[target])
    assert cast.request(actor, target, action) == expected
    after = cast.harness.service.user_row(cast.ids[target])
    if expected != OK:
        # a refusal changes nothing (reactivate's target was suspended by the owner first)
        changed = {key for key in before if before[key] != after[key]}
        assert changed <= ({'status', 'suspended_at', 'authz_version', 'row_version',
                            'updated_at'} if action == 'reactivate' else set())
        refused = cast.harness.db.query("SELECT * FROM audit_events WHERE result = 'refused'")
        assert [row['actor_user_id'] for row in refused] == [cast.ids[actor]]


def test_a_supervisor_creates_annotators_and_nothing_else(tmp_path):
    cast = Cast(tmp_path)
    harness, supervisor = cast.harness, cast.tokens['sup1']
    for role in ('annotator', 'supervisor', 'admin'):
        assert harness.code('create_user', supervisor, login_name='eve', display_name='Eve',
                            role=role) == 'unexpected_field'
    # fields the operation does not take are refused, whatever they are
    for extra in ({'status': 'active'}, {'created_by': cast.ids['owner']},
                  {'actor_id': cast.ids['owner']}, {'is_owner': True}):
        assert harness.code('create_user', supervisor, login_name='eve', display_name='Eve',
                            **extra) == 'unexpected_field'
    created = harness.call('create_user', supervisor, login_name='eve', display_name='Eve')
    assert created['user']['role'] == 'annotator' and created['user']['status'] == 'pending'
    assert harness.code('create_user', cast.tokens['ann1'], login_name='mallory',
                        display_name='M') == 'forbidden'
    # an admin creates a supervisor, never another admin
    assert harness.call('create_user', cast.tokens['owner'], login_name='sam',
                        display_name='Sam', role='supervisor')['user']['role'] == 'supervisor'
    assert harness.code('create_user', cast.tokens['owner'], login_name='adam',
                        display_name='Adam', role='admin') == 'invalid'
    assert harness.code('create_user', cast.tokens['owner'], login_name='root',
                        display_name='Root') == 'invalid'  # reserved
    assert harness.code('create_user', cast.tokens['owner'], login_name='eve',
                        display_name='Eve again') == 'conflict'


def test_the_lists_hold_whom_the_caller_manages(tmp_path):
    cast = Cast(tmp_path)
    harness = cast.harness
    names = lambda token, **args: sorted(user['login_name'] for user in harness.call(
        'list_users', token, **args))
    assert names(cast.tokens['owner']) == ['ann1', 'ann2', 'owner', 'sup1', 'sup2']
    assert names(cast.tokens['sup1']) == ['ann1', 'ann2']
    assert names(cast.tokens['owner'], query='sup') == ['sup1', 'sup2']
    assert names(cast.tokens['owner'], query='%') == []  # a pattern character is just text
    assert harness.code('list_users', cast.tokens['ann1']) == 'forbidden'
    harness.call('remove_user', cast.tokens['sup1'], user_id=cast.ids['ann2'])
    assert names(cast.tokens['sup1']) == ['ann1']
    assert names(cast.tokens['sup1'], include_archived=True) == ['ann1', 'ann2']
    # the audit: an admin sees everything, a supervisor annotators' events and their own
    supervisor_view = harness.call('list_audit', cast.tokens['sup1'])
    assert supervisor_view and all(
        event['actor_user_id'] == cast.ids['sup1'] or event['target_id'] in
        (cast.ids['ann1'], cast.ids['ann2']) for event in supervisor_view)
    everything = harness.call('list_audit', cast.tokens['owner'], limit=1000)
    assert len(everything) > len(supervisor_view)
    assert harness.code('list_audit', cast.tokens['ann1']) == 'forbidden'
    assert harness.code('get_settings', cast.tokens['sup1']) == 'forbidden'


def test_a_target_whose_role_changed_is_checked_as_it_is_now(tmp_path):
    cast = Cast(tmp_path)
    harness = cast.harness
    supervisor = cast.tokens['sup1']
    assert harness.code('get_user', supervisor, user_id=cast.ids['ann1']) == 'ok'
    harness.call('set_role', cast.tokens['owner'], user_id=cast.ids['ann1'], role='supervisor')
    # ann1 is a supervisor now: sup1 no longer manages them
    assert harness.code('suspend_user', supervisor, user_id=cast.ids['ann1']) == 'forbidden'
    # and a demoted supervisor loses management at once
    harness.call('set_role', cast.tokens['owner'], user_id=cast.ids['sup2'], role='annotator')
    assert harness.code('list_users', cast.tokens['sup2']) == 'session_revoked'
    demoted = harness.login('sup2')
    assert harness.code('list_users', demoted) == 'forbidden'
    assert harness.code('get_user', supervisor, user_id=cast.ids['sup2']) == 'ok'


def test_suspended_and_removed_users_cannot_sign_in(tmp_path):
    cast = Cast(tmp_path)
    harness = cast.harness
    harness.call('suspend_user', cast.tokens['sup1'], user_id=cast.ids['ann1'])
    assert harness.code('whoami', cast.tokens['ann1']) == 'session_revoked'
    assert harness.code('login', login_name='ann1', password='correct horse battery') == \
        'invalid_credentials'
    harness.call('reactivate_user', cast.tokens['sup1'], user_id=cast.ids['ann1'])
    token = harness.login('ann1')
    harness.call('remove_user', cast.tokens['sup1'], user_id=cast.ids['ann1'])
    assert harness.code('whoami', token) == 'session_revoked'
    removed = harness.service.user_row(cast.ids['ann1'])
    assert removed['status'] == 'archived' and removed['login_name'] == 'ann1'  # kept
    assert harness.code('reactivate_user', cast.tokens['sup1'], user_id=cast.ids['ann1']) == \
        'invalid'


def test_profile_edits_detect_a_change_meanwhile(tmp_path):
    cast = Cast(tmp_path)
    harness = cast.harness
    seen = harness.call('get_user', cast.tokens['sup1'], user_id=cast.ids['ann1'])
    harness.call('update_user', cast.tokens['owner'], user_id=cast.ids['ann1'],
                 row_version=seen['row_version'], contact='ann1@example.org')
    assert harness.code('update_user', cast.tokens['sup1'], user_id=cast.ids['ann1'],
                        row_version=seen['row_version'], display_name='Stale') == 'conflict'


def test_only_an_admin_changes_settings_within_their_bounds(tmp_path):
    cast = Cast(tmp_path)
    harness = cast.harness
    owner = cast.tokens['owner']
    harness.call('update_setting', owner, key='pending_jobs_per_user', value=5)
    assert harness.service.setting('pending_jobs_per_user') == 5
    assert harness.code('update_setting', owner, key='pending_jobs_per_user', value=0) == \
        'invalid'
    assert harness.code('update_setting', owner, key='owner_user_id',
                        value=cast.ids['sup1']) == 'forbidden'
    assert harness.code('update_setting', owner, key='gpu_jobs_at_once', value=2) == \
        'not_found'
    assert harness.code('update_setting', cast.tokens['sup1'], key='backup_keep',
                        value=3) == 'forbidden'
