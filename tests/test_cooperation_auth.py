"""
Signing in to the cooperation service (cutieplus/cooperation/service.py, auth.py, admin_cli.py):
the owner made by the protected bootstrap, one-time setup tokens, passwords, sessions bound to
their origin, expiry, rate limits, and sessions ending when a user's permissions change.
"""
import pytest

from cutieplus.cooperation.admin_cli import (AdminError, bootstrap_owner, owner_reset,
                                             owner_transfer)
from cutieplus.cooperation.client import ClientError
from tests.cooperation_harness import PASSWORD, Harness


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def everything_stored(db) -> str:
    """Every value of every table, as one text."""
    tables = [row['name'] for row in db.query("SELECT name FROM sqlite_master WHERE type='table'")]
    return repr([db.query(f'SELECT * FROM "{table}"') for table in tables])


def test_the_owner_comes_from_the_bootstrap_and_sets_up_once(harness):
    assert harness.call('hello')['owner_configured'] is False
    setup = bootstrap_owner(harness.service, 'zexuan_admin', 'Zexuan')
    with pytest.raises(AdminError):
        bootstrap_owner(harness.service, 'other', 'Other')
    # nobody signs in before the setup, and the token sets this one user's password only
    assert harness.code('login', login_name='zexuan_admin', password=PASSWORD) == \
        'invalid_credentials'
    assert harness.code('setup_credential', login_name='someone', setup_token=setup,
                        new_password=PASSWORD) == 'invalid_credentials'
    session = harness.call('setup_credential', login_name='zexuan_admin',
                           setup_token=setup.lower().replace('-', ' '), new_password=PASSWORD)
    me = harness.call('whoami', session['token'])
    assert (me['user']['role'], me['user']['status'], me['is_owner']) == ('admin', 'active', True)
    assert me['permissions']['manage_supervisors']
    # consumed: a second use fails
    assert harness.code('setup_credential', login_name='zexuan_admin', setup_token=setup,
                        new_password='another password') == 'invalid_credentials'
    # only hashes are stored
    stored = everything_stored(harness.db)
    assert setup not in stored and session['token'] not in stored and PASSWORD not in stored


def test_a_setup_token_expires(harness):
    owner = harness.owner()
    created = harness.call('create_user', owner, login_name='alice', display_name='Alice')
    harness.clock.advance(hours=harness.service.setting('setup_token_hours'), seconds=1)
    assert harness.code('setup_credential', login_name='alice',
                        setup_token=created['setup_token'],
                        new_password=PASSWORD) == 'invalid_credentials'
    # a new token from a reset works (the owner's session went idle meanwhile)
    owner = harness.login('owner')
    reset = harness.call('reset_credential', owner, user_id=created['user']['id'])
    harness.call('setup_credential', login_name='alice', setup_token=reset['setup_token'],
                 new_password=PASSWORD)


def test_wrong_passwords_say_nothing_and_slow_down(harness):
    harness.owner()
    wrong = harness.raw('login', login_name='owner', password='wrong password')
    unknown = harness.raw('login', login_name='nobody', password='wrong password')
    assert wrong['error'] == unknown['error']  # an unknown name reads the same
    for _ in range(4):
        harness.raw('login', login_name='owner', password='wrong password')
    assert harness.code('login', login_name='owner', password='wrong password') == 'rate_limited'
    # even the right password waits now
    assert harness.code('login', login_name='owner', password=PASSWORD) == 'rate_limited'
    failed = harness.db.query("SELECT * FROM audit_events WHERE action = 'login' AND "
                              "result = 'failed'")
    assert len(failed) == 6 and all(row['actor_user_id'] is None for row in failed)


def test_a_password_needs_some_length(harness):
    setup = bootstrap_owner(harness.service, 'owner', 'Owner')
    for password in ('short', 'owner', 'x' * 300, None):
        assert harness.code('setup_credential', login_name='owner', setup_token=setup,
                            new_password=password) == 'invalid'


def test_sessions_expire_end_and_stay_on_their_desktop(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    harness.add_slot(alice['id'], 2)
    desktop = harness.peer(display=2)
    token = harness.login('alice', desktop)
    assert harness.code('whoami', token, desktop) == 'ok'
    # the token is no good from elsewhere, not even the console
    assert harness.code('whoami', token) == 'session_invalid'
    # idle for too long
    harness.clock.advance(minutes=harness.service.setting('session_idle_minutes') + 1)
    assert harness.code('whoami', token, desktop) == 'session_expired'
    # used regularly, it still ends after session_max_hours
    token = harness.login('alice', desktop)
    for _ in range(harness.service.setting('session_max_hours') // 6 - 1):
        harness.clock.advance(hours=6)
        assert harness.code('whoami', token, desktop) == 'ok'
    harness.clock.advance(hours=6)  # session_max_hours after the sign-in
    assert harness.code('whoami', token, desktop) == 'session_expired'
    token = harness.login('alice', desktop)
    harness.call('logout', token, desktop)
    assert harness.code('whoami', token, desktop) == 'session_revoked'
    assert harness.code('whoami', 'made-up token') == 'session_invalid'
    assert harness.code('whoami') == 'session_invalid'


def test_a_desktop_is_signed_in_by_its_own_user_only(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    harness.add(owner, 'bob')
    harness.add_slot(alice['id'], 2)
    assert harness.code('login', peer=harness.peer(display=2), login_name='bob',
                        password=PASSWORD) == 'wrong_desktop'
    assert harness.code('login', peer=harness.peer(display=7), login_name='alice',
                        password=PASSWORD) == 'wrong_desktop'  # nobody's desktop
    assert harness.code('login', peer=harness.peer(display=2), login_name='alice',
                        password=PASSWORD) == 'ok'
    assert harness.code('login', login_name='bob', password=PASSWORD) == 'ok'  # the console


def test_changing_a_password_ends_the_other_sessions(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    other = harness.login('alice')
    harness.call('change_password', alice['token'], current_password=PASSWORD,
                 new_password='a new long password')
    assert harness.code('whoami', alice['token']) == 'ok'
    assert harness.code('whoami', other) == 'session_revoked'
    assert harness.code('login', login_name='alice', password=PASSWORD) == 'invalid_credentials'
    assert harness.code('change_password', alice['token'], current_password='wrong one',
                        new_password='x' * 12) == 'invalid_credentials'


def test_a_permission_change_needs_a_new_sign_in(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    assert harness.code('list_users', alice['token']) == 'forbidden'
    harness.call('set_role', owner, user_id=alice['id'], role='supervisor')
    # the old session predates the change: refused at once, not at its next expiry
    assert harness.code('list_users', alice['token']) == 'session_revoked'
    promoted = harness.login('alice')
    assert harness.call('whoami', promoted)['permissions']['manage_annotators']
    assert harness.code('list_users', promoted) == 'ok'
    # a change the session was not revoked for (e.g. by maintenance) is caught by its version
    with harness.db.transaction() as cursor:
        cursor.execute('UPDATE users SET authz_version = authz_version + 1 WHERE id = ?',
                       (alice['id'], ))
    assert harness.code('list_users', promoted) == 'session_stale'


def test_role_changes_need_a_recent_password(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    harness.clock.advance(minutes=harness.service.setting('reauth_minutes') + 1)
    assert harness.code('set_role', owner, user_id=alice['id'], role='supervisor') == \
        'reauth_required'
    assert harness.code('reauthenticate', owner, password='wrong password') == \
        'invalid_credentials'
    harness.call('reauthenticate', owner, password=PASSWORD)
    assert harness.code('set_role', owner, user_id=alice['id'], role='supervisor') == 'ok'


def test_the_maintenance_commands_recover_and_transfer_the_owner(harness):
    owner = harness.owner()
    alice = harness.add(owner, 'alice')
    setup = owner_reset(harness.service)
    assert harness.code('whoami', owner) == 'session_revoked'
    assert harness.code('login', login_name='owner', password=PASSWORD) == 'invalid_credentials'
    owner = harness.call('setup_credential', login_name='owner', setup_token=setup,
                         new_password='the new owner password')['token']
    owner_transfer(harness.service, 'alice')
    alice_token = harness.login('alice')
    me = harness.call('whoami', alice_token)
    assert (me['user']['role'], me['is_owner']) == ('admin', True)
    former = harness.call('login', login_name='owner',
                          password='the new owner password')['token']
    assert harness.call('whoami', former)['user']['role'] == 'supervisor'
    with pytest.raises(ClientError) as refused:
        harness.call('set_role', alice_token, user_id=alice['id'], role='annotator')
    assert refused.value.code == 'owner_protected'
