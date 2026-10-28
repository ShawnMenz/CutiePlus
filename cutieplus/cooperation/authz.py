"""
Who may do what: the one policy table of the cooperation module. The service asks it for every
request, with the caller from its session (Actor) and the target as the database has it NOW: a
Supervisor's request on a user is checked against that user's current role, a key's revocation
against the role of the key's owner, a job's cancellation against the job's owner.

Roles:
    annotator   annotation, and their own jobs, login state and password
    supervisor  the same, plus everything about annotators (create, edit, suspend, reactivate,
                remove, credentials, keys, desktops, sessions, jobs); never another supervisor
                or an admin, never a role
    admin       everything about annotators and supervisors, including appointing and revoking
                supervisors and deleting them for good; every job; the settings; the whole audit
                log

The owner (system_settings.owner_user_id, set by the protected bootstrap) cannot be suspended,
removed, deleted, demoted or have their credentials reset through the API. Being an Admin grants no access
to the operating system. GUI buttons follow this table, but it is enforced here.
"""
from dataclasses import dataclass
from typing import Optional

from cutieplus.cooperation.errors import FORBIDDEN, OWNER_PROTECTED, Refused

ADMIN, SUPERVISOR, ANNOTATOR = 'admin', 'supervisor', 'annotator'
ROLES = (ADMIN, SUPERVISOR, ANNOTATOR)
MANAGERS = (ADMIN, SUPERVISOR)
# the roles an admin may give with create_user or set_role (one owner admin in this version)
ASSIGNABLE_ROLES = (ANNOTATOR, SUPERVISOR)

# actions on a user
VIEW = 'view'
EDIT_PROFILE = 'edit_profile'
SUSPEND = 'suspend'
REACTIVATE = 'reactivate'
REMOVE = 'remove'
RESET_CREDENTIAL = 'reset_credential'
MANAGE_KEYS = 'manage_keys'
MANAGE_DESKTOP = 'manage_desktop'
MANAGE_SESSIONS = 'manage_sessions'
SET_ROLE = 'set_role'
DELETE = 'delete'  # for good: the account and its records (users.delete_user)
USER_ACTIONS = (VIEW, EDIT_PROFILE, SUSPEND, REACTIVATE, REMOVE, RESET_CREDENTIAL, MANAGE_KEYS,
                MANAGE_DESKTOP, MANAGE_SESSIONS, SET_ROLE, DELETE)
# what everyone may do to themselves through these actions (their password: change_password)
SELF_ACTIONS = (VIEW, EDIT_PROFILE, MANAGE_SESSIONS)
# an admin also gives themselves a key and a desktop (a supervisor's come from an admin)
ADMIN_SELF_ACTIONS = (MANAGE_KEYS, MANAGE_DESKTOP)
# what nobody may do to the owner through the API, the owner included
OWNER_PROTECTED_ACTIONS = (SUSPEND, REMOVE, RESET_CREDENTIAL, SET_ROLE, DELETE)
# what nobody may do to themselves: lock themselves out, or change their own role
NOT_ON_SELF = (SUSPEND, REMOVE, SET_ROLE, RESET_CREDENTIAL, DELETE)
# what only an admin does, to the users they manage
ADMIN_ONLY = {SET_ROLE: 'only an admin changes roles', DELETE: 'only an admin deletes a user'}


@dataclass(frozen=True)
class Actor:
    user_id: str
    login_name: str
    display_name: str
    role: str
    session_id: str
    desktop_slot_id: Optional[str]
    is_owner: bool


def is_manager(actor: Actor) -> bool:
    return actor.role in MANAGERS


def require_manager(actor: Actor) -> None:
    if not is_manager(actor):
        raise Refused(FORBIDDEN, 'only a supervisor or an admin manages users')


def require_admin(actor: Actor) -> None:
    if actor.role != ADMIN:
        raise Refused(FORBIDDEN, 'only an admin may do that')


def manages_role(actor: Actor, role: str) -> bool:
    """Whether the actor manages users who currently have that role."""
    if actor.role == ADMIN:
        return role in ASSIGNABLE_ROLES
    if actor.role == SUPERVISOR:
        return role == ANNOTATOR
    return False


def check_user_action(actor: Actor, target: dict, action: str, owner_id: Optional[str]) -> None:
    """Raises Refused unless the actor may do the action to the target user (a users row, as the
    database has it now)."""
    if action not in USER_ACTIONS:
        raise ValueError(action)
    is_self = target['id'] == actor.user_id
    if target['id'] == owner_id and action in OWNER_PROTECTED_ACTIONS:
        raise Refused(OWNER_PROTECTED, 'the owner is changed only by the maintenance commands '
                      '(cutieplus-admin)')
    if is_self:
        if action in SELF_ACTIONS or (actor.role == ADMIN and action in ADMIN_SELF_ACTIONS):
            return
        if action in NOT_ON_SELF:
            raise Refused(FORBIDDEN, 'not on your own account')
    if action in ADMIN_ONLY and actor.role != ADMIN:
        raise Refused(FORBIDDEN, ADMIN_ONLY[action])
    if not manages_role(actor, target['role']):
        raise Refused(FORBIDDEN, f'a {actor.role} does not manage a {target["role"]}')


def check_job_action(actor: Actor, owner_id: str, owner_role: str) -> None:
    """Viewing a job's details or cancelling it: your own jobs; a supervisor also annotators'
    jobs; an admin every job. The scheduling order is the same for every role."""
    if owner_id == actor.user_id or actor.role == ADMIN:
        return
    if actor.role == SUPERVISOR and owner_role == ANNOTATOR:
        return
    raise Refused(FORBIDDEN, "not your job")


def can_see_job_owner(actor: Actor, owner_id: str, owner_role: str) -> bool:
    try:
        check_job_action(actor, owner_id, owner_role)
    except Refused:
        return False
    return True
