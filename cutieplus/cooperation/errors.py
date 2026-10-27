"""
Why the service refuses a request. The code travels to the window (client.ClientError.code); the
message is meant for its log. Refusals never say more than the caller may know: a wrong password
and an unknown login name read the same.
"""

BAD_REQUEST = 'bad_request'
UNEXPECTED_FIELD = 'unexpected_field'
INVALID_CREDENTIALS = 'invalid_credentials'
RATE_LIMITED = 'rate_limited'
WRONG_DESKTOP = 'wrong_desktop'
SESSION_INVALID = 'session_invalid'
SESSION_EXPIRED = 'session_expired'
SESSION_REVOKED = 'session_revoked'
SESSION_STALE = 'session_stale'  # the user's permissions changed: sign in again
SETUP_REQUIRED = 'setup_required'
FORBIDDEN = 'forbidden'
OWNER_PROTECTED = 'owner_protected'
REAUTH_REQUIRED = 'reauth_required'
NOT_FOUND = 'not_found'
CONFLICT = 'conflict'  # changed meanwhile (row_version), or already exists
INVALID = 'invalid'
LIMIT = 'limit'
UNAVAILABLE = 'unavailable'
INTERNAL = 'internal'

# a session that cannot be used any more: the window asks its user to sign in again
SIGN_IN_AGAIN = frozenset((SESSION_INVALID, SESSION_EXPIRED, SESSION_REVOKED, SESSION_STALE,
                           SETUP_REQUIRED))


class Refused(Exception):

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
