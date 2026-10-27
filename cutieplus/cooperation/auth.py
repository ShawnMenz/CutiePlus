"""
Credentials of the application users: Argon2id password hashes, session tokens and one-time setup
tokens. The database keeps hashes only; a token is shown once (a session token to its window, a
setup token to whoever hands it to the user) and never written anywhere else.

Failed sign-ins slow down per login name and per origin (a remote desktop, or the host's console):
every window of the shared account connects from the same host, so "localhost" would be everybody.
"""
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from cutieplus.cooperation.errors import INVALID, RATE_LIMITED, Refused

MIN_PASSWORD = 10
MAX_PASSWORD = 256
# setup tokens are typed or pasted by people: no 0/O, 1/I/L
SETUP_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'
SETUP_GROUPS, SETUP_GROUP_LENGTH = 4, 5  # about 99 bits


def password_hasher(fast: bool = False) -> PasswordHasher:
    """Argon2id with argon2-cffi's defaults (RFC 9106's low-memory choice); fast=True only for
    the tests."""
    if fast:
        return PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    return PasswordHasher()


class Passwords:

    def __init__(self, hasher: Optional[PasswordHasher] = None):
        self.hasher = hasher or password_hasher()
        # verified against for unknown names, so they take as long as known ones
        self.dummy_hash = self.hasher.hash(secrets.token_hex(16))

    def check_policy(self, password, login_name: str) -> str:
        if not isinstance(password, str) or not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
            raise Refused(INVALID, f'a password has {MIN_PASSWORD} to {MAX_PASSWORD} characters')
        if password.strip().lower() == login_name.lower():
            raise Refused(INVALID, 'the password must not be the login name')
        return password

    def hash(self, password: str) -> str:
        return self.hasher.hash(password)

    def verify(self, password_hash: Optional[str], password) -> bool:
        if not isinstance(password, str):
            password = ''
        try:
            return self.hasher.verify(password_hash or self.dummy_hash, password) \
                and password_hash is not None
        except (VerificationError, InvalidHashError):
            return False


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def new_setup_token() -> str:
    groups = (''.join(secrets.choice(SETUP_ALPHABET) for _ in range(SETUP_GROUP_LENGTH))
              for _ in range(SETUP_GROUPS))
    return '-'.join(groups)


def normalize_setup_token(token) -> str:
    if not isinstance(token, str):
        return ''
    return ''.join(character for character in token.upper() if character.isalnum())


def token_hash(token: str) -> str:
    """Tokens are random and long, so a plain SHA-256 is enough to keep them from being reused
    from a copy of the database."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def same_hash(left: Optional[str], right: Optional[str]) -> bool:
    return left is not None and right is not None and hmac.compare_digest(left, right)


class LoginLimiter:
    """
    Per login name: 5 failures in 15 minutes are free, then each attempt waits twice as long as
    the last (30 s up to 15 min). Per origin: 30 failures in 10 minutes close it for 10 minutes.
    Kept in memory: a restart of the service forgets them.
    """
    FREE_FAILURES = 5
    WINDOW_S = 15 * 60
    FIRST_WAIT_S = 30.0
    MAX_WAIT_S = 15 * 60.0
    ORIGIN_FAILURES = 30
    ORIGIN_WINDOW_S = 10 * 60

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.failures: Dict[str, Deque[float]] = defaultdict(deque)

    def _recent(self, key: str, window: float) -> Deque[float]:
        times = self.failures[key]
        now = self.clock()
        while times and times[0] < now - window:
            times.popleft()
        return times

    def check(self, login_name: str, origin: str) -> None:
        with self.lock:
            now = self.clock()
            origin_times = self._recent(f'origin:{origin}', self.ORIGIN_WINDOW_S)
            if len(origin_times) >= self.ORIGIN_FAILURES:
                wait = origin_times[-1] + self.ORIGIN_WINDOW_S - now
                raise Refused(RATE_LIMITED, 'too many failed sign-ins from here; wait '
                              f'{int(wait) + 1} s', retry_after=int(wait) + 1)
            times = self._recent(f'login:{login_name}', self.WINDOW_S)
            excess = len(times) - self.FREE_FAILURES
            if excess >= 0:
                wait = min(self.FIRST_WAIT_S * 2**excess, self.MAX_WAIT_S)
                remaining = times[-1] + wait - now
                if remaining > 0:
                    raise Refused(RATE_LIMITED, f'too many failed sign-ins for this name; wait '
                                  f'{int(remaining) + 1} s', retry_after=int(remaining) + 1)

    def failed(self, keys: Iterable[str]) -> None:
        with self.lock:
            now = self.clock()
            for key in keys:
                self.failures[key].append(now)

    def failed_sign_in(self, login_name: str, origin: str) -> None:
        self.failed((f'login:{login_name}', f'origin:{origin}'))

    def succeeded(self, login_name: str) -> None:
        with self.lock:
            self.failures.pop(f'login:{login_name}', None)
