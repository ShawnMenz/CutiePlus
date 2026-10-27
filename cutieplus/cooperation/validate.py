"""Checks of the values a request carries; each raises Refused(INVALID) with a message for the
window."""
import re
import unicodedata
from typing import Optional

from cutieplus.cooperation.errors import INVALID, Refused

LOGIN_NAME = re.compile(r'[a-z][a-z0-9._-]{1,31}')
# names that would read as the system, the account or a role
RESERVED_LOGIN_NAMES = frozenset(('root', 'admin', 'administrator', 'supervisor', 'annotator',
                                  'system', 'cutieplus', 'zexuan', 'nobody', 'owner'))


def login_name(value) -> str:
    """Lower case, starting with a letter, 2-32 of a-z 0-9 . _ -"""
    if not isinstance(value, str):
        raise Refused(INVALID, 'the login name must be text')
    name = unicodedata.normalize('NFKC', value).strip().lower()
    if not LOGIN_NAME.fullmatch(name):
        raise Refused(INVALID, 'a login name has 2-32 characters: a letter, then letters, '
                      'digits, ".", "_" or "-"')
    return name


def new_login_name(value) -> str:
    name = login_name(value)
    if name in RESERVED_LOGIN_NAMES:
        raise Refused(INVALID, f'{name} is reserved')
    return name


def text(value, field: str, max_length: int, required: bool = True) -> Optional[str]:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise Refused(INVALID, f'{field} must be text')
    value = unicodedata.normalize('NFC', value).strip()
    if required and not value:
        raise Refused(INVALID, f'{field} is empty')
    if len(value) > max_length or any(unicodedata.category(c) in ('Cc', 'Cf') for c in value):
        raise Refused(INVALID, f'{field} has more than {max_length} characters or control '
                      'characters')
    return value or None


def identifier(value, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f-]{36}', value):
        raise Refused(INVALID, f'{field} is not an id')
    return value


def integer(value, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise Refused(INVALID, f'{field} must be a whole number from {low} to {high}')
    return value
