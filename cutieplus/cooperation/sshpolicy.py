"""
OpenSSH public keys and the managed SSH policy, on the standard library only: the service uses it
to check a key it is given, the root helper to write the policy file. The helper renders every
line itself from checked parts, so nothing a caller wrote (a key comment, an option) reaches the
file.

A line of the policy (the annotation sshd's AuthorizedKeysFile):

    restrict,port-forwarding,permitopen="localhost:5902",permitopen="127.0.0.1:5902",
    command="/usr/sbin/nologin"[,expiry-time="202612312359Z"] ssh-ed25519 AAAA... cutieplus:<key id>

restrict turns off everything (pty, agent, X11 and port forwarding, ~/.ssh/rc); port-forwarding
gives back local forwarding, limited by permitopen to the user's own VNC port; command= stops a
shell or SFTP even if the sshd configuration allowed them; expiry-time makes sshd refuse the key
after its expiry by itself, whether or not the policy was published again since.

Collaborators reach the annotation sshd, which listens on this computer only, through port 22:
a network's firewall may pass SSH on port 22 only, as campus networks often do. There they sign
in as a jump account of its own (cutieplus-jump: no shell, no password), never as the desktop
account, with the jump file's keys (render_jump_policy: the same options, permitopen the
annotation sshd's port). Port 22's sshd itself holds that account to a channel to the annotation
sshd and nothing else (render_jump_dropin: no Unix-socket or remote forwarding, no tty, no
command), whatever a key's options would allow: port-forwarding gives back every kind of
forwarding, and permitopen limits TCP only. ssh -J does just that.
"""
import base64
import binascii
import hashlib
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

NOLOGIN = '/usr/sbin/nologin'
MAX_KEY_TEXT = 16 * 1024
MIN_RSA_BITS = 3072
ECDSA_CURVES = {'ecdsa-sha2-nistp256': 'nistp256', 'ecdsa-sha2-nistp384': 'nistp384',
                'ecdsa-sha2-nistp521': 'nistp521'}
KEY_TYPES = ('ssh-ed25519', 'sk-ssh-ed25519@openssh.com', 'sk-ecdsa-sha2-nistp256@openssh.com',
             'ssh-rsa', *ECDSA_CURVES)
KEY_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


class KeyError_(ValueError):
    """A key or policy entry that is refused; the message says why."""


@dataclass(frozen=True)
class PublicKey:
    key_type: str
    blob: bytes
    bits: Optional[int] = None

    @property
    def base64(self) -> str:
        return base64.b64encode(self.blob).decode('ascii')

    @property
    def canonical(self) -> str:
        """The key without a comment: what is stored and published."""
        return f'{self.key_type} {self.base64}'

    @property
    def fingerprint(self) -> str:
        """As OpenSSH shows it (ssh-keygen -l, sshd's log): SHA256:<base64 without padding>."""
        digest = base64.b64encode(hashlib.sha256(self.blob).digest()).decode('ascii')
        return 'SHA256:' + digest.rstrip('=')


def _strings(blob: bytes) -> List[bytes]:
    """The length-prefixed strings of a key blob (RFC 4251); refuses trailing bytes."""
    parts, offset = [], 0
    while offset < len(blob):
        if offset + 4 > len(blob):
            raise KeyError_('the key data is cut short')
        (length, ) = struct.unpack('>I', blob[offset:offset + 4])
        offset += 4
        if offset + length > len(blob):
            raise KeyError_('the key data is cut short')
        parts.append(blob[offset:offset + length])
        offset += length
    return parts


def parse_public_key(text) -> PublicKey:
    """One OpenSSH public key line ('<type> <base64> [comment]'). Refuses options in front of it,
    private keys, several lines, unknown types, malformed data and RSA keys under 3072 bits. The
    comment is dropped: it names nothing anyone checked."""
    if not isinstance(text, str) or len(text) > MAX_KEY_TEXT:
        raise KeyError_('a public key is one line of text')
    if 'PRIVATE KEY' in text:
        raise KeyError_('that is a private key: never send it; send the .pub file')
    text = text.strip()
    if not text or any(character in text for character in '\r\n\0'):
        raise KeyError_('a public key is one line of text')
    fields = text.split()
    if fields[0] not in KEY_TYPES:
        if any(field in KEY_TYPES for field in fields[1:]):
            raise KeyError_('options in front of the key are not accepted: send the key only')
        raise KeyError_(f'not an accepted key type (one of {", ".join(KEY_TYPES)})')
    if len(fields) < 2:
        raise KeyError_('the key data is missing')
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (binascii.Error, ValueError):
        raise KeyError_('the key data is not base64') from None
    parts = _strings(blob)
    if not parts or parts[0].decode('ascii', 'replace') != fields[0]:
        raise KeyError_('the key data does not match its type')
    key_type, bits = fields[0], None
    if key_type == 'ssh-ed25519':
        if len(parts) != 2 or len(parts[1]) != 32:
            raise KeyError_('a malformed ed25519 key')
    elif key_type == 'sk-ssh-ed25519@openssh.com':
        if len(parts) != 3 or len(parts[1]) != 32:
            raise KeyError_('a malformed security-key ed25519 key')
    elif key_type in ECDSA_CURVES or key_type == 'sk-ecdsa-sha2-nistp256@openssh.com':
        curve = ECDSA_CURVES.get(key_type, 'nistp256')
        expected = 3 if key_type in ECDSA_CURVES else 4
        if len(parts) != expected or parts[1] != curve.encode() or not parts[2].startswith(b'\x04'):
            raise KeyError_('a malformed ECDSA key')
    else:  # ssh-rsa: e, n
        if len(parts) != 3:
            raise KeyError_('a malformed RSA key')
        bits = int.from_bytes(parts[2], 'big').bit_length()
        if bits < MIN_RSA_BITS:
            raise KeyError_(f'an RSA key of {bits} bits is too weak: {MIN_RSA_BITS} at least, '
                            'or better an ed25519 key (ssh-keygen -t ed25519)')
    return PublicKey(key_type, blob, bits)


@dataclass(frozen=True)
class PolicyEntry:
    key_id: str
    public_key: str  # canonical '<type> <base64>'
    port: int  # the user's VNC port
    expires_at: Optional[str] = None  # UTC, clock.iso format

    def as_dict(self) -> dict:
        return {'key_id': self.key_id, 'public_key': self.public_key, 'port': self.port,
                'expires_at': self.expires_at}

    @classmethod
    def from_dict(cls, value: dict) -> 'PolicyEntry':
        if not isinstance(value, dict) or set(value) != {'key_id', 'public_key', 'port',
                                                         'expires_at'}:
            raise KeyError_('a policy entry has key_id, public_key, port and expires_at')
        return cls(value['key_id'], value['public_key'], value['port'], value['expires_at'])


def _expiry(expires_at: Optional[str]) -> Optional[str]:
    if expires_at is None:
        return None
    if not isinstance(expires_at, str):
        raise KeyError_('expires_at must be text')
    try:
        moment = datetime.strptime(expires_at, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        raise KeyError_(f'expires_at is not a UTC time: {expires_at!r}') from None
    return moment.replace(tzinfo=timezone.utc).strftime('%Y%m%d%H%MZ')


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _line(entry: PolicyEntry, ports: Tuple[int, int], target: int) -> str:
    """One entry's line, every part checked here: forwarding to the target port only."""
    if not isinstance(entry.key_id, str) or not KEY_ID.fullmatch(entry.key_id):
        raise KeyError_('a key id is a UUID')
    if not _whole(entry.port) or not ports[0] <= entry.port <= ports[1]:
        raise KeyError_(f'port {entry.port!r} is outside the managed VNC ports')
    if not _whole(target) or not 0 < target < 65536:
        raise KeyError_(f'port {target!r} is not a port')
    key = parse_public_key(entry.public_key)
    if key.canonical != entry.public_key:
        raise KeyError_('the key is not in its canonical form')
    options = ['restrict', 'port-forwarding', f'permitopen="localhost:{target}"',
               f'permitopen="127.0.0.1:{target}"', f'command="{NOLOGIN}"']
    expiry = _expiry(entry.expires_at)
    if expiry is not None:
        options.append(f'expiry-time="{expiry}"')
    return f'{",".join(options)} {key.canonical} cutieplus:{entry.key_id}'


def render_line(entry: PolicyEntry, ports: Tuple[int, int]) -> str:
    """The entry's line in the annotation sshd's policy: to its user's VNC port only."""
    return _line(entry, ports, entry.port)


def render_jump_line(entry: PolicyEntry, ports: Tuple[int, int], annotation_port: int) -> str:
    """The entry's line in port 22's jump file: to the annotation sshd only, whose own line
    (render_line) decides the rest."""
    return _line(entry, ports, annotation_port)


def _render(revision: int, entries: Iterable[PolicyEntry], what: str, line) -> str:
    if not _whole(revision) or revision < 0:
        raise KeyError_('the revision is a whole number')
    lines = [f'# Written by cutieplus-helper from the cooperation database, revision {revision}:',
             f'# {what}. Do not edit: the next publication replaces it.']
    seen = set()
    for entry in sorted(entries, key=lambda item: item.key_id):
        if entry.key_id in seen:
            raise KeyError_(f'key {entry.key_id} appears twice')
        seen.add(entry.key_id)
        lines.append(line(entry))
    return '\n'.join(lines) + '\n'


def render_policy(revision: int, entries: Iterable[PolicyEntry], ports: Tuple[int, int]) -> str:
    return _render(revision, entries, "the keys, each to its user's desktop",
                   lambda entry: render_line(entry, ports))


def render_jump_policy(revision: int, entries: Iterable[PolicyEntry], ports: Tuple[int, int],
                       annotation_port: int) -> str:
    return _render(revision, entries, 'the keys that may hop from port 22 to the annotation sshd',
                   lambda entry: render_jump_line(entry, ports, annotation_port))


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def policies_checksum(policy: str, jump: str) -> str:
    """Of the two files a publication writes, together."""
    return checksum(policy + '\0' + jump)


def render_jump_dropin(jump_user: str, jump_keys: str, annotation_port: int) -> str:
    """Port 22's sshd_config.d drop-in for the jump account: the jump file's keys only, and a
    local TCP channel to the annotation sshd only, held by sshd whatever a key allows. Nobody
    else's configuration changes."""
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', jump_user):
        raise KeyError_(f'{jump_user!r} is not an account name')
    if not _whole(annotation_port) or not 0 < annotation_port < 65536:
        raise KeyError_(f'port {annotation_port!r} is not a port')
    return '\n'.join([
        f'# CutiePlus (deploy/install.sh): collaborators sign in here as {jump_user}, with the keys',
        f'# of {jump_keys} only, to hop to the annotation sshd on this computer (ssh -J) and do',
        '# nothing else. Nobody else\'s configuration changes.',
        f'Match User {jump_user}',
        f'    AuthorizedKeysFile {jump_keys}',
        '    AuthenticationMethods publickey',
        '    PasswordAuthentication no',
        '    KbdInteractiveAuthentication no',
        '    AllowTcpForwarding local',
        f'    PermitOpen localhost:{annotation_port} 127.0.0.1:{annotation_port}',
        '    PermitListen none',
        '    AllowStreamLocalForwarding no',
        '    AllowAgentForwarding no',
        '    X11Forwarding no',
        '    PermitTunnel no',
        '    PermitTTY no',
        '    PermitUserRC no',
        '    GatewayPorts no',
        f'    ForceCommand {NOLOGIN}',
    ]) + '\n'


def render_sshd_config(port: int, authorized_keys: str, ports: Tuple[int, int],
                       pid_file: str = '/run/cutieplus-sshd.pid', host_keys: Iterable[str] = (
                           '/etc/ssh/ssh_host_ed25519_key', '/etc/ssh/ssh_host_ecdsa_key',
                           '/etc/ssh/ssh_host_rsa_key'), *, user: str,
                       listen: Iterable[str] = ('127.0.0.1', '::1')) -> str:
    """The annotation entrypoint's own sshd configuration: on this computer only (reached from
    port 22, the jump file), keys of the managed policy only, local forwarding to the managed VNC
    ports only, no shell, no SFTP, and no PAM (no session is opened for the account: a
    forwarding needs none)."""
    permit = ' '.join(f'localhost:{number} 127.0.0.1:{number}'
                      for number in range(ports[0], ports[1] + 1))
    lines = [
        '# The annotation SSH entrypoint of CutiePlus (deploy/install.sh). Collaborators reach',
        '# their own remote desktop through it, hopping from port 22; nothing else is open here.',
        f'Port {port}',
        *(f'ListenAddress {address}' for address in listen),
        *(f'HostKey {key}' for key in host_keys),
        f'PidFile {pid_file}',
        f'AllowUsers {user}',
        f'AuthorizedKeysFile {authorized_keys}',
        'AuthorizedKeysCommand none',
        'AuthorizedPrincipalsFile none',
        'PubkeyAuthentication yes',
        'AuthenticationMethods publickey',
        'PasswordAuthentication no',
        'KbdInteractiveAuthentication no',
        'HostbasedAuthentication no',
        'IgnoreRhosts yes',
        'PermitRootLogin no',
        'PermitEmptyPasswords no',
        'StrictModes yes',
        'MaxAuthTries 3',
        'LoginGraceTime 30',
        'ClientAliveInterval 30',
        'ClientAliveCountMax 4',
        'AllowTcpForwarding local',
        f'PermitOpen {permit}',
        'PermitListen none',
        'GatewayPorts no',
        'AllowStreamLocalForwarding no',
        'AllowAgentForwarding no',
        'X11Forwarding no',
        'PermitTunnel no',
        'PermitTTY no',
        'PermitUserRC no',
        'PermitUserEnvironment no',
        'DisableForwarding no',
        f'ForceCommand {NOLOGIN}',
        '# no Subsystem line: no SFTP',
        'PrintMotd no',
        'LogLevel VERBOSE',
        'UsePAM no',
    ]
    return '\n'.join(lines) + '\n'
