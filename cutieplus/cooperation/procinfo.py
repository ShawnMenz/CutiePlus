"""
What the kernel says about a process: the peer of a Unix socket (SO_PEERCRED), the systemd unit
whose control group holds it, its start time, and the flock locks it holds. A process cannot
forge any of these; the service trusts them where a process's own words would prove nothing (every
desktop and the GPU worker run as the same account).

A PID alone identifies nothing over time: PIDs are reused. A ProcessIdentity is the PID plus the
process's start time (in clock ticks since boot) plus the boot's id, and is compared whole.
"""
import os
import socket
import struct
from dataclasses import asdict, dataclass
from typing import Optional, Set

PROC = '/proc'
_PEERCRED = struct.Struct('3i')  # struct ucred: pid, uid, gid


@dataclass(frozen=True)
class Peer:
    pid: int
    uid: int
    gid: int


def peer_of(sock: socket.socket) -> Peer:
    data = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
    return Peer(*_PEERCRED.unpack(data))


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> Optional['ProcessIdentity']:
        if not value:
            return None
        return cls(int(value['pid']), int(value['start_ticks']), str(value['boot_id']))


def boot_id(proc: str = PROC) -> str:
    with open(os.path.join(proc, 'sys', 'kernel', 'random', 'boot_id'), encoding='ascii') as f:
        return f.read().strip()


def process_identity(pid: int, proc: str = PROC) -> Optional[ProcessIdentity]:
    """None when there is no such process."""
    try:
        with open(os.path.join(proc, str(pid), 'stat'), encoding='utf-8',
                  errors='replace') as f:
            stat = f.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # the command name, in parentheses, may itself hold spaces and parentheses
    fields = stat[stat.rindex(')') + 2:].split()
    # fields[0] is field 3 of proc(5) (state); the start time is field 22
    return ProcessIdentity(pid, int(fields[19]), boot_id(proc))


def same_process_alive(identity: Optional[ProcessIdentity], proc: str = PROC) -> bool:
    """Whether that very process still runs (not another one that got its PID)."""
    if identity is None:
        return False
    return process_identity(identity.pid, proc) == identity


def system_unit_of(pid: int, proc: str = PROC) -> Optional[str]:
    """The system service whose control group holds the process, e.g.
    'cutieplus-desktop@2.service'; None for anything else: a login session, or a service of the
    account's own systemd (user.slice), which the account may name as it likes. Only root can
    put a process into a group below system.slice."""
    try:
        with open(os.path.join(proc, str(pid), 'cgroup'), encoding='utf-8') as f:
            lines = f.read().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in lines:
        hierarchy, _, group = line.split(':', 2)
        if hierarchy != '0':  # the unified (cgroup v2) hierarchy only
            continue
        # /system.slice/<unit> or /system.slice/system-<template>.slice/<unit>@<instance>.service
        parts = group.split('/')
        if len(parts) < 3 or parts[0] != '' or parts[1] != 'system.slice':
            return None
        middle = parts[2:-1]
        if not parts[-1].endswith('.service') or any(not part.endswith('.slice')
                                                     for part in middle):
            return None
        return parts[-1]
    return None


def flock_holders(file_path: str, proc: str = PROC) -> Set[int]:
    """The PIDs holding a flock (shared or exclusive) on that file, from /proc/locks."""
    status = os.stat(file_path)
    holders = set()
    with open(os.path.join(proc, 'locks'), encoding='ascii') as f:
        for line in f:
            # "1: FLOCK  ADVISORY  WRITE 1234 08:02:131073 0 EOF"; waiters have "->" after the id
            fields = line.split()
            if len(fields) < 6 or fields[1] != 'FLOCK':
                continue
            device, inode = fields[5].rsplit(':', 1)
            major, minor = (int(part, 16) for part in device.split(':'))
            if (int(inode) == status.st_ino and major == os.major(status.st_dev)
                    and minor == os.minor(status.st_dev)):
                holders.add(int(fields[4]))
    return holders
