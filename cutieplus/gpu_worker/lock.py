"""
The GPU execution lock: an exclusive flock on a root-owned file (service.toml gpu_lock), taken by
the worker before CUDA is initialized and held while the process lives. A second worker cannot
take it, nor can a replacement while any process that inherited the lock's file (a child of the
previous worker) still runs. The file is never deleted or replaced: a lock on a new file would
exclude nothing. It coordinates CutiePlus's own processes; other programs of the account may still
use the GPU.
"""
import fcntl
import os
from typing import IO, Set

from cutieplus.cooperation.procinfo import flock_holders


class GpuBusy(Exception):
    """Another process holds the GPU lock."""

    def __init__(self, lock_path: str, holders: Set[int]):
        super().__init__(f'the GPU lock {lock_path} is held by '
                         f'{", ".join(map(str, sorted(holders))) or "another process"}')
        self.holders = holders


def acquire_gpu_lock(lock_path: str) -> IO:
    """The open lock file, locked; keep it open for as long as the GPU is used."""
    stream = open(lock_path, 'rb')  # reading is enough to flock; the file stays root's
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise GpuBusy(lock_path, flock_holders(lock_path)) from None
    return stream


def holds_gpu_lock(lock_path: str, pid: int = None) -> bool:
    return (pid or os.getpid()) in flock_holders(lock_path)
