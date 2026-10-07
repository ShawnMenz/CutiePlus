"""Where the shared service keeps its state and its socket, derived the same way by the
coordinator and by the labeling windows."""
import hashlib
import os
from os import path
from typing import Optional

# Unix socket paths are limited to 108 bytes (including the terminating zero)
SOCKET_PATH_LIMIT = 100


def default_state_dir(task_root: str) -> str:
    return path.join(path.abspath(task_root), '.cutieplus')


def default_socket_path(state_dir: str) -> str:
    """<state dir>/coordinator.sock, or, when that path is too long for a socket, a short one in
    the user's runtime directory named after the state directory."""
    socket_path = path.join(path.abspath(state_dir), 'coordinator.sock')
    if len(socket_path.encode()) <= SOCKET_PATH_LIMIT:
        return socket_path
    runtime = os.environ.get('XDG_RUNTIME_DIR') or '/tmp'
    digest = hashlib.sha1(path.abspath(state_dir).encode()).hexdigest()[:16]
    return path.join(runtime, f'cutieplus-{os.getuid()}-{digest}.sock')


def socket_for(task_root: str, state_dir: Optional[str] = None, socket: Optional[str] = None) -> str:
    if socket:
        return socket
    return default_socket_path(state_dir or default_state_dir(task_root))
