"""
Immutable mask objects: every committed mask is a PNG file named by the SHA-256 of its bytes,
written to a unique temporary file, synced, and renamed into place; it is never changed after.
The same mask committed twice is the same object, so a retried commit writes nothing new.
"""
import hashlib
import os
import uuid
from os import path
from typing import Iterator, Optional

import numpy as np

from cutieplus.engine.images import read_index_mask


def fsync_directory(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_file_durably(file_path: str, data: bytes, mode: Optional[int] = None) -> None:
    """Writes to a unique temporary file in the same directory, syncs it, renames it over
    file_path and syncs the directory: readers see the old file or the new one, never half."""
    directory = path.dirname(file_path)
    temporary = path.join(directory, f'.{path.basename(file_path)}.{uuid.uuid4().hex}.tmp')
    try:
        with open(temporary, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, file_path)
    except BaseException:
        if path.exists(temporary):
            os.remove(temporary)
        raise
    fsync_directory(directory)


class ObjectStore:

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, sha: str) -> str:
        return path.join(self.root, sha[:2], sha + '.png')

    def put(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        final = self.path(sha)
        if path.exists(final) and path.getsize(final) == len(data):
            return sha  # immutable: already published
        os.makedirs(path.dirname(final), exist_ok=True)
        write_file_durably(final, data, mode=0o444)
        return sha

    def get(self, sha: str) -> bytes:
        with open(self.path(sha), 'rb') as f:
            return f.read()

    def mask(self, sha: str) -> np.ndarray:
        return read_index_mask(self.path(sha))

    def exists(self, sha: str) -> bool:
        return path.exists(self.path(sha))

    def objects(self) -> Iterator[str]:
        for directory in os.listdir(self.root):
            full = path.join(self.root, directory)
            if path.isdir(full):
                for name in os.listdir(full):
                    if name.endswith('.png') and not name.startswith('.'):
                        yield name[:-4]

    def remove_temporary_files(self) -> int:
        """Temporary files of writes interrupted by a crash."""
        removed = 0
        for directory, _, files in os.walk(self.root):
            for name in files:
                if name.endswith('.tmp'):
                    os.remove(path.join(directory, name))
                    removed += 1
        return removed

    def remove(self, sha: str) -> None:
        file_path = self.path(sha)
        if path.exists(file_path):
            os.chmod(file_path, 0o644)
            os.remove(file_path)
