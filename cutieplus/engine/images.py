"""Reading frames ahead of propagation, with a bounded number of frames in flight."""
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


def read_rgb(file_path: str) -> np.ndarray:
    """H * W * 3 uint8, as the labeling window reads frames."""
    return np.array(Image.open(file_path).convert('RGB'))


def read_index_mask(file_path: str) -> np.ndarray:
    return np.array(Image.open(file_path))


def read_mask_source(source: Optional[dict], shape: Optional[tuple] = None) -> Optional[np.ndarray]:
    """A mask given as {'path': ..., 'binary': bool}: an index mask, or a binary image (a prefill:
    any non-zero pixel is the target, as the labeling window reads it). None gives zeros of
    `shape` when a shape is given, else None."""
    if source is None:
        return None if shape is None else np.zeros(shape, np.uint8)
    mask = read_index_mask(source['path'])
    if source.get('binary'):
        mask = (mask > 0).astype(np.uint8)
    return mask


class FramePrefetcher:
    """Reads the frames of `order` ahead of their use with a few threads; at most `depth` frames
    are read but not yet taken. get() of a frame that was not prefetched reads it directly."""

    def __init__(self, read: Callable[[int], np.ndarray], order: Iterable[int], depth: int = 4,
                 threads: int = 2):
        self.read = read
        self.order: List[int] = list(order)
        self.depth = max(0, depth)
        self.position = 0
        self.pending: Dict[int, Future] = {}
        self.pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=threads, thread_name_prefix='prefetch')
            if self.depth > 0 else None)
        self._fill()

    def _fill(self) -> None:
        if self.pool is None:
            return
        while len(self.pending) < self.depth and self.position < len(self.order):
            ti = self.order[self.position]
            self.position += 1
            if ti not in self.pending:
                self.pending[ti] = self.pool.submit(self.read, ti)

    def get(self, ti: int) -> np.ndarray:
        future = self.pending.pop(ti, None)
        image = future.result() if future is not None else self.read(ti)
        self._fill()
        return image

    def set_depth(self, depth: int) -> None:
        """Fewer frames in flight (memory pressure); frames already read are kept."""
        self.depth = max(0, depth)

    def close(self) -> None:
        if self.pool is not None:
            for future in self.pending.values():
                future.cancel()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        self.pending.clear()
