"""Latency of requests and GPU work, kept in bounded rings for the status report."""
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List, Sequence, Tuple


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float('nan')
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Metrics:

    def __init__(self, keep: int = 5000):
        self.keep = keep
        self.samples: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.keep))
        self.lock = threading.Lock()

    def record(self, name: str, seconds: float, wait: float) -> None:
        """seconds: time spent doing it; wait: time spent queued before."""
        with self.lock:
            self.samples[name].append((seconds, wait))

    def values(self, name: str) -> List[Tuple[float, float]]:
        with self.lock:
            return list(self.samples.get(name, ()))

    def summary(self) -> dict:
        with self.lock:
            items = {name: list(values) for name, values in self.samples.items()}
        summary = {}
        for name, values in items.items():
            seconds = [value[0] for value in values]
            waits = [value[1] for value in values]
            summary[name] = {'count': len(values), 'p50': percentile(seconds, 0.5),
                             'p95': percentile(seconds, 0.95),
                             'wait_p50': percentile(waits, 0.5), 'wait_p95': percentile(waits, 0.95)}
        return summary
