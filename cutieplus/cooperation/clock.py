"""Times in UTC, written in one fixed format so that comparing the strings compares the times."""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def parse(text: Optional[str]) -> Optional[datetime]:
    if text is None:
        return None
    return datetime.strptime(text, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)


class Clock:
    """The service's time; the tests move a FakeClock instead of waiting."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def iso_now(self) -> str:
        return iso(self.now())

    def iso_in(self, **delta) -> str:
        return iso(self.now() + timedelta(**delta))


class FakeClock(Clock):

    def __init__(self, start: Optional[datetime] = None):
        self.moment = start or datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.moment

    def advance(self, **delta) -> None:
        self.moment += timedelta(**delta)


def new_id() -> str:
    return str(uuid.uuid4())
