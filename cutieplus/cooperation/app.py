"""Building the service with every operation registered (each module registers its own)."""
from typing import Callable, Optional

from cutieplus.cooperation import auth
from cutieplus.cooperation.clock import Clock
from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.db import Database
from cutieplus.cooperation.procinfo import system_unit_of
from cutieplus.cooperation.service import Service


def build_service(config: ServiceConfig, db: Optional[Database] = None,
                  clock: Optional[Clock] = None, passwords: Optional[auth.Passwords] = None,
                  unit_resolver: Callable[[int], Optional[str]] = system_unit_of) -> Service:
    from cutieplus.cooperation import users  # noqa: F401 (registers its operations)
    if db is None:
        db = Database(config.db_path, clock, config.require_patched_sqlite)
    return Service(config, db, clock, passwords, unit_resolver)
