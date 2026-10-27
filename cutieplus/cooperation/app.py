"""Building the service with every operation registered (each module registers its own)."""
import functools
from typing import Callable, Optional

from cutieplus.cooperation import auth
from cutieplus.cooperation.clock import Clock
from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.db import Database
from cutieplus.cooperation.procinfo import system_unit_of
from cutieplus.cooperation.service import Service


def build_service(config: ServiceConfig, db: Optional[Database] = None,
                  clock: Optional[Clock] = None, passwords: Optional[auth.Passwords] = None,
                  unit_resolver: Callable[[int], Optional[str]] = system_unit_of,
                  helper=None, gpu: bool = False) -> Service:
    """helper: the root helper's client (call(op, **args)); with one, the service reconciles the
    system (SSH policy, connections, desktops) with its database. gpu: run the GPU queue (its
    worker socket; started with service.gpu.start())."""
    # registering their operations
    from cutieplus.cooperation import desktops, gpu as gpu_module, keys, users  # noqa: F401
    from cutieplus.cooperation.operations import Reconciler, on_user_change
    if db is None:
        db = Database(config.db_path, clock, config.require_patched_sqlite)
    service = Service(config, db, clock, passwords, unit_resolver)
    service.user_hooks.append(functools.partial(on_user_change, service))
    service.user_details.append(functools.partial(keys.key_details, service))
    service.user_details.append(functools.partial(desktops.desktop_details, service))
    if helper is not None:
        service.reconciler = Reconciler(service, helper)
    if gpu:
        service.gpu = gpu_module.GpuQueue(service)
        if helper is not None:
            service.gpu.link.restart_worker = lambda: helper.call('restart_gpu_worker')
    return service
