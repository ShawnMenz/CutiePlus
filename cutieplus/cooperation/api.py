"""
The service's operations, registered by name with @op. A handler is f(service, ctx, **args); the
arguments it declares are the only ones a request may carry: any other field is refused
(unexpected_field), so a request cannot slip in a role, an actor or an owner the handler does not
expect. Public operations (hello, login, setup_credential) need no session; every other one runs
with ctx.actor, the user of the request's validated session.
"""
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Optional

from cutieplus.cooperation.authz import Actor
from cutieplus.cooperation.procinfo import Peer


@dataclass(frozen=True)
class Operation:
    name: str
    function: Callable
    public: bool
    parameters: FrozenSet[str]
    required: FrozenSet[str]


OPS: Dict[str, Operation] = {}


def op(name: Optional[str] = None, public: bool = False):

    def register(function: Callable) -> Callable:
        parameters = list(inspect.signature(function).parameters.values())[2:]  # service, ctx
        operation = Operation(
            name or function.__name__, function, public,
            frozenset(parameter.name for parameter in parameters),
            frozenset(parameter.name for parameter in parameters
                      if parameter.default is inspect.Parameter.empty))
        if operation.name in OPS:
            raise ValueError(f'{operation.name} is registered twice')
        OPS[operation.name] = operation
        return function

    return register


@dataclass(frozen=True)
class Origin:
    """Where a request comes from, as the kernel says: a managed remote desktop (display), or
    'console' for anything else (the host's own login)."""
    display: Optional[int] = None
    slot: Optional[dict] = None  # the desktop's slot; None: console, or a display nobody has

    @property
    def is_desktop(self) -> bool:
        return self.display is not None

    @property
    def key(self) -> str:
        return f'desktop:{self.display}' if self.is_desktop else 'console'

    @property
    def slot_id(self) -> Optional[str]:
        return self.slot['id'] if self.slot else None


@dataclass
class Context:
    peer: Peer
    origin: Origin
    request_id: Optional[str] = None
    actor: Optional[Actor] = None
    session: Optional[dict] = field(default=None, repr=False)
