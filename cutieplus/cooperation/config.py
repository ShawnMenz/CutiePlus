"""
The cooperation module's two configuration files, both written by the installer (deploy/) and
owned by root:

/etc/cutieplus/service.toml   the service's paths, the managed display and port ranges, the SSH
                              entrypoint, the systemd units it may name
/etc/cutieplus/client.toml    present on a host that runs the service: every labeling window then
                              signs in and sends its GPU work to the service, and none loads a
                              model itself (client_config)

Settings an Admin may change while the service runs (timeouts, the pending-job limit) are kept in
the database (system_settings), within the bounds set here.
"""
import os
import tomllib
from dataclasses import dataclass, field, fields
from os import path
from typing import Optional

SERVICE_CONFIG = '/etc/cutieplus/service.toml'
CLIENT_CONFIG = '/etc/cutieplus/client.toml'
CLIENT_CONFIG_VARIABLE = 'CUTIEPLUS_CLIENT_CONFIG'  # another file, for the tests
COOPERATION = 'cooperation'
MIN_SQLITE = (3, 51, 3)  # the WAL-reset fix


class ConfigError(Exception):
    pass


@dataclass
class ServiceConfig:
    state_dir: str = '/var/lib/cutieplus'  # the database, snapshots, backups
    app_socket: str = '/run/cutieplus/app.sock'
    worker_socket: str = '/run/cutieplus/worker.sock'
    helper_socket: str = '/run/cutieplus-helper/helper.sock'
    gpu_lock: str = '/var/lib/cutieplus-gpu/gpu.lock'
    # the only folder jobs may name; the GPU worker stages its results below it
    workspace_root: str = ''
    staging_dir: str = ''  # default: <workspace_root>/.cutie_jobs
    # the remote desktops: display :N listens on 127.0.0.1:<vnc_port_base + N>
    first_display: int = 2
    last_display: int = 20
    vnc_port_base: int = 5900
    ssh_port: int = 2222
    ssh_host: str = ''  # how collaborators name this host; default: its host name
    worker_unit: str = 'cutieplus-gpu-worker.service'
    desktop_unit: str = 'cutieplus-desktop@{display}.service'
    sshd_unit: str = 'cutieplus-sshd.service'
    # checks the tests turn off: the SQLite version, and that a peer runs in its systemd unit
    require_patched_sqlite: bool = True
    verify_peers: bool = True
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def db_path(self) -> str:
        return path.join(self.state_dir, 'cooperation.sqlite3')

    @property
    def snapshots_dir(self) -> str:
        return path.join(self.state_dir, 'snapshots')

    @property
    def backups_dir(self) -> str:
        return path.join(self.state_dir, 'backups')

    @property
    def jobs_dir(self) -> str:
        return self.staging_dir or path.join(self.workspace_root, '.cutie_jobs')

    def desktop_unit_of(self, display: int) -> str:
        return self.desktop_unit.format(display=int(display))

    def vnc_port_of(self, display: int) -> int:
        return self.vnc_port_base + int(display)

    def check(self) -> 'ServiceConfig':
        if not path.isabs(self.workspace_root):
            raise ConfigError('workspace_root must be an absolute path')
        if not 1 <= self.first_display <= self.last_display <= 99:
            raise ConfigError('the displays must be a range within 1-99')
        if not 1024 <= self.vnc_port_of(self.first_display) <= self.vnc_port_of(
                self.last_display) <= 65535:
            raise ConfigError('the VNC ports must lie within 1024-65535')
        if not 1 <= self.ssh_port <= 65535:
            raise ConfigError('ssh_port must be a port number')
        return self

    @classmethod
    def from_mapping(cls, values: dict) -> 'ServiceConfig':
        known = {item.name for item in fields(cls)} - {'extra'}
        flat = {}
        for key, value in values.items():
            # sections only group the keys: [paths] state_dir = ... is state_dir
            for name, item in (value.items() if isinstance(value, dict) else [(key, value)]):
                if name not in known:
                    raise ConfigError(f'unknown setting {name}')
                flat[name] = item
        return cls(**flat).check()

    @classmethod
    def load(cls, file_path: str = SERVICE_CONFIG) -> 'ServiceConfig':
        with open(file_path, 'rb') as f:
            return cls.from_mapping(tomllib.load(f))


@dataclass(frozen=True)
class ClientConfig:
    mode: str
    app_socket: str


def client_config() -> Optional[ClientConfig]:
    """This host's client configuration; None when it runs no cooperation service (development,
    tests): the windows then load the models themselves, as before."""
    file_path = os.environ.get(CLIENT_CONFIG_VARIABLE) or CLIENT_CONFIG
    try:
        with open(file_path, 'rb') as f:
            values = tomllib.load(f)
    except FileNotFoundError:
        return None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f'{file_path} cannot be read: {exc}') from None
    mode = values.get('mode', COOPERATION)
    if mode != COOPERATION:
        raise ConfigError(f'{file_path}: unknown mode {mode!r}')
    return ClientConfig(mode, values.get('app_socket', ServiceConfig.app_socket))
