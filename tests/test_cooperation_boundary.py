"""
The cooperation module's boundary (cutieplus/cooperation): the socket protocol, what the kernel
says about a peer process, the two configuration files, and a client talking to a server.
"""
import fcntl
import os
import socket
import subprocess
import sys

import numpy as np
import pytest

from cutieplus.cooperation import procinfo
from cutieplus.cooperation.client import ClientError, ServiceClient, ServiceUnavailable
from cutieplus.cooperation.config import (CLIENT_CONFIG_VARIABLE, ConfigError, ServiceConfig,
                                          client_config)
from cutieplus.cooperation.protocol import (ProtocolError, error, ok, receive_message,
                                            send_message)
from cutieplus.cooperation.server import Server


def test_messages_carry_arrays_and_bytes_but_never_objects():
    left, right = socket.socketpair()
    with left, right:
        message = {'mask': np.arange(12, dtype=np.uint8).reshape(3, 4), 'png': b'\x89PNG',
                   'items': [1, 'two', None], 'nested': {'prob': np.ones((2, 2), np.float16)}}
        send_message(left, message)
        received = receive_message(right)
        assert received['mask'].dtype == np.uint8 and (received['mask'] == message['mask']).all()
        assert received['png'] == b'\x89PNG' and received['items'] == [1, 'two', None]
        assert received['nested']['prob'].dtype == np.float16
        # an object array would be unpickled by numpy: refused
        send_message(left, {'x': {'__nd__': 0, 'dtype': '|O', 'shape': [1]}})
        with pytest.raises(ProtocolError):
            receive_message(right)


def test_the_kernel_names_the_peer_and_its_process():
    left, right = socket.socketpair()
    with left, right:
        peer = procinfo.peer_of(right)
    assert (peer.pid, peer.uid) == (os.getpid(), os.getuid())
    me = procinfo.process_identity(os.getpid())
    assert me.pid == os.getpid() and procinfo.same_process_alive(me)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    identity = procinfo.process_identity(child.pid)
    child.kill()
    child.wait()
    assert identity is not None and not procinfo.same_process_alive(identity)
    # the same PID with another start time is another process
    assert not procinfo.same_process_alive(
        procinfo.ProcessIdentity(me.pid, me.start_ticks + 1, me.boot_id))


def write_cgroup(tmp_path, group: str) -> str:
    proc = tmp_path / 'proc'
    (proc / '42').mkdir(parents=True)
    (proc / '42' / 'cgroup').write_text(f'0::{group}\n')
    return str(proc)


@pytest.mark.parametrize('group, unit', [
    ('/system.slice/cutieplus-gpu-worker.service', 'cutieplus-gpu-worker.service'),
    ('/system.slice/system-cutieplus\\x2ddesktop.slice/cutieplus-desktop@2.service',
     'cutieplus-desktop@2.service'),
    # the account's own systemd can name a service anything: never trusted
    ('/user.slice/user-1000.slice/user@1000.service/app.slice/cutieplus-gpu-worker.service', None),
    ('/user.slice/user-1000.slice/session-4.scope', None),
    ('/system.slice/cutieplus-gpu-worker.service/child', None),
])
def test_only_a_system_service_names_a_process(tmp_path, group, unit):
    assert procinfo.system_unit_of(42, write_cgroup(tmp_path, group)) == unit


def test_flock_holders_come_from_proc_locks(tmp_path):
    lock_file = tmp_path / 'gpu.lock'
    lock_file.write_text('')
    assert procinfo.flock_holders(str(lock_file)) == set()
    with open(lock_file) as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        assert procinfo.flock_holders(str(lock_file)) == {os.getpid()}
    assert procinfo.flock_holders(str(lock_file)) == set()


def test_the_service_config_reads_sections_and_refuses_unknown_settings():
    config = ServiceConfig.from_mapping({'paths': {'workspace_root': '/data/workspace'},
                                         'desktops': {'first_display': 3, 'last_display': 5}})
    assert config.jobs_dir == '/data/workspace/.cutie_jobs'
    assert config.vnc_port_of(3) == 5903 and config.desktop_unit_of(4) == \
        'cutieplus-desktop@4.service'
    with pytest.raises(ConfigError):
        ServiceConfig.from_mapping({'paths': {'workspace_root': '/w', 'shell': '/bin/sh'}})
    with pytest.raises(ConfigError):
        ServiceConfig.from_mapping({'workspace_root': 'relative'})


def test_a_host_without_client_config_runs_the_windows_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv(CLIENT_CONFIG_VARIABLE, str(tmp_path / 'missing.toml'))
    assert client_config() is None
    config = tmp_path / 'client.toml'
    config.write_text('mode = "cooperation"\napp_socket = "/run/x.sock"\n')
    monkeypatch.setenv(CLIENT_CONFIG_VARIABLE, str(config))
    assert client_config().app_socket == '/run/x.sock'


def test_a_client_talks_to_a_server(tmp_path):
    seen = []

    def handler(request, peer):
        seen.append((request['op'], request.get('token'), peer.pid))
        if request['op'] == 'fail':
            return error('forbidden', 'not you', target='x')
        return ok({'echo': request['args']})

    socket_path = str(tmp_path / 'app.sock')
    server = Server(handler, socket_path, mode=0o666)
    server.start()
    try:
        assert os.stat(socket_path).st_mode & 0o777 == 0o666
        client = ServiceClient(socket_path, timeout=5)
        assert client.call('ping', value=1) == {'echo': {'value': 1}}
        client.token = 'secret'
        with pytest.raises(ClientError) as refused:
            client.call('fail')
        assert refused.value.code == 'forbidden' and refused.value.details == {'target': 'x'}
        assert seen == [('ping', None, os.getpid()), ('fail', 'secret', os.getpid())]
        client.close()
    finally:
        server.stop()
    with pytest.raises(ServiceUnavailable):
        ServiceClient(socket_path, timeout=1).call('ping')
