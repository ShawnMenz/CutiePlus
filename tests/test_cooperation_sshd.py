"""
The annotation SSH entrypoint against a real OpenSSH sshd, run unprivileged on a free port with the
configuration and policy file the cooperation module writes (sshpolicy.render_sshd_config,
render_policy): a collaborator's key forwards to their own desktop's VNC port and nowhere else,
gets no shell and no SFTP, and stops working once its revocation is published; the connection it
has open ends when its sshd process is cut. sshd's own log line is the one the helper reads from
the journal.

The real entrypoint runs as root with privilege separation and keeps StrictModes; the test turns
StrictModes off only because its files lie below /tmp.
"""
import getpass
import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import pytest

from cutieplus.cooperation import helper, sshpolicy

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which('sshd') is None and not
                                                   os.path.exists('/usr/sbin/sshd'),
                                                   reason='no OpenSSH server')]
KEY_ID = '1f0e2d3c-4b5a-4968-8776-655443322110'


def free_ports(count: int):
    """count consecutive free ports."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            first = probe.getsockname()[1]
        sockets = []
        try:
            for port in range(first, first + count):
                sock = socket.socket()
                sock.bind(('127.0.0.1', port))
                sockets.append(sock)
            return list(range(first, first + count))
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError('no free ports')


def echo_server(port: int, stop: threading.Event):
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen()
    server.settimeout(0.2)

    def serve():
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                data = connection.recv(64)
                connection.sendall(b'vnc:' + data)
        server.close()

    threading.Thread(target=serve, daemon=True).start()


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(port)


def through(port: int) -> bytes:
    with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
        sock.sendall(b'hello')
        try:
            return sock.recv(64)
        except (ConnectionResetError, socket.timeout):
            return b''


def children_of(pid: int):
    found = []
    for name in os.listdir('/proc'):
        if name.isdigit():
            try:
                with open(f'/proc/{name}/stat') as f:
                    stat = f.read()
            except OSError:
                continue
            if int(stat[stat.rindex(')') + 2:].split()[1]) == pid:
                found.append(int(name))
    return found


@pytest.fixture
def entrypoint(tmp_path):
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(tmp_path / 'host')],
                   check=True)
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(tmp_path / 'alice')],
                   check=True)
    ssh_port, own_port, other_port = free_ports(3)
    policy_file = tmp_path / 'authorized_keys'
    alice = sshpolicy.parse_public_key((tmp_path / 'alice.pub').read_text())
    ports = (own_port, other_port)

    def publish(entries):
        policy_file.write_text(sshpolicy.render_policy(1, entries, ports))

    publish([sshpolicy.PolicyEntry(KEY_ID, alice.canonical, own_port)])
    config = sshpolicy.render_sshd_config(ssh_port, str(policy_file), ports,
                                          pid_file=str(tmp_path / 'sshd.pid'),
                                          host_keys=[str(tmp_path / 'host')],
                                          user=getpass.getuser())
    config = config.replace('StrictModes yes', 'StrictModes no')
    (tmp_path / 'sshd_config').write_text(config)
    log = open(tmp_path / 'sshd.log', 'w')
    sshd = subprocess.Popen([shutil.which('sshd') or '/usr/sbin/sshd', '-D', '-e', '-f',
                             str(tmp_path / 'sshd_config')], stderr=log)
    stop = threading.Event()
    echo_server(own_port, stop)
    echo_server(other_port, stop)
    try:
        wait_for_port(ssh_port)
        yield {'tmp': tmp_path, 'ssh_port': ssh_port, 'own': own_port, 'other': other_port,
               'sshd': sshd, 'publish': publish, 'alice': alice}
    finally:
        stop.set()
        sshd.send_signal(signal.SIGTERM)
        sshd.wait(10)
        log.close()


def ssh(entrypoint, *arguments, background=False):
    command = ['ssh', '-F', '/dev/null', '-i', str(entrypoint['tmp'] / 'alice'),
               '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes',
               '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
               '-o', 'LogLevel=ERROR', '-o', 'ExitOnForwardFailure=yes',
               '-p', str(entrypoint['ssh_port']), *arguments, f'{getpass.getuser()}@127.0.0.1']
    if background:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def test_a_key_reaches_its_own_desktop_only(entrypoint):
    local_own, local_other = free_ports(2)
    tunnel = ssh(entrypoint, '-N', '-L', f'{local_own}:localhost:{entrypoint["own"]}',
                 '-L', f'{local_other}:localhost:{entrypoint["other"]}', background=True)
    try:
        wait_for_port(local_own)
        assert through(local_own) == b'vnc:hello'
        assert through(local_other) == b''  # another user's desktop: the channel is refused
    finally:
        tunnel.terminate()
        tunnel.wait(10)
    # the line sshd logs is the one the helper reads from the journal
    log = (entrypoint['tmp'] / 'sshd.log').read_text()
    accepted = [line for line in log.splitlines() if 'Accepted publickey' in line]
    match = helper.ACCEPTED.fullmatch(accepted[0].strip())
    assert match and match.group('fingerprint') == entrypoint['alice'].fingerprint


def test_no_shell_command_or_sftp(entrypoint):
    shell = ssh(entrypoint, 'id')
    assert shell.returncode != 0 and 'uid=' not in shell.stdout
    tty = ssh(entrypoint, '-tt')
    assert tty.returncode != 0 and 'uid=' not in tty.stdout
    sftp = subprocess.run(
        ['sftp', '-F', '/dev/null', '-i', str(entrypoint['tmp'] / 'alice'), '-o',
         'IdentitiesOnly=yes', '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '-o',
         'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-P',
         str(entrypoint['ssh_port']), '-b', '-', f'{getpass.getuser()}@127.0.0.1'],
        input='ls /\n', capture_output=True, text=True, timeout=30)
    assert sftp.returncode != 0
    # a remote forward (opening a port here) is refused too
    remote = ssh(entrypoint, '-N', '-R', f'{free_ports(1)[0]}:localhost:22', '-o',
                 'ConnectTimeout=10')
    assert remote.returncode != 0


def test_a_published_revocation_blocks_the_key_and_cutting_ends_its_tunnel(entrypoint):
    local = free_ports(1)[0]
    tunnel = ssh(entrypoint, '-N', '-L', f'{local}:localhost:{entrypoint["own"]}',
                 background=True)
    wait_for_port(local)
    assert through(local) == b'vnc:hello'
    entrypoint['publish']([])  # revoked
    refused = ssh(entrypoint, '-N', '-L', f'{free_ports(1)[0]}:localhost:{entrypoint["own"]}')
    assert refused.returncode == 255 and 'Permission denied' in refused.stderr
    # the open connection stays until its sshd process is cut, as the helper does
    for pid in children_of(entrypoint['sshd'].pid):
        os.kill(pid, signal.SIGTERM)
    tunnel.wait(10)
    assert tunnel.returncode is not None


# through port 22: collaborators hop from port 22's sshd to the annotation sshd, with ssh -J, as
# the jump account (played by the test's own account here). Port 22's own settings are as
# permissive as a stock sshd's: the jump account's drop-in (render_jump_dropin) and the jump
# file's lines do the restricting

@pytest.fixture
def hop(entrypoint):
    tmp = entrypoint['tmp']
    jump_port = free_ports(1)[0]
    alice = entrypoint['alice']

    def publish_jump(entries):
        (tmp / 'jump_keys').write_text(sshpolicy.render_jump_policy(
            1, entries, (entrypoint['own'], entrypoint['other']), entrypoint['ssh_port']))

    publish_jump([sshpolicy.PolicyEntry(KEY_ID, alice.canonical, entrypoint['own'])])
    me = getpass.getuser()
    (tmp / 'jump.conf').write_text(sshpolicy.render_jump_dropin(me, str(tmp / 'jump_keys'),
                                                                entrypoint['ssh_port']))
    (tmp / 'port22_config').write_text('\n'.join([
        f'Include {tmp / "jump.conf"}',
        f'Port {jump_port}', 'ListenAddress 127.0.0.1', f'HostKey {tmp / "host"}',
        f'PidFile {tmp / "port22.pid"}', 'UsePAM no', 'StrictModes no',
        'AllowTcpForwarding yes', 'AllowStreamLocalForwarding yes', 'PermitTTY yes',
        'GatewayPorts no', 'Subsystem sftp internal-sftp', f'AllowUsers {me}',
        'LogLevel VERBOSE']) + '\n')
    log = open(tmp / 'port22.log', 'w')
    sshd = subprocess.Popen([shutil.which('sshd') or '/usr/sbin/sshd', '-D', '-e', '-f',
                             str(tmp / 'port22_config')], stderr=log)
    common = ['  User ' + me, f'  IdentityFile {tmp / "alice"}', '  IdentitiesOnly yes',
              '  IdentityAgent none', '  BatchMode yes', '  StrictHostKeyChecking no',
              '  UserKnownHostsFile /dev/null', '  LogLevel ERROR', '  ExitOnForwardFailure yes']
    (tmp / 'client_config').write_text('\n'.join(
        ['Host port22', '  HostName 127.0.0.1', f'  Port {jump_port}', *common,
         'Host desktop', '  HostName localhost', f'  Port {entrypoint["ssh_port"]}',
         '  ProxyJump port22', *common]) + '\n')
    try:
        wait_for_port(jump_port)
        yield dict(entrypoint, publish_jump=publish_jump)
    finally:
        sshd.send_signal(signal.SIGTERM)
        sshd.wait(10)
        log.close()


def client(hop, *arguments, background=False):
    command = ['ssh', '-F', str(hop['tmp'] / 'client_config'), *arguments]
    if background:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def test_through_port_22_a_key_reaches_its_own_desktop_only(hop):
    local_own, local_other = free_ports(2)
    tunnel = client(hop, '-N', '-L', f'{local_own}:localhost:{hop["own"]}', '-L',
                    f'{local_other}:localhost:{hop["other"]}', 'desktop', background=True)
    try:
        wait_for_port(local_own)
        assert through(local_own) == b'vnc:hello'
        assert through(local_other) == b''  # the annotation sshd refuses another desktop
    finally:
        tunnel.terminate()
        tunnel.wait(10)
    log = (hop['tmp'] / 'sshd.log').read_text()  # the annotation sshd saw it, from here
    assert any('Accepted publickey' in line and '127.0.0.1' in line for line in log.splitlines())


def test_port_22_lets_the_key_open_nothing_but_the_annotation_sshd(hop):
    assert 'uid=' not in client(hop, 'port22', 'id').stdout  # no command
    assert 'uid=' not in client(hop, '-tt', 'port22').stdout  # no shell
    sftp = subprocess.run(['sftp', '-F', str(hop['tmp'] / 'client_config'), '-b', '-',
                           'port22'], input='ls /\n', capture_output=True, text=True,
                          timeout=30)
    assert sftp.returncode != 0
    # straight to a desktop's VNC port, past the annotation sshd: refused
    local = free_ports(1)[0]
    tunnel = client(hop, '-N', '-L', f'{local}:localhost:{hop["own"]}', 'port22',
                    background=True)
    try:
        wait_for_port(local)
        assert through(local) == b''
    finally:
        tunnel.terminate()
        tunnel.wait(10)


def unix_echo_server(socket_path: str, stop: threading.Event) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen()
    server.settimeout(0.2)

    def serve():
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.sendall(b'socket:' + connection.recv(64))
        server.close()

    threading.Thread(target=serve, daemon=True).start()


def test_port_22_forwards_no_unix_socket_and_listens_nowhere(hop):
    """Port-forwarding in a key's options gives back every kind of forwarding: the jump
    account's drop-in refuses all but the TCP channel to the annotation sshd."""
    # a socket this computer's account could reach (a stand-in for Docker's, or D-Bus's)
    socket_path = str(hop['tmp'] / 'private.sock')
    stop = threading.Event()
    unix_echo_server(socket_path, stop)
    local = free_ports(1)[0]
    tunnel = client(hop, '-N', '-L', f'{local}:{socket_path}', 'port22', background=True)
    try:
        wait_for_port(local)
        assert through(local) == b''
    finally:
        stop.set()
        tunnel.terminate()
        tunnel.wait(10)
    for remote in (f'{free_ports(1)[0]}:localhost:22', f'{hop["tmp"] / "planted.sock"}:localhost:22'):
        refused = client(hop, '-N', '-R', remote, '-o', 'ConnectTimeout=10', 'port22')
        assert refused.returncode == 255, remote
    assert not (hop['tmp'] / 'planted.sock').exists()


def test_a_revoked_key_no_longer_passes_port_22(hop):
    hop['publish_jump']([])
    refused = client(hop, '-N', '-L', f'{free_ports(1)[0]}:localhost:{hop["own"]}', 'desktop')
    assert refused.returncode == 255 and 'Permission denied' in refused.stderr
