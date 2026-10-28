"""
deploy/install.sh, the parts that run without root: --ssh-host sets the address collaborators'
ssh command names this computer by (ssh_host in /etc/cutieplus/service.toml), replacing the one
there, adding it when missing, and refusing anything but an address or a host name; a new
service.toml takes the address of the default route. The functions are taken from the script and
run with bash on copies of a service.toml. Whatever service.toml says, the cards never show this
computer's own host name, which resolves on it only: they show its IP address.
"""
import re
import subprocess
from os import path

from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.operations import desktop_card

SCRIPT = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'deploy', 'install.sh')
SERVICE_TOML = """[paths]
workspace_root = "/srv/workspace"

[ssh]
ssh_port = 2222
# how collaborators name this computer
ssh_host = "lab-host"
desktop_user = "labeler"
"""


def functions(*names: str) -> str:
    with open(SCRIPT, encoding='utf-8') as f:
        script = f.read()
    return '\n'.join(re.search(rf'^{name}\(\) \{{\n.*?^\}}\n', script, re.M | re.S).group(0)
                     for name in names)


def set_ssh_host(file_path, host: str) -> int:
    code = functions('valid_host', 'set_ssh_host') + 'set_ssh_host "$0" "$1"'
    return subprocess.run(['bash', '-c', code, str(file_path), host]).returncode


def test_ssh_host_is_replaced_and_the_cards_use_it(tmp_path):
    toml = tmp_path / 'service.toml'
    toml.write_text(SERVICE_TOML)
    assert set_ssh_host(toml, '203.0.113.10') == 0
    assert toml.read_text().count('ssh_host') == 1
    config = ServiceConfig.load(str(toml))
    assert config.ssh_host == '203.0.113.10' and config.desktop_user == 'labeler'

    class Service:  # what desktop_card reads
        pass

    service = Service()
    service.config = config
    # through port 22 to the annotation sshd on this computer, then the desktop's VNC port
    assert desktop_card(service, 2)['ssh_command'] == \
        'ssh -N -o ExitOnForwardFailure=yes -J cutieplus-jump@203.0.113.10 -p 2222 ' \
        '-L 5902:localhost:5902 labeler@localhost'
    assert desktop_card(service, 2)['vnc_address'] == 'localhost:5902'  # the same port


def test_the_cards_show_the_address_never_the_own_host_name(monkeypatch):
    from cutieplus.cooperation import config as config_module
    monkeypatch.setattr(config_module, 'default_address', lambda: '203.0.113.10')
    monkeypatch.setattr(config_module.socket, 'gethostname', lambda: 'lab-host')

    def shown(ssh_host: str) -> str:
        return ServiceConfig(state_dir='/srv', workspace_root='/srv',
                             ssh_host=ssh_host).public_host

    for own in ('', 'lab-host', 'LAB-HOST'):
        assert shown(own) == '203.0.113.10'
    assert shown('aurora.example.edu') == 'aurora.example.edu'  # set on purpose: kept
    assert shown('10.0.0.7') == '10.0.0.7'

    class Service:  # what desktop_card reads
        config = ServiceConfig(state_dir='/srv', workspace_root='/srv',
                               ssh_host='lab-host', desktop_user='labeler')

    command = desktop_card(Service(), 2)['ssh_command']
    assert ' -J cutieplus-jump@203.0.113.10 ' in command and 'lab-host' not in command
    monkeypatch.setattr(config_module, 'default_address', lambda: None)  # no route at all
    assert shown('') == 'lab-host'


def test_ssh_host_is_added_when_missing(tmp_path):
    under_ssh = tmp_path / 'with-section.toml'
    under_ssh.write_text(SERVICE_TOML.replace('ssh_host = "lab-host"\n', ''))
    assert set_ssh_host(under_ssh, 'aurora.example.edu') == 0
    assert ServiceConfig.load(str(under_ssh)).ssh_host == 'aurora.example.edu'
    no_section = tmp_path / 'without-section.toml'
    no_section.write_text('desktop_user = "labeler"\n[paths]\nworkspace_root = "/srv/workspace"\n')
    assert set_ssh_host(no_section, '10.0.0.7') == 0
    assert ServiceConfig.load(str(no_section)).ssh_host == '10.0.0.7'


def test_anything_but_an_address_is_refused(tmp_path):
    toml = tmp_path / 'service.toml'
    toml.write_text(SERVICE_TOML)
    for bad in ('1.2.3.4"; evil', 'a b', '', 'x|y', 'host\nssh_port = 22'):
        assert set_ssh_host(toml, bad) != 0
    assert toml.read_text() == SERVICE_TOML


def test_a_new_service_toml_names_the_default_routes_address():
    code = functions('default_host') + 'default_host'
    address = subprocess.run(['bash', '-c', 'set -euo pipefail; ' + code], capture_output=True,
                             text=True, check=True).stdout.strip()
    assert address  # an address where there is a route, else the host name
    with open(SCRIPT, encoding='utf-8') as f:
        script = f.read()
    assert 'ssh_host = "${SSH_HOST:-$(default_host)}"' in script
    # collaborators hop on port 22 as an account of their own, held by its drop-in
    assert 'useradd --system --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin ' \
        'cutieplus-jump' in script
    assert 'render_jump_dropin(c.jump_user, c.jump_keys, c.ssh_port)' in script
    assert subprocess.run(['bash', '-n', SCRIPT]).returncode == 0
