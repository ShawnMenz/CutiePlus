"""
The annotators' VNC screens, kept in step with the annotator list without root.

Screen :N (port 5900+N) is cutie-screen@N, an instance of a service of this account's own systemd
(UNIT_FILE, written here): `tigervncserver :N -fg` with the account's ~/.vnc settings (config,
passwd, xstartup), so its xstartup opens the task browser. Saving the list enables and starts the
screen of each saved port and disables and stops the screen of each removed one (sync). Enabled
screens start at boot once the account lingers (scripts/setup_annotator_screens.sh, run once).

A port that listens without being a cutie-screen@N, such as a system tigervncserver@:N screen, is
reported and left alone.

    python -m gui.vnc_screens sync --tasks workspace/cutie_tasks   # start every saved port
"""
import argparse
import fcntl
import os
import socket
import subprocess
import sys
import time
from os import path
from typing import Dict, Iterable, Optional, Tuple

from gui.annotators import VNC_BASE_PORT, load_annotators

UNIT = 'cutie-screen@{}.service'
UNIT_FILE = path.expanduser('~/.config/systemd/user/cutie-screen@.service')
START_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 10.0
# one sync at a time, so browsers saving together do not start the same screen twice
LOCK_FILE = path.expanduser('~/.vnc/cutie-screens.lock')
SETUP_SCRIPT = 'scripts/setup_annotator_screens.sh'

# The screen's session keeps nothing of the user manager's environment: env -i drops its DISPLAY
# (:1, the real monitor) and GDM's XAUTHORITY.
UNIT_TEXT = """\
# Written by CutiePlus (gui/vnc_screens.py): one annotator's VNC screen, :%i on port 5900+%i.
# Enabled and disabled by Save in the task browser's Manage annotators page.
[Unit]
Description=Cutie VNC screen :%i

[Service]
ExecStart=/usr/bin/env -i HOME=%h USER=%u LOGNAME=%u SHELL=/bin/bash LANG={lang} \
PATH=/usr/local/bin:/usr/bin:/bin XDG_RUNTIME_DIR=%t DBUS_SESSION_BUS_ADDRESS=unix:path=%t/bus \
/usr/bin/tigervncserver :%i -fg -localhost yes
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""

Result = Tuple[str, Optional[str]]  # (what happened, why it failed or None)


def port_listening(port: int) -> bool:
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5):
            return True
    except OSError:
        return False


def run(command: list) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=30)


def systemctl(*arguments: str) -> subprocess.CompletedProcess:
    return run(['systemctl', '--user', *arguments])


def unit_state(display: int) -> str:
    """The unit's SubState: running, auto-restart, failed, dead, ..."""
    return systemctl('show', '--property', 'SubState', '--value', UNIT.format(display)).stdout.strip()


def last_log_lines(display: int) -> str:
    result = run(['journalctl', '--user', '--unit', UNIT.format(display), '--lines', '3',
                  '--output', 'cat', '--no-pager'])
    return ' '.join(result.stdout.split())


def write_unit_file() -> None:
    """Writes UNIT_FILE when it differs and reloads the user manager."""
    text = UNIT_TEXT.replace('{lang}', os.environ.get('LANG') or 'C.UTF-8')
    try:
        with open(UNIT_FILE, encoding='utf-8') as stream:
            if stream.read() == text:
                return
    except OSError:
        pass
    os.makedirs(path.dirname(UNIT_FILE), exist_ok=True)
    temporary = UNIT_FILE + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        stream.write(text)
    os.replace(temporary, UNIT_FILE)
    systemctl('daemon-reload')


def failure(result: subprocess.CompletedProcess, fallback: str) -> str:
    return ' '.join((result.stderr or result.stdout).split()) or fallback


def enable_screen(port: int) -> Result:
    """Enables cutie-screen@N and makes sure it listens: ('started' or 'running', None), or
    ('did not start', why)."""
    display = port - VNC_BASE_PORT
    unit = UNIT.format(display)
    managed = unit_state(display) == 'running'
    if port_listening(port) and not managed:
        return 'did not start', (f'it is served by a screen Cutie does not manage; run '
                                 f'{SETUP_SCRIPT} once to hand it over')
    if not managed and path.exists(f'/tmp/.X11-unix/X{display}'):
        return 'did not start', f'screen :{display} is taken by an X server that is not a VNC screen'
    if managed and systemctl('is-enabled', '--quiet', unit).returncode == 0:
        return 'running', None
    result = systemctl('enable', '--now', unit)
    if result.returncode != 0:
        return 'did not start', failure(result, 'systemctl enable failed')
    if managed:
        return 'running', None
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if port_listening(port):
            return 'started', None
        time.sleep(0.25)
        if unit_state(display) in ('auto-restart', 'failed', 'dead'):
            break
    problem = last_log_lines(display) or f'not listening after {START_TIMEOUT_S:.0f} s'
    systemctl('disable', '--now', unit)  # not retried every few seconds forever
    return 'did not start', problem


def disable_screen(port: int) -> Result:
    """Disables and stops cutie-screen@N: ('stopped' or 'absent', None), or ('did not stop', why)."""
    display = port - VNC_BASE_PORT
    was_listening = port_listening(port)
    result = systemctl('disable', '--now', UNIT.format(display))
    if result.returncode != 0:
        return 'did not stop', failure(result, 'systemctl disable failed')
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while port_listening(port):
        if time.monotonic() > deadline:
            return 'did not stop', (f'it is still served by a screen Cutie does not manage; run '
                              f'{SETUP_SCRIPT} once to hand it over')
        time.sleep(0.25)
    return ('stopped' if was_listening else 'absent'), None


def sync(saved_ports: Iterable[int], removed_ports: Iterable[int] = ()) -> Dict[int, Result]:
    """Makes the screens match the list: every saved port enabled and listening, every removed
    port disabled and stopped. Doing it again changes nothing."""
    os.makedirs(path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_unit_file()
        results = {port: disable_screen(port) for port in removed_ports}
        results.update({port: enable_screen(port) for port in saved_ports})
        return results


def describe(port: int, result: Result) -> str:
    what, problem = result
    if problem:
        return f'Port {port} {what}: {problem}.'
    return {'started': f'Started port {port}, ready to connect.',
            'running': f'Port {port} is running.',
            'stopped': f'Stopped port {port}.',
            'absent': f'Port {port} was not running.'}[what]


def main() -> int:
    parser = argparse.ArgumentParser(description='Start the VNC screen of every saved annotator.')
    parser.add_argument('command', choices=['sync'])
    parser.add_argument('--tasks', required=True, help='the task root with annotators.json')
    args = parser.parse_args()
    results = sync(entry['port'] for entry in load_annotators(args.tasks))
    for port, result in sorted(results.items()):
        print(describe(port, result))
    return int(any(problem for _, problem in results.values()))


if __name__ == '__main__':
    sys.exit(main())
