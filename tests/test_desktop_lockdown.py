"""
A remote desktop runs Cutie and nothing else (deploy/openbox, deploy/desktop-session): openbox's
own defaults offer a terminal (a shell as the desktop account) from the desktop's menu and from
key bindings, so the desktop's openbox gets the release's configuration, with no key binding, no
binding on the desktop background, no action that runs a program or opens a menu, and an empty
menu named by its full path in the release. It still passes every click on to Cutie: openbox
keeps the clicks of a Frame binding, even inside the window, so there is none (checked with a real
X server, openbox and a window, clicked through the X server as a mouse would).
"""
import ctypes.util
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from os import path

import pytest

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
DEPLOY = path.join(ROOT, 'deploy')
RC = '{http://openbox.org/3.4/rc}'
MENU = '{http://openbox.org/3.4/menu}'


def read(*parts: str) -> str:
    with open(path.join(DEPLOY, *parts), encoding='utf-8') as f:
        return f.read()


def test_openbox_binds_no_key_and_no_program_or_menu():
    rc = ET.parse(path.join(DEPLOY, 'openbox', 'rc.xml')).getroot()
    assert list(rc.find(f'{RC}keyboard')) == []
    actions = {action.get('name') for action in rc.iter(f'{RC}action')}
    assert actions <= {'Focus', 'Raise', 'Move', 'Resize', 'ToggleMaximize', 'Close'}, actions
    contexts = {context.get('name') for context in rc.iter(f'{RC}context')}
    assert not contexts & {'Root', 'Desktop', 'Icon', 'Iconify'}, contexts
    assert 'Frame' not in contexts  # openbox would keep those clicks: Cutie would get none
    assert 'N' not in rc.find(f'{RC}theme/{RC}titleLayout').text  # no window icon: its menu
    files = [element.text for element in rc.iter(f'{RC}file')]
    assert files == ['/opt/cutieplus/current/deploy/openbox/menu.xml']


def test_the_only_menu_is_empty():
    menu = ET.parse(path.join(DEPLOY, 'openbox', 'menu.xml')).getroot()
    menus = list(menu.iter(f'{MENU}menu'))
    assert [entry.get('id') for entry in menus] == ['root-menu'] and list(menus[0]) == []


def test_the_desktop_starts_openbox_with_it_and_the_installer_copies_it():
    session = read('desktop-session')
    assert re.search(r'OPENBOX_RC="\$\(dirname "\$0"\)/openbox/rc\.xml"', session)
    assert '/usr/bin/openbox --config-file "$OPENBOX_RC" &' in session
    assert not re.search(r'openbox(?! --config-file)[^\n]*&', session)  # never without it
    assert 'cp -r "$REPO/deploy/openbox" "$target.new/deploy/"' in read('install.sh')


APP = """
from PySide6.QtWidgets import QApplication, QPushButton
app = QApplication([])
button = QPushButton('click me')
button.clicked.connect(lambda: print('CLICKED', flush=True))
button.setFixedSize(300, 200)
button.show()
app.exec()
"""
CLICK = """
import ctypes, sys, time
x11, xtst = ctypes.CDLL('libX11.so.6'), ctypes.CDLL('libXtst.so.6')
x11.XOpenDisplay.restype = ctypes.c_void_p
display = ctypes.c_void_p(x11.XOpenDisplay(None))
xtst.XTestFakeMotionEvent(display, -1, int(sys.argv[1]), int(sys.argv[2]), 0)
x11.XFlush(display)
time.sleep(0.3)
for pressed in (True, False):
    xtst.XTestFakeButtonEvent(display, 1, pressed, 0)
    x11.XFlush(display)
    time.sleep(0.1)
"""


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.slow
@pytest.mark.skipif(not (shutil.which('Xtigervnc') and shutil.which('openbox') and
                         shutil.which('xwininfo') and ctypes.util.find_library('Xtst')),
                    reason='needs Xtigervnc, openbox, xwininfo and libXtst')
def test_a_click_on_the_desktop_reaches_the_window(tmp_path):
    rc = tmp_path / 'rc.xml'
    rc.write_text(read('openbox', 'rc.xml').replace(
        '/opt/cutieplus/current/deploy/openbox/menu.xml', path.join(DEPLOY, 'openbox',
                                                                     'menu.xml')))
    (tmp_path / 'app.py').write_text(APP)
    (tmp_path / 'click.py').write_text(CLICK)
    number = next(n for n in range(90, 120) if not path.exists(f'/tmp/.X11-unix/X{n}'))
    env = dict(os.environ, DISPLAY=f':{number}', QT_QPA_PLATFORM='xcb')
    env.pop('PYTHONPATH', None)
    started = [subprocess.Popen(['Xtigervnc', f':{number}', '-localhost', '-rfbport',
                                 str(6000 + number), '-SecurityTypes', 'None', '-geometry',
                                 '1000x800', '-nolisten', 'tcp'], stderr=subprocess.DEVNULL)]
    try:
        assert wait_until(lambda: path.exists(f'/tmp/.X11-unix/X{number}'))
        started.append(subprocess.Popen(['openbox', '--config-file', str(rc)], env=env,
                                        stderr=subprocess.DEVNULL))
        time.sleep(1)
        log = open(tmp_path / 'app.log', 'w')
        started.append(subprocess.Popen([sys.executable, str(tmp_path / 'app.py')], env=env,
                                        stdout=log, stderr=subprocess.DEVNULL))

        def window():
            shown = subprocess.run(['xwininfo', '-name', 'app.py'], env=env,
                                   capture_output=True, text=True).stdout
            values = dict(re.findall(r'^\s*(Absolute upper-left [XY]|Width|Height):\s+(-?\d+)',
                                     shown, re.M))
            return {key: int(value) for key, value in values.items()} if len(values) == 4 \
                else None

        assert wait_until(lambda: window() is not None and window()['Width'] == 300)
        time.sleep(0.5)
        place = window()
        subprocess.run([sys.executable, str(tmp_path / 'click.py'),
                        str(place['Absolute upper-left X'] + 150),
                        str(place['Absolute upper-left Y'] + 100)], env=env, check=True)
        assert wait_until(lambda: 'CLICKED' in (tmp_path / 'app.log').read_text(), 5)
    finally:
        for process in reversed(started):
            process.terminate()
            process.wait(10)
