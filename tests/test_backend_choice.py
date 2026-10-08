"""
Which mode a window starts in: --backend, else $CUTIEPLUS_BACKEND, else the task root's choice in
<task root>/.cutieplus/backend, else local.
"""
import os
import subprocess
import sys
from os import path

from cutieplus.paths import chosen_backend

REPO = path.dirname(path.dirname(path.abspath(__file__)))


def window_backend(arguments, environment=None) -> str:
    """The backend interactive_demo.py's arguments resolve to (parsed in a separate process: the
    script sets process-wide signal handlers when imported)."""
    env = {k: v for k, v in os.environ.items() if k != 'CUTIEPLUS_BACKEND'}
    env.update(environment or {})
    code = ('import sys; sys.argv = ["interactive_demo.py"] + sys.argv[1:]; '
            'import interactive_demo; print(interactive_demo.get_arguments().backend)')
    result = subprocess.run([sys.executable, '-c', code] + arguments, cwd=REPO, env=env,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_task_root_chooses_the_mode(tmp_path):
    root = str(tmp_path)
    assert chosen_backend(root) == 'local'
    assert chosen_backend(None) == 'local'
    os.makedirs(path.join(root, '.cutieplus'))
    with open(path.join(root, '.cutieplus', 'backend'), 'w') as f:
        f.write('shared\n')
    assert chosen_backend(root) == 'shared'
    with open(path.join(root, '.cutieplus', 'backend'), 'w') as f:
        f.write('gpu')
    assert chosen_backend(root) == 'local'  # not a mode: the original one, with a warning


def test_the_command_line_and_the_environment_come_first(tmp_path):
    root = str(tmp_path)
    os.makedirs(path.join(root, '.cutieplus'))
    with open(path.join(root, '.cutieplus', 'backend'), 'w') as f:
        f.write('shared')
    assert window_backend(['--tasks', root]) == 'shared'
    assert window_backend(['--tasks', root], {'CUTIEPLUS_BACKEND': 'local'}) == 'local'
    assert window_backend(['--tasks', root, '--backend', 'local']) == 'local'
    assert window_backend(['--tasks', str(tmp_path / 'other')]) == 'local'
