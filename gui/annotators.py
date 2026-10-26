"""
Who labels from which screen. Each annotator connects to their own VNC port; port 5900+N is the
virtual screen :N, so a Cutie session finds its annotator from its own DISPLAY.

<task root>/annotators.json   {"format": 1, "annotators": [{"port": 5902, "name": "..."}, ...],
                               "inactive": ["...", ...], "admin_port": 5904}

inactive holds the names that were on the list and were removed or renamed: the task browser draws
them red in its Annotators column. A name added back is active again.

admin_port is the admin's screen: only a session on that port may change the list (Manage
annotators); every other one only sees it (Annotator list), and the admin's port cannot be removed.
Without admin_port every session may change the list, as before admins existed. It is set, or moved
to another port of the list, from a shell:

    python -m gui.annotators set-admin --tasks workspace/cutie_tasks --port 5904

The list is edited in the task browser (Manage annotators). It holds people's names, so it lives in
the task root with the labeling data and never in the code.
"""
import argparse
import json
import os
import re
import sys
from os import path
from typing import Dict, Iterable, List, Optional, Set

ANNOTATORS_FILE = 'annotators.json'
VNC_BASE_PORT = 5900
MIN_PORT = VNC_BASE_PORT + 1  # screen :1
MAX_PORT = VNC_BASE_PORT + 99
MAX_NAME_LENGTH = 40

Entry = Dict[str, object]  # {'port': int, 'name': str}


def annotators_path(task_root: str) -> str:
    return path.join(task_root, ANNOTATORS_FILE)


def read_file(task_root: str) -> dict:
    try:
        with open(annotators_path(task_root), encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_annotators(task_root: str) -> List[Entry]:
    """The saved list, sorted by port. A missing or unreadable file is an empty list."""
    records = read_file(task_root).get('annotators')
    if not isinstance(records, list):
        return []
    entries = [{'port': record['port'], 'name': record['name'].strip()} for record in records
               if isinstance(record, dict) and isinstance(record.get('name'), str)
               and isinstance(record.get('port'), int) and not isinstance(record['port'], bool)]
    return sorted(entries, key=lambda entry: entry['port'])


def load_inactive(task_root: str) -> Set[str]:
    """The names removed from the list (and not added back)."""
    names = read_file(task_root).get('inactive')
    if not isinstance(names, list):
        return set()
    return {name.strip() for name in names if isinstance(name, str) and name.strip()}


def load_admin_port(task_root: str) -> Optional[int]:
    port = read_file(task_root).get('admin_port')
    return port if isinstance(port, int) and not isinstance(port, bool) else None


def session_port(display: Optional[str] = None) -> Optional[int]:
    """This session's VNC port, from DISPLAY by default."""
    return display_port(os.environ.get('DISPLAY') if display is None else display)


def may_manage(task_root: str, display: Optional[str] = None) -> bool:
    """Whether this session may change the list: it is on the admin's port, or there is no admin."""
    admin_port = load_admin_port(task_root)
    return admin_port is None or session_port(display) == admin_port


def next_inactive(inactive: Iterable[str], old: List[Entry], new: List[Entry]) -> Set[str]:
    """The inactive names after the list changed from old to new: a name no longer on it becomes
    inactive, a name on it is active."""
    names = {str(entry['name']).strip() for entry in new}
    return (set(inactive) | {str(entry['name']).strip() for entry in old}) - names


def check_annotators(entries: List[Entry], admin_port: Optional[int] = None) -> List[str]:
    """What is wrong with a list, one message each; empty when it can be saved."""
    problems: List[str] = []
    if admin_port is not None and all(entry['port'] != admin_port for entry in entries):
        problems.append(f'Port {admin_port} is the admin and cannot be removed.')
    seen = set()
    for entry in entries:
        port, name = entry['port'], str(entry['name']).strip()
        if not isinstance(port, int) or not MIN_PORT <= port <= MAX_PORT:
            problems.append(f'Port {port} is not between {MIN_PORT} and {MAX_PORT}.')
        elif port in seen:
            problems.append(f'Port {port} is assigned twice.')
        seen.add(port)
        if not name:
            problems.append(f'Port {port} has no annotator name.')
        elif len(name) > MAX_NAME_LENGTH:
            problems.append(f'The name for port {port} is longer than {MAX_NAME_LENGTH} characters.')
        elif any(ord(character) < 32 for character in name):
            problems.append(f'The name for port {port} has a control character.')
    return list(dict.fromkeys(problems))


def save_annotators(task_root: str, entries: List[Entry],
                    inactive: Optional[Iterable[str]] = None) -> List[Entry]:
    """Writes the list (sorted by port) and the inactive names (by default the file's, less the
    saved names), keeping the file's admin, and returns the list as saved. Raises ValueError,
    writing nothing, when check_annotators finds problems (the admin's port missing among them)."""
    cleaned = [{'port': entry['port'], 'name': str(entry['name']).strip()} for entry in entries]
    admin_port = load_admin_port(task_root)
    problems = check_annotators(cleaned, admin_port)
    if problems:
        raise ValueError('\n'.join(problems))
    cleaned.sort(key=lambda entry: entry['port'])
    if inactive is None:
        inactive = load_inactive(task_root)
    inactive = sorted(set(inactive) - {entry['name'] for entry in cleaned})
    write_file(task_root, {'format': 1, 'annotators': cleaned, 'inactive': inactive,
                           'admin_port': admin_port})
    return cleaned


def set_admin(task_root: str, port: int) -> None:
    """Makes the annotator of port the admin. Raises ValueError when the port is not on the list."""
    data = read_file(task_root)
    if all(entry['port'] != port for entry in load_annotators(task_root)):
        raise ValueError(f'Port {port} is not on the annotator list.')
    data['admin_port'] = port
    write_file(task_root, data)


def write_file(task_root: str, data: dict) -> None:
    data = {key: value for key, value in data.items() if value is not None}
    file_path = annotators_path(task_root)
    temporary = file_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    os.replace(temporary, file_path)


def display_port(display: Optional[str]) -> Optional[int]:
    """The VNC port of an X display: ':4', ':4.0' and 'host:4' are port 5904. None for anything else,
    including screen :0 (there is no port 5900 here)."""
    match = re.fullmatch(r'[^:]*:(\d+)(?:\.\d+)?', display or '')
    if match is None:
        return None
    port = VNC_BASE_PORT + int(match.group(1))
    return port if MIN_PORT <= port <= MAX_PORT else None


def annotator_for_display(task_root: str, display: Optional[str] = None) -> Optional[str]:
    """The annotator assigned to this session's port (DISPLAY by default), or None."""
    port = display_port(os.environ.get('DISPLAY') if display is None else display)
    if port is None:
        return None
    for entry in load_annotators(task_root):
        if entry['port'] == port:
            return entry['name']
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description='Set the admin of the annotator list.')
    parser.add_argument('command', choices=['set-admin'])
    parser.add_argument('--tasks', required=True, help='the task root with annotators.json')
    parser.add_argument('--port', type=int, required=True, help="the admin's port, e.g. 5904")
    args = parser.parse_args()
    try:
        set_admin(args.tasks, args.port)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f'Port {args.port} is the admin.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
