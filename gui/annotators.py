"""
Who labels from which screen. Each annotator connects to their own VNC port; port 5900+N is the
virtual screen :N, so a Cutie session finds its annotator from its own DISPLAY.

<task root>/annotators.json   {"format": 1, "annotators": [{"port": 5902, "name": "..."}, ...]}

The list is edited in the task browser (Manage annotators). It holds people's names, so it lives in
the task root with the labeling data and never in the code.
"""
import json
import os
import re
from os import path
from typing import Dict, List, Optional

ANNOTATORS_FILE = 'annotators.json'
VNC_BASE_PORT = 5900
MIN_PORT = VNC_BASE_PORT + 1  # screen :1
MAX_PORT = VNC_BASE_PORT + 99
MAX_NAME_LENGTH = 40

Entry = Dict[str, object]  # {'port': int, 'name': str}


def annotators_path(task_root: str) -> str:
    return path.join(task_root, ANNOTATORS_FILE)


def load_annotators(task_root: str) -> List[Entry]:
    """The saved list, sorted by port. A missing or unreadable file is an empty list."""
    try:
        with open(annotators_path(task_root), encoding='utf-8') as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return []
    records = data.get('annotators') if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    entries = [{'port': record['port'], 'name': record['name'].strip()} for record in records
               if isinstance(record, dict) and isinstance(record.get('name'), str)
               and isinstance(record.get('port'), int) and not isinstance(record['port'], bool)]
    return sorted(entries, key=lambda entry: entry['port'])


def check_annotators(entries: List[Entry]) -> List[str]:
    """What is wrong with a list, one message each; empty when it can be saved."""
    problems: List[str] = []
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


def save_annotators(task_root: str, entries: List[Entry]) -> List[Entry]:
    """Writes the list (sorted by port) and returns it as saved. Raises ValueError, writing
    nothing, when check_annotators finds problems."""
    cleaned = [{'port': entry['port'], 'name': str(entry['name']).strip()} for entry in entries]
    problems = check_annotators(cleaned)
    if problems:
        raise ValueError('\n'.join(problems))
    cleaned.sort(key=lambda entry: entry['port'])
    file_path = annotators_path(task_root)
    temporary = file_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump({'format': 1, 'annotators': cleaned}, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    os.replace(temporary, file_path)
    return cleaned


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
