"""
The tool's input: the raw frames to label, by subset (a tab of the task browser) and sequence.

    <input root>/                        workspace/cutie_tasks by default (DEFAULT_INPUT)
    ├── main/                            a subset: its folder's name is the tab's
    │   ├── visual/<sequence>/<frame>    mandatory
    │   ├── thermal/<sequence>/<frame>   mandatory
    │   ├── categories.json              mandatory: [{"name": ..., "color": "#RRGGBB"}, ...]; names
    │   │                                alone (["car", ...]) get colors on Import
    │   ├── targets.json                 optional: {"<sequence>": {"category", "description"}}
    │   ├── annotation/<sequence>/<frame>.png   optional: pregenerated masks (any value > 0)
    │   └── seqinfo/<sequence>_seqinfo.ini      optional: scripts/extract_targets.py reads it
    └── long/ ...                        another subset, another tab

The task browser's Import window (gui/import_dialog.py, gui/task_import.py) puts subsets there; it
also moves the parts of a subset found directly in the root into main/. A subset's sequences are
the sequence folders under any of its modalities; a sequence opens on its first modality in
MODALITIES. Frames are matched across modalities by name, so 000001.png and 000001.jpg are the
same frame. A sequence without a target has no category (shown as NULL_CATEGORY) and an empty
description. Nothing here writes to the input.
"""
import json
import os
import re
from dataclasses import dataclass, field
from os import path
from typing import Dict, List, Optional

MODALITIES = ('visual', 'thermal')  # in this order: a sequence is labeled on the first it has
MAIN_TAB = 'main'  # the subset that the parts found directly in the root are moved into
TARGETS_FILE = 'targets.json'
CATEGORIES_FILE = 'categories.json'
ANNOTATION_DIR = 'annotation'  # pregenerated masks
SEQINFO_DIR = 'seqinfo'  # <subset>/seqinfo/<sequence>_seqinfo.ini, read by scripts/extract_targets.py
SEQINFO_SUFFIX = '_seqinfo.ini'
# what a subset folder holds; a subset needs the modalities and the categories
INPUT_PARTS = (*MODALITIES, ANNOTATION_DIR, SEQINFO_DIR, TARGETS_FILE, CATEGORIES_FILE)
NULL_CATEGORY = 'null'  # shown for a sequence without a category
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
COLOR_PATTERN = re.compile(r'#[0-9A-Fa-f]{6}')
DEFAULT_INPUT = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'workspace',
                          'cutie_tasks')


class InputError(Exception):
    """The input cannot be read as it is laid out; the message names the files or folders."""


@dataclass(frozen=True)
class Target:
    category: Optional[str] = None
    description: str = ''


@dataclass(frozen=True)
class Category:
    name: str
    color: Optional[str]  # #RRGGBB; None before Import has picked one


@dataclass
class Tab:
    name: str
    folder: str  # the subset folder
    modality_dirs: Dict[str, str]  # modality -> <folder>/<modality>, only the ones present


@dataclass
class SequenceInput:
    tab: str
    sequence: str
    frame_dirs: Dict[str, str]  # modality -> the folder of its frames, only the ones present
    anchor_modality: str  # the one labeled first
    category: Optional[str] = None
    description: str = ''
    annotation_dir: Optional[str] = None  # its pregenerated masks, None without
    folder: str = ''  # its subset's folder (categories.json)


@dataclass
class TabContents:
    tab: Tab
    sequences: List[SequenceInput]
    # what could not be read, e.g. a broken targets.json, for the browser's messages
    problems: List[str] = field(default_factory=list)


@dataclass
class SubsetCheck:
    """What a subset folder holds, for the Import window: errors stop the import."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sequences: int = 0
    with_targets: int = 0
    with_masks: int = 0
    categories: int = 0
    uncolored: int = 0  # categories listed without a color: Import picks them
    has_targets: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def frame_files(directory: str) -> Dict[str, str]:
    # frame name -> file name, in frame order
    files = sorted(f for f in os.listdir(directory)
                   if path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
    return {path.splitext(f)[0]: f for f in files}


def _folders(directory: str) -> List[str]:
    # the visible subfolders, sorted
    with os.scandir(directory) as entries:
        return sorted(entry.name for entry in entries
                      if not entry.name.startswith('.') and entry.is_dir())


def _modality_dirs(folder: str) -> Dict[str, str]:
    return {modality: path.join(folder, modality) for modality in MODALITIES
            if path.isdir(path.join(folder, modality))}


def find_tabs(root: str) -> List[Tab]:
    """The subsets of an input root: its subfolders holding a modality, main first, then the
    others by name."""
    root = path.abspath(root)
    if not path.isdir(root):
        raise InputError(f'There is no input folder {root}.')
    tabs = []
    for name in _folders(root):
        if name in INPUT_PARTS:
            continue
        modality_dirs = _modality_dirs(path.join(root, name))
        if modality_dirs:
            tabs.append(Tab(name, path.join(root, name), modality_dirs))
    tabs.sort(key=lambda tab: (tab.name != MAIN_TAB, tab.name))
    return tabs


def unorganized_parts(root: str) -> List[str]:
    """The parts of a subset that sit directly in the root (Import moves them into main/)."""
    return [name for name in INPUT_PARTS if path.exists(path.join(root, name))]


def read_targets(folder: str) -> Dict[str, Target]:
    """Sequence -> its target, from the subset's targets.json; empty without the file."""
    file_path = path.join(folder, TARGETS_FILE)
    if not path.isfile(file_path):
        return {}
    try:
        with open(file_path, encoding='utf-8') as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InputError(f'{file_path} cannot be read: {exc}') from exc
    if not isinstance(payload, dict):
        raise InputError(f'{file_path} is not an object keyed by sequence.')
    targets = {}
    for sequence, entry in payload.items():
        if not isinstance(entry, dict):
            raise InputError(f'{file_path}: the entry of {sequence} is not an object.')
        category = str(entry.get('category') or '').strip()
        targets[str(sequence)] = Target(category or None,
                                        str(entry.get('description') or '').strip())
    return targets


def read_categories(folder: str, require_colors: bool = True) -> List[Category]:
    """The subset's categories.json: [{"name": ..., "color": "#RRGGBB"}, ...], names unique. With
    require_colors False, a name may come alone, as a string or without "color" (color None)."""
    file_path = path.join(folder, CATEGORIES_FILE)
    if not path.isfile(file_path):
        raise InputError(f'{CATEGORIES_FILE} is missing in {folder}.')
    try:
        with open(file_path, encoding='utf-8') as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InputError(f'{file_path} cannot be read: {exc}') from exc
    if not isinstance(payload, list):
        raise InputError(f'{file_path} is not a list of categories.')
    categories = []
    for entry in payload:
        if isinstance(entry, str):
            entry = {'name': entry}
        name = entry.get('name') if isinstance(entry, dict) else None
        color = entry.get('color') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise InputError(f'{file_path}: {entry!r} has no name.')
        if color is None and not require_colors:
            categories.append(Category(name.strip(), None))
            continue
        if color is None:
            raise InputError(f'{file_path}: {name} has no color; Import picks one.')
        if not isinstance(color, str) or not COLOR_PATTERN.fullmatch(color):
            raise InputError(f'{file_path}: the color of {name} is not #RRGGBB.')
        categories.append(Category(name.strip(), color.upper()))
    names = [category.name for category in categories]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InputError(f'{file_path} lists {", ".join(duplicates)} more than once.')
    return categories


def _sequence(tab: Tab, name: str, targets: Dict[str, Target]) -> Optional[SequenceInput]:
    frame_dirs = {modality: path.join(directory, name)
                  for modality, directory in tab.modality_dirs.items()
                  if path.isdir(path.join(directory, name))}
    if not frame_dirs:
        return None
    annotation_dir = path.join(tab.folder, ANNOTATION_DIR, name)
    target = targets.get(name, Target())
    return SequenceInput(tab.name, name, frame_dirs, next(iter(frame_dirs)), target.category,
                         target.description,
                         annotation_dir if path.isdir(annotation_dir) else None, tab.folder)


def list_sequences(tab: Tab, targets: Dict[str, Target]) -> List[SequenceInput]:
    """The tab's sequences, by name: every sequence folder under any of its modalities."""
    names = sorted({name for directory in tab.modality_dirs.values()
                    for name in _folders(directory)})
    return [_sequence(tab, name, targets) for name in names]


def read_tab(tab: Tab) -> TabContents:
    """The tab's sequences with their targets. A targets.json that cannot be read leaves the
    descriptions empty and is reported, as is a target of a sequence with no frames."""
    problems = []
    try:
        targets = read_targets(tab.folder)
    except InputError as exc:
        targets = {}
        problems.append(f'{exc} The descriptions of tab {tab.name} start empty.')
    sequences = list_sequences(tab, targets)
    unknown = sorted(set(targets) - {sequence.sequence for sequence in sequences})
    if unknown:
        problems.append(f'{path.join(tab.folder, TARGETS_FILE)} describes {len(unknown)} '
                        f'sequence(s) without frames: {", ".join(unknown[:10])}'
                        f'{", ..." if len(unknown) > 10 else ""}.')
    return TabContents(tab, sequences, problems)


def read_sequence(root: str, tab_name: str, sequence: str) -> SequenceInput:
    """One sequence of the input, as the labeling window opens it; a broken targets.json leaves
    its description empty (the browser reports it)."""
    folder = path.join(path.abspath(root), tab_name)
    tab = Tab(tab_name, folder, _modality_dirs(folder))
    try:
        targets = read_targets(folder)
    except InputError:
        targets = {}
    found = _sequence(tab, sequence, targets)
    if found is None:
        raise InputError(f'There are no frames of sequence {sequence} in {folder}.')
    return found


def pregenerated_masks(sequence: SequenceInput) -> Dict[str, str]:
    """Frame name -> the file of its pregenerated mask."""
    if sequence.annotation_dir is None:
        return {}
    return {name: path.join(sequence.annotation_dir, file_name)
            for name, file_name in frame_files(sequence.annotation_dir).items()}


def check_subset(folder: str) -> SubsetCheck:
    """Whether a folder can be imported as a subset, and what it holds."""
    check = SubsetCheck()
    if not path.isdir(folder):
        check.errors.append(f'There is no folder {folder}.')
        return check
    modality_dirs = _modality_dirs(folder)
    for modality in MODALITIES:
        if modality not in modality_dirs:
            check.errors.append(f'{modality}/ is missing.')
    categories = []
    try:
        categories = read_categories(folder, require_colors=False)
        check.categories = len(categories)
        check.uncolored = sum(category.color is None for category in categories)
    except InputError as exc:
        check.errors.append(str(exc))
    try:
        targets = read_targets(folder)
        check.has_targets = path.isfile(path.join(folder, TARGETS_FILE))
    except InputError as exc:
        targets = {}
        check.errors.append(str(exc))
    names = {modality: set(_folders(directory)) for modality, directory in modality_dirs.items()}
    sequences = set().union(*names.values()) if names else set()
    check.sequences = len(sequences)
    check.with_targets = len(sequences & set(targets))
    annotation = path.join(folder, ANNOTATION_DIR)
    if path.isdir(annotation):
        check.with_masks = len(sequences & set(_folders(annotation)))
    if len(names) == len(MODALITIES):
        single = sorted(sequences - set.intersection(*names.values()))
        if single:
            check.warnings.append(f'{len(single)} sequence(s) have only one modality: '
                                  f'{", ".join(single[:5])}{", ..." if len(single) > 5 else ""}.')
    unknown = sorted(set(targets) - sequences)
    if unknown:
        check.warnings.append(f'{TARGETS_FILE} describes {len(unknown)} sequence(s) without '
                              f'frames: {", ".join(unknown[:5])}{", ..." if len(unknown) > 5 else ""}.')
    listed = {category.name for category in categories}
    unlisted = sorted({target.category for target in targets.values()
                       if target.category and target.category not in listed})
    if categories and unlisted:
        check.warnings.append(f'Categories in {TARGETS_FILE} but not in {CATEGORIES_FILE}: '
                              f'{", ".join(unlisted)}.')
    return check
