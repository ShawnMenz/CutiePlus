"""
What the Import window (gui/import_dialog.py) checks and does on Save, without Qt.

Every subset of the input is a folder <input root>/<tab>/ (gui/task_input.py). A row of the window
names a tab and a folder, and Save puts the folder there:
- the folder is <root>/<tab> already (e.g. a subset imported before): nothing moves;
- the folder is <root>/<tab>, which does not exist yet, while the parts of a subset (visual/,
  thermal/, ...) sit directly in the root: those parts move into it (anything else stays);
- any other folder moves to <root>/<tab>: a rename on the same disk, else a copy, after which the
  source is deleted (a failed copy is removed, and the source stays).
Then every subset whose categories.json lists a name without a color gets one (complete_colors),
new subsets and imported ones alike, and the file is written again with them.
"""
import errno
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from os import path
from typing import Callable, Iterable, List, Optional

from gui.category_colors import assign_colors, categories_text, write_text
from gui.task_input import (CATEGORIES_FILE, INPUT_PARTS, InputError, SubsetCheck, check_subset,
                            find_tabs, read_categories, read_targets, unorganized_parts)

Progress = Callable[[str], None]
PROGRESS_EVERY = 200  # files between two progress lines of a copy


@dataclass
class ImportRow:
    tab: str
    folder: str
    imported: bool = False  # a subset that is already a tab: checked again, never moved


@dataclass
class Move:
    source: str  # the folder, or the root whose parts move
    target: str  # <root>/<tab>
    parts: Optional[List[str]] = None  # the root's parts that move; None: the whole folder

    def describe(self) -> str:
        if self.parts is not None:
            return f'{", ".join(self.parts)} → {self.target}'
        return f'{self.source} → {self.target}'


@dataclass
class RowPlan:
    row: ImportRow
    check: Optional[SubsetCheck] = None
    errors: List[str] = field(default_factory=list)
    move: Optional[Move] = None
    note: str = ''  # what Save does with it, in words
    colors_folder: Optional[str] = None  # its folder after Save, when colors are to be picked

    @property
    def ok(self) -> bool:
        return not self.errors and self.check is not None and self.check.ok


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def tab_name_errors(name: str) -> List[str]:
    if not name:
        return ['Give the tab a name.']
    if name.startswith('.') or '/' in name or '\\' in name:
        return [f'{name!r} cannot be a folder name (no "/", no leading ".").']
    if name in INPUT_PARTS:
        return [f'{name!r} is the name of a part of a subset; choose another.']
    return []


def plan_import(root: str, rows: List[ImportRow]) -> List[RowPlan]:
    """Each row's check and move, or why it cannot be imported."""
    root = path.abspath(root)
    imported = {tab.name: tab.folder for tab in find_tabs(root)} if path.isdir(root) else {}
    names = [row.tab.strip() for row in rows]
    plans = []
    for row, name in zip(rows, names):
        plan = RowPlan(row)
        plans.append(plan)
        folder = path.abspath(path.expanduser(row.folder.strip())) if row.folder.strip() else ''
        if row.imported:
            plan.check = check_subset(folder)
            plan.note = 'imported'
            continue
        plan.errors += tab_name_errors(name)
        if names.count(name) > 1:
            plan.errors.append(f'Two rows are called {name}.')
        if name in imported:
            plan.errors.append(f'{name} is a tab already.')
        if not folder:
            plan.errors.append('Choose a folder.')
        if plan.errors:
            continue
        target = path.join(root, name)
        if folder == target and not path.exists(target) and unorganized_parts(root):
            plan.check = check_subset(root)
            plan.move = Move(root, target, unorganized_parts(root))
            plan.note = f'{", ".join(plan.move.parts)} move from {root} into {name}/ on Save'
            stray = sorted(entry for entry in os.listdir(root) if not entry.startswith('.')
                           and entry not in INPUT_PARTS and entry not in imported)
            if stray:
                plan.check.warnings.append(f'Not part of a subset, left in {root}: '
                                           f'{", ".join(stray)}.')
            continue
        if folder == target:
            plan.check = check_subset(folder)
            plan.note = 'already in place'
            continue
        if not path.isdir(folder):
            plan.errors.append(f'There is no folder {folder}.')
        elif folder == root or _inside(root, folder):
            plan.errors.append(f'{folder} holds the input itself; choose a subset folder.')
        elif any(_inside(folder, tab_folder) for tab_folder in imported.values()):
            plan.errors.append(f'{folder} is inside a tab already.')
        elif path.exists(target):
            plan.errors.append(f'{target} exists already.')
        if plan.errors:
            continue
        plan.check = check_subset(folder)
        plan.move = Move(folder, target)
        plan.note = f'moves from {folder} into {name}/ on Save'
    for plan in plans:
        if plan.ok and plan.check.uncolored:
            plan.colors_folder = plan.move.target if plan.move is not None else \
                path.abspath(path.expanduser(plan.row.folder.strip()))
    return plans


def complete_colors(folder: str) -> int:
    """Gives every category of the subset's categories.json listed without a color one: the
    color the same name has in another subset of the input, when no other category here has it,
    else a new one far from all the colors of this subset (the most frequent in its targets.json
    first). Writes the file again, one category per line; returns how many colors it picked."""
    folder = path.abspath(folder)
    categories = read_categories(folder, require_colors=False)
    missing = [category.name for category in categories if category.color is None]
    if not missing:
        return 0
    colors = {category.name: category.color for category in categories if category.color}
    elsewhere = {}
    for tab in find_tabs(path.dirname(folder)):
        if path.abspath(tab.folder) == folder:
            continue
        try:
            others = read_categories(tab.folder, require_colors=False)
        except InputError:
            continue
        for category in others:
            if category.color:
                elsewhere.setdefault(category.name, category.color)
    for name in missing:
        if elsewhere.get(name) and elsewhere[name] not in colors.values():
            colors[name] = elsewhere[name]
    try:
        uses = Counter(target.category for target in read_targets(folder).values())
    except InputError:
        uses = Counter()
    new = sorted((name for name in missing if name not in colors), key=lambda n: (-uses[n], n))
    colors = assign_colors(new, colors)
    write_text(path.join(folder, CATEGORIES_FILE), categories_text(colors))
    return len(missing)


def _count_files(folder: str) -> int:
    return sum(len(files) for _, _, files in os.walk(folder))


def _move_folder(source: str, target: str, progress: Progress) -> None:
    try:
        os.rename(source, target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    # another disk: copy, then delete the source once the copy is complete
    total = _count_files(source)
    copied = 0

    def copy(source_file: str, target_file: str) -> None:
        nonlocal copied
        shutil.copy2(source_file, target_file)
        copied += 1
        if copied % PROGRESS_EVERY == 0 or copied == total:
            progress(f'Copying {source}: {copied} / {total} files')

    try:
        shutil.copytree(source, target, copy_function=copy)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    progress(f'Deleting {source}')
    shutil.rmtree(source)


def apply_import(moves: List[Move], progress: Progress, color_folders: Iterable[str] = ()) -> None:
    """Makes the moves in order, then picks the missing category colors of color_folders; raises
    on the first step that fails (the earlier ones stay)."""
    for move in moves:
        progress(f'Moving {move.describe()}')
        os.makedirs(path.dirname(move.target), exist_ok=True)  # the input folder, the first time
        if move.parts is not None:
            os.makedirs(move.target)
            for part in move.parts:
                os.rename(path.join(move.source, part), path.join(move.target, part))
        else:
            _move_folder(move.source, move.target, progress)
    for folder in color_folders:
        picked = complete_colors(folder)
        progress(f'Picked colors for {picked} categories in {path.join(folder, CATEGORIES_FILE)}')
    progress('Done.')
