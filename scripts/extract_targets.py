"""
Writes the target files of every subset of the input (gui/task_input.py) from its seqinfo:

- <subset>/targets.json: {"<sequence>": {"category": ..., "description": ...}}, from the
  [Sequence] section of every <subset>/seqinfo/<sequence>_seqinfo.ini. Nothing else is copied:
  the interval in particular is not accurate there, and a task's interval is the whole sequence.
- <subset>/categories.json: the subset's categories, alphabetically, one {"name", "color"} per
  line. Each category has a color of its own, as far as can be from every other one in CIELAB,
  at a lightness and saturation that read on the light and the dark theme; the most frequent
  categories get the most distinct ones. A category has the same color in every subset. A later
  run keeps the colors already in the files, so a category never changes color, and keeps entries
  that no task uses any more.

A seqinfo that cannot be read, or whose task has no frames, is left out and reported; frames
without a seqinfo are reported too. The exit status is 1 when anything was reported.

    python -m scripts.extract_targets [--input workspace/cutie_tasks]
"""
import configparser
import json
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from os import path
from typing import Dict, List, Optional, Tuple

from gui.category_colors import assign_colors, categories_text, write_text
from gui.task_input import (CATEGORIES_FILE, DEFAULT_INPUT, SEQINFO_DIR, SEQINFO_SUFFIX,
                            TARGETS_FILE, InputError, Tab, Target, find_tabs, list_sequences,
                            read_categories)

SECTION = 'Sequence'


@dataclass
class Report:
    tasks: Dict[str, int] = field(default_factory=dict)  # tab -> tasks written
    categories: Dict[str, int] = field(default_factory=dict)  # category -> tasks
    unused: List[str] = field(default_factory=list)  # in categories.json, used by no task
    problems: List[str] = field(default_factory=list)


def read_seqinfo(file_path: str) -> Tuple[str, Target]:
    """(sequence, its target); raises ValueError naming what is wrong."""
    parser = configparser.ConfigParser(interpolation=None)  # a % in a description stays
    try:
        with open(file_path, encoding='utf-8') as stream:
            parser.read_file(stream)
    except (OSError, UnicodeDecodeError, configparser.Error) as exc:
        raise ValueError(f'cannot be read: {exc}') from exc
    if SECTION not in parser:
        raise ValueError(f'has no [{SECTION}] section')
    section = parser[SECTION]
    name = section.get('name', '').strip()
    expected = path.basename(file_path)[:-len(SEQINFO_SUFFIX)]
    if name != expected:
        raise ValueError(f'names the sequence {name!r}, not {expected!r}')
    target = Target(section.get('category', '').strip(), section.get('description', '').strip())
    missing = [key for key, value in (('category', target.category),
                                      ('description', target.description)) if not value]
    if missing:
        raise ValueError(f'has no {" or ".join(missing)}')
    return name, target


def tab_targets(tab: Tab, problems: List[str]) -> Optional[Dict[str, Target]]:
    """The tab's targets from its seqinfo, None when it has no seqinfo folder."""
    seqinfo_dir = path.join(tab.folder, SEQINFO_DIR)
    if not path.isdir(seqinfo_dir):
        return None
    with_frames = {sequence.sequence for sequence in list_sequences(tab, {})}
    targets = {}
    for file_name in sorted(os.listdir(seqinfo_dir)):
        if file_name.startswith('.') or not file_name.endswith(SEQINFO_SUFFIX):
            continue
        file_path = path.join(seqinfo_dir, file_name)
        try:
            sequence, target = read_seqinfo(file_path)
        except ValueError as exc:
            problems.append(f'{file_path} {exc}; left out.')
            continue
        if sequence not in with_frames:
            problems.append(f'{file_path}: sequence {sequence} has no frames; left out.')
            continue
        targets[sequence] = target
    without = sorted(with_frames - set(targets))
    if without:
        problems.append(f'Tab {tab.name}: {len(without)} sequence(s) have frames but no target: '
                        f'{", ".join(without[:10])}{", ..." if len(without) > 10 else ""}.')
    return targets


def kept_categories(tab: Tab) -> Dict[str, Optional[str]]:
    """The categories already in the subset's categories.json, name -> color (None for a name
    without one); raises ValueError when it cannot be read, so it is not overwritten."""
    if not path.isfile(path.join(tab.folder, CATEGORIES_FILE)):
        return {}
    try:
        return {category.name: category.color
                for category in read_categories(tab.folder, require_colors=False)}
    except InputError as exc:
        raise ValueError(f'{exc} Fix or remove it first.') from exc


def targets_text(targets: Dict[str, Target]) -> str:
    payload = {sequence: {'category': target.category, 'description': target.description}
               for sequence, target in sorted(targets.items())}
    return json.dumps(payload, indent=2, ensure_ascii=False) + '\n'


def extract(root: str) -> Report:
    """Writes every subset's targets.json and categories.json; raises InputError or ValueError
    (before writing anything) when the input or a categories.json cannot be read."""
    root = path.abspath(root)
    tabs = find_tabs(root)
    kept = {tab.name: kept_categories(tab) for tab in tabs}
    report = Report()
    written = []
    for tab in tabs:
        targets = tab_targets(tab, report.problems)
        if targets is None:
            report.problems.append(f'Tab {tab.name} has no {SEQINFO_DIR}/ folder; its '
                                   f'{TARGETS_FILE} is left as it is.')
            continue
        written.append((tab, targets))
    uses = Counter(target.category for _, targets in written for target in targets.values())
    by_use = sorted(uses, key=lambda name: (-uses[name], name))
    # one color per category in every subset: the first subset's, when two kept different ones;
    # a name listed without a color gets one too
    every_kept = {}
    for colors in kept.values():
        for name, color in colors.items():
            if color:
                every_kept.setdefault(name, color)
    uncolored = sorted({name for colors in kept.values() for name, color in colors.items()
                        if not color} - set(by_use))
    colors = assign_colors(by_use + uncolored, every_kept)
    for tab, targets in written:
        write_text(path.join(tab.folder, TARGETS_FILE), targets_text(targets))
        names = {target.category for target in targets.values()} | set(kept[tab.name])
        write_text(path.join(tab.folder, CATEGORIES_FILE),
                   categories_text({name: colors[name] for name in names}))
        report.tasks[tab.name] = len(targets)
        report.unused += [f'{tab.name}/{name}' for name in sorted(set(kept[tab.name]) - {
            target.category for target in targets.values()})]
    report.categories = dict(uses)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'the input root (default {DEFAULT_INPUT})')
    args = parser.parse_args(argv)
    try:
        report = extract(args.input)
    except (InputError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for tab, count in report.tasks.items():
        print(f'Tab {tab}: {count} tasks in {TARGETS_FILE}.')
    print(f'{len(report.categories)} categories in {CATEGORIES_FILE}: ' +
          ', '.join(f'{name} {count}' for name, count in
                    sorted(report.categories.items(), key=lambda item: (-item[1], item[0]))))
    if report.unused:
        print(f'In {CATEGORIES_FILE} but used by no task (kept): {", ".join(report.unused)}.')
    for problem in report.problems:
        print(f'Problem: {problem}')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
