"""
What the Import window does on Save (gui/task_import.py): the parts of a subset found directly in
the input folder move into main/ (anything else stays); a folder elsewhere moves to
<input>/<tab> (a rename on one disk; across disks a copy, then the source is deleted, and a failed
copy leaves the source as it was); every row is checked first.
"""
import errno
import json
import os
import shutil
from os import path

import pytest

from gui import task_import
from gui.task_import import ImportRow, apply_import, plan_import
from gui.task_input import find_tabs


def subset(folder: str, *sequences: str, categories: bool = True) -> str:
    for sequence in sequences or ('0001', ):
        for modality in ('visual', 'thermal'):
            os.makedirs(path.join(folder, modality, sequence), exist_ok=True)
            for frame in ('000001.jpg', '000002.jpg'):
                open(path.join(folder, modality, sequence, frame), 'w').close()
    if categories:
        with open(path.join(folder, 'categories.json'), 'w', encoding='utf-8') as stream:
            json.dump([{'name': 'car', 'color': '#00CF00'}], stream)
    return folder


def moves_of(plans):
    assert all(plan.ok for plan in plans), [(plan.errors, plan.check) for plan in plans]
    return [plan.move for plan in plans if plan.move is not None]


def test_the_input_in_the_root_moves_into_main(tmp_path):
    root = str(tmp_path / 'cutie_tasks')
    subset(root, '0001', '0002')
    os.makedirs(path.join(root, 'annotation', '0001'))
    os.makedirs(path.join(root, 'seqinfo'))
    with open(path.join(root, 'targets.json'), 'w') as stream:
        stream.write('{}')
    open(path.join(root, 'notes.txt'), 'w').close()  # not part of a subset
    plans = plan_import(root, [ImportRow('main', path.join(root, 'main'))])
    [move] = moves_of(plans)
    assert move.parts == ['visual', 'thermal', 'annotation', 'seqinfo', 'targets.json',
                          'categories.json']
    assert 'move from' in plans[0].note and plans[0].check.sequences == 2
    assert any('notes.txt' in warning for warning in plans[0].check.warnings)
    said = []
    apply_import([move], said.append)
    assert sorted(os.listdir(root)) == ['main', 'notes.txt']
    assert sorted(os.listdir(path.join(root, 'main'))) == sorted(move.parts)
    assert [tab.name for tab in find_tabs(root)] == ['main'] and said[-1] == 'Done.'


def test_a_folder_elsewhere_moves_in_under_its_tab(tmp_path):
    root = str(tmp_path / 'cutie_tasks')
    source = subset(str(tmp_path / 'incoming' / 'long_set'), '0001')
    plans = plan_import(root, [ImportRow('long', source)])  # the input folder is made too
    [move] = moves_of(plans)
    assert (move.source, move.target) == (source, path.join(root, 'long'))
    apply_import([move], lambda _: None)
    assert not path.exists(source)
    assert [tab.name for tab in find_tabs(root)] == ['long']


def test_across_disks_the_folder_is_copied_then_deleted(tmp_path, monkeypatch):
    root = str(tmp_path / 'cutie_tasks')
    source = subset(str(tmp_path / 'usb' / 'night'), '0001', '0002')
    rename = os.rename

    def other_disk(src, dst):
        if src == source:
            raise OSError(errno.EXDEV, 'Invalid cross-device link')
        rename(src, dst)

    monkeypatch.setattr(os, 'rename', other_disk)
    monkeypatch.setattr(task_import, 'PROGRESS_EVERY', 3)
    said = []
    apply_import(moves_of(plan_import(root, [ImportRow('night', source)])), said.append)
    assert not path.exists(source)
    assert sorted(os.listdir(path.join(root, 'night', 'visual'))) == ['0001', '0002']
    assert any('Copying' in line and '9 / 9 files' in line for line in said)  # 8 frames + json


def test_a_failed_copy_leaves_the_source_and_no_half_subset(tmp_path, monkeypatch):
    root = str(tmp_path / 'cutie_tasks')
    source = subset(str(tmp_path / 'usb' / 'night'), '0001')
    monkeypatch.setattr(os, 'rename', lambda src, dst: (_ for _ in ()).throw(
        OSError(errno.EXDEV, 'Invalid cross-device link')))
    copied = []

    def copy_then_fail(src, dst):
        if copied:
            raise OSError(errno.ENOSPC, 'No space left on device')
        copied.append(src)
        shutil.copyfile(src, dst)

    monkeypatch.setattr(shutil, 'copy2', copy_then_fail)
    with pytest.raises(Exception, match='No space left'):
        apply_import(moves_of(plan_import(root, [ImportRow('night', source)])), lambda _: None)
    assert path.isdir(path.join(source, 'visual')) and not path.exists(path.join(root, 'night'))


def test_every_row_is_checked_before_save(tmp_path):
    root = str(tmp_path / 'cutie_tasks')
    subset(path.join(root, 'main'))
    os.makedirs(path.join(root, 'taken'))  # a folder, not a subset
    good = subset(str(tmp_path / 'good'))
    rows = {
        '': ImportRow('', good),
        'visual': ImportRow('visual', good),
        '.hidden': ImportRow('.hidden', good),
        'a/b': ImportRow('a/b', good),
        'main': ImportRow('main', good),
        'twice': ImportRow('twice', good),
        'missing': ImportRow('missing', str(tmp_path / 'nowhere')),
        'root': ImportRow('root', root),
        'inside': ImportRow('inside', path.join(root, 'main', 'visual')),
        'taken': ImportRow('taken', good),
        'nofolder': ImportRow('nofolder', ''),
    }
    plans = plan_import(root, list(rows.values()) + [ImportRow('twice', good)])
    errors = {name: ' '.join(plan.errors) for name, plan in zip(rows, plans)}
    assert errors[''] == 'Give the tab a name.'
    assert 'part of a subset' in errors['visual']
    assert 'cannot be a folder name' in errors['.hidden'] and 'cannot be' in errors['a/b']
    assert errors['main'] == 'main is a tab already.'
    assert errors['twice'] == 'Two rows are called twice.'
    assert errors['missing'].startswith('There is no folder')
    assert 'holds the input itself' in errors['root']
    assert 'inside a tab already' in errors['inside']
    assert errors['taken'].endswith('taken exists already.')
    assert errors['nofolder'] == 'Choose a folder.'
    assert not any(plan.ok for plan in plans)

    # a folder that lacks a mandatory part fails its check; an imported subset never moves
    lacking = subset(str(tmp_path / 'lacking'), categories=False)
    imported, new = plan_import(root, [ImportRow('main', path.join(root, 'main'), imported=True),
                                       ImportRow('lacking', lacking)])
    assert imported.ok and imported.move is None and imported.note == 'imported'
    assert not new.ok and any('categories.json is missing' in e for e in new.check.errors)


def test_save_picks_distinct_colors_for_names_listed_alone(tmp_path):
    import itertools
    import numpy as np
    from gui.category_colors import lab_of
    from gui.task_import import complete_colors
    from gui.task_input import read_categories
    root = str(tmp_path / 'cutie_tasks')
    subset(path.join(root, 'main'))  # car is #00CF00 there
    source = subset(str(tmp_path / 'incoming' / 'night'), categories=False)
    with open(path.join(source, 'categories.json'), 'w', encoding='utf-8') as stream:
        json.dump(['truck', 'car', {'name': 'bus', 'color': '#4800FF'}, 'person'], stream)
    [plan] = plan_import(root, [ImportRow('night', source)])
    assert plan.ok and plan.check.uncolored == 3
    assert plan.colors_folder == path.join(root, 'night')
    said = []
    apply_import([plan.move], said.append, [plan.colors_folder])
    assert any('Picked colors for 3 categories' in line for line in said)
    night = path.join(root, 'night')
    categories = {category.name: category.color for category in read_categories(night)}
    assert list(categories) == ['bus', 'car', 'person', 'truck']  # alphabetical, all colored
    assert categories['bus'] == '#4800FF'  # given: kept
    assert categories['car'] == '#00CF00'  # the same name in another subset: the same color
    lab = lab_of(list(categories.values()))
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(lab, 2)) >= 25
    with open(path.join(night, 'categories.json'), encoding='utf-8') as stream:
        assert len(stream.read().splitlines()) == 6  # one category per line
    assert complete_colors(night) == 0  # nothing left to pick

    # an imported subset that lists names only is colored by Save too
    with open(path.join(root, 'main', 'categories.json'), 'w', encoding='utf-8') as stream:
        json.dump(['car', 'dog'], stream)
    plan = plan_import(root, [ImportRow('main', path.join(root, 'main'), imported=True)])[0]
    assert plan.ok and plan.move is None and plan.colors_folder == path.join(root, 'main')
