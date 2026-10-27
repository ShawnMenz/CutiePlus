"""
scripts/extract_targets.py: every subset's targets.json from its seqinfo (category and
description only, never the interval), and its categories.json: each category once,
alphabetically, one per line, with a color of its own that a later run keeps and that is the same
in every subset.
"""
import itertools
import json
import os
from os import path

import numpy as np
import pytest

from gui.category_colors import MAX_LIGHTNESS, MIN_CHROMA, MIN_LIGHTNESS, lab_of
from gui.task_input import find_tabs, read_categories, read_tab
from scripts.extract_targets import extract, main

MIN_DISTANCE = 25.0  # CIELAB, between any two category colors


def add_task(subset: str, sequence: str, category: str = 'car',
             description: str = 'The red car.', name: str = None, frames: bool = True) -> None:
    if frames:
        os.makedirs(path.join(subset, 'visual', sequence), exist_ok=True)
        open(path.join(subset, 'visual', sequence, '000001.jpg'), 'w').close()
    os.makedirs(path.join(subset, 'seqinfo'), exist_ok=True)
    with open(path.join(subset, 'seqinfo', f'{sequence}_seqinfo.ini'), 'w',
              encoding='utf-8') as stream:
        stream.write(f'[Sequence]\nname={sequence if name is None else name}\ndataset=VTMOT\n'
                     f'category={category}\ndescription={description}\nproblem=PO\n'
                     'seqLength=120\nintervalStart=20\nintervalEnd=90\n')


def read_json(file_path: str):
    with open(file_path, encoding='utf-8') as stream:
        return json.load(stream)


def colors_of(subset: str) -> dict:
    return {entry['name']: entry['color']
            for entry in read_json(path.join(subset, 'categories.json'))}


def roots(tmp_path):
    # (the input root, its subset main)
    root = str(tmp_path)
    return root, path.join(root, 'main')


def test_targets_and_categories_come_from_seqinfo(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0002', 'cable car', ' The cable car, 50% hidden. ')
    add_task(subset, '0001', 'person', 'The person in black.')
    add_task(subset, '0003', 'person', 'The person in red.')
    report = extract(root)
    assert report.tasks == {'main': 3} and not report.problems
    assert read_json(path.join(subset, 'targets.json')) == {
        '0001': {'category': 'person', 'description': 'The person in black.'},
        '0002': {'category': 'cable car', 'description': 'The cable car, 50% hidden.'},
        '0003': {'category': 'person', 'description': 'The person in red.'},
    }
    with open(path.join(subset, 'targets.json'), encoding='utf-8') as stream:
        assert 'interval' not in stream.read()  # nor anything else of seqinfo
    with open(path.join(subset, 'categories.json'), encoding='utf-8') as stream:
        lines = stream.read().splitlines()
    assert lines[0] == '[' and lines[-1] == ']' and len(lines) == 4  # one category per line
    assert [json.loads(line.strip().rstrip(','))['name'] for line in lines[1:-1]] == \
        ['cable car', 'person']
    assert report.categories == {'person': 2, 'cable car': 1}
    # the input module reads them back
    contents = read_tab(find_tabs(root)[0])
    assert not contents.problems
    assert contents.sequences[1].description == 'The cable car, 50% hidden.'
    assert [category.name for category in read_categories(subset)] == ['cable car', 'person']


def test_category_colors_are_distinct_and_readable_on_both_themes(tmp_path):
    root, subset = roots(tmp_path)
    for index in range(24):
        add_task(subset, f'{index + 1:04d}', f'category {index:02d}')
    assert main(['--input', root]) == 0
    colors = colors_of(subset)
    assert len(colors) == 24 and len(set(colors.values())) == 24
    lab = lab_of(list(colors.values()))
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(lab, 2)) >= MIN_DISTANCE
    assert lab[:, 0].min() >= MIN_LIGHTNESS - 1 and lab[:, 0].max() <= MAX_LIGHTNESS + 1
    assert np.hypot(lab[:, 1], lab[:, 2]).min() >= MIN_CHROMA - 1


def test_a_later_run_keeps_the_colors_and_the_unused_categories(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0001', 'person')
    add_task(subset, '0002', 'car')
    extract(root)
    before = colors_of(subset)
    os.remove(path.join(subset, 'seqinfo', '0002_seqinfo.ini'))  # no task is a car any more
    add_task(subset, '0002', 'dog')
    report = extract(root)
    after = colors_of(subset)
    assert after['person'] == before['person'] and after['car'] == before['car']
    assert after['dog'] not in before.values()
    lab = lab_of(list(after.values()))
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(lab, 2)) >= MIN_DISTANCE
    assert report.unused == ['main/car']


def test_the_most_frequent_category_is_colored_first(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0001', 'swan')
    for index in range(2, 5):
        add_task(subset, f'{index:04d}', 'person')
    extract(root)
    lone = path.join(str(tmp_path), 'lone')
    add_task(path.join(lone, 'main'), '0001', 'person')
    extract(lone)
    assert colors_of(subset)['person'] == colors_of(path.join(lone, 'main'))['person']


def test_bad_seqinfo_is_reported_and_left_out(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0001')
    add_task(subset, '0002', name='0020')  # names another sequence
    add_task(subset, '0003', description='')
    add_task(subset, '0004', frames=False)  # no frames
    os.makedirs(path.join(subset, 'visual', '0005'))  # frames, no seqinfo
    os.makedirs(path.join(subset, 'visual', '0006'))
    with open(path.join(subset, 'seqinfo', '0006_seqinfo.ini'), 'w') as stream:
        stream.write('name=0006\n')  # no section
    report = extract(root)
    assert list(read_json(path.join(subset, 'targets.json'))) == ['0001']
    text = '\n'.join(report.problems)
    for expected in ("'0020', not '0002'", '0003_seqinfo.ini has no description',
                     'sequence 0004 has no frames', 'frames but no target: 0002, 0003, 0005, 0006',
                     '0006_seqinfo.ini cannot be read'):
        assert expected in text
    assert main(['--input', root]) == 1


def test_each_subset_lists_its_categories_in_the_same_colors(tmp_path):
    root = str(tmp_path)
    add_task(path.join(root, 'long'), '0001', 'truck')
    add_task(path.join(root, 'long'), '0002', 'bike')
    add_task(path.join(root, 'short'), '0001', 'bike')
    os.makedirs(path.join(root, 'annotation', '0001'))  # not a subset
    report = extract(root)
    assert report.tasks == {'long': 2, 'short': 1}
    assert read_json(path.join(root, 'long', 'targets.json'))['0001']['category'] == 'truck'
    long, short = colors_of(path.join(root, 'long')), colors_of(path.join(root, 'short'))
    assert list(long) == ['bike', 'truck'] and list(short) == ['bike']
    assert long['bike'] == short['bike']
    assert not path.exists(path.join(root, 'categories.json'))


def test_a_broken_category_file_is_not_overwritten(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0001')
    with open(path.join(subset, 'categories.json'), 'w') as stream:
        stream.write('[{"name": "car", "color": "green"}]')
    with pytest.raises(ValueError, match='Fix or remove it'):
        extract(root)
    with open(path.join(subset, 'categories.json')) as stream:
        assert stream.read() == '[{"name": "car", "color": "green"}]'
    assert not path.exists(path.join(subset, 'targets.json'))


def test_names_listed_without_a_color_get_one(tmp_path):
    root, subset = roots(tmp_path)
    add_task(subset, '0001', 'car')
    with open(path.join(subset, 'categories.json'), 'w') as stream:
        stream.write('["swan", {"name": "car", "color": "#00CF00"}]')
    extract(root)
    colors = colors_of(subset)
    assert colors['car'] == '#00CF00' and colors['swan'].startswith('#')  # kept, and picked
    assert colors['swan'] != colors['car']
