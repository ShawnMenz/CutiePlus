"""
The tool's input (gui/task_input.py): a tab per subfolder holding the modalities (parts found
directly in the root are not a tab until Import moves them into main/), sequences from the
folders under any modality, frames paired by name across modalities, each subset's targets.json
(no target: category None, empty description), its mandatory categories.json, and its
pregenerated masks (annotation/). check_subset is what the Import window shows.
"""
import json
import os
from os import path

import pytest

from gui.task_input import (Category, InputError, Target, check_subset, find_tabs, frame_files,
                            pregenerated_masks, read_categories, read_sequence, read_tab,
                            read_targets, unorganized_parts)

CATEGORIES = [{'name': 'car', 'color': '#00CF00'}, {'name': 'person', 'color': '#4800ff'}]


def add_frames(folder: str, *names: str) -> None:
    # the listing reads names only, so empty files stand in for the frames
    os.makedirs(folder, exist_ok=True)
    for name in names:
        open(path.join(folder, name), 'w').close()


def write_json(folder: str, name: str, payload) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(path.join(folder, name), 'w', encoding='utf-8') as stream:
        stream.write(payload if isinstance(payload, str) else json.dumps(payload))


def subset(folder: str, *sequences: str, modalities=('visual', 'thermal')) -> str:
    for sequence in sequences:
        for modality in modalities:
            add_frames(path.join(folder, modality, sequence), '000001.jpg', '000002.jpg')
    write_json(folder, 'categories.json', CATEGORIES)
    return folder


def sequences_of(tab):
    return {sequence.sequence: sequence for sequence in read_tab(tab).sequences}


def test_subfolders_holding_the_modalities_are_tabs_main_first(tmp_path):
    root = str(tmp_path)
    subset(path.join(root, 'long'), '0002')
    subset(path.join(root, 'main'), '0001')
    add_frames(path.join(root, 'short', 'thermal', '0003'), '000001.png')
    add_frames(path.join(root, 'notes'), 'readme.png')  # no modality: not a tab
    add_frames(path.join(root, '.partial', 'visual', '0004'), '000001.png')  # hidden
    add_frames(path.join(root, 'visual', '0005'), '000001.png')  # in the root: not a tab yet
    tabs = find_tabs(root)
    assert [tab.name for tab in tabs] == ['main', 'long', 'short']
    assert tabs[0].folder == path.join(root, 'main')
    assert list(sequences_of(tabs[1])) == ['0002']
    assert tabs[2].modality_dirs == {'thermal': path.join(root, 'short', 'thermal')}


def test_the_parts_of_a_subset_in_the_root_wait_for_import(tmp_path):
    root = str(tmp_path)
    subset(root, '0001')
    add_frames(path.join(root, 'annotation', '0001'), '000001.png')
    add_frames(root, 'notes.png')
    assert find_tabs(root) == []
    assert unorganized_parts(root) == ['visual', 'thermal', 'annotation', 'categories.json']


def test_a_missing_input_folder_is_a_clear_error(tmp_path):
    with pytest.raises(InputError, match='no input folder'):
        find_tabs(str(tmp_path / 'absent'))


def test_a_sequence_with_one_modality_is_labeled_on_it(tmp_path):
    main = path.join(str(tmp_path), 'main')
    subset(main, '0001', modalities=('visual', ))
    subset(main, '0002', modalities=('thermal', ))
    add_frames(path.join(main, 'visual', '.0003-copying'), '000001.png')  # hidden: skipped
    sequences = sequences_of(find_tabs(str(tmp_path))[0])
    assert list(sequences) == ['0001', '0002']
    assert list(sequences['0001'].frame_dirs) == ['visual']
    assert sequences['0002'].anchor_modality == 'thermal'


def test_frames_pair_by_name_whatever_their_extension(tmp_path):
    visual, thermal = str(tmp_path / 'visual'), str(tmp_path / 'thermal')
    add_frames(visual, '000002.png', '000001.png', 'notes.txt', 'thumbs.db')
    add_frames(thermal, '000001.jpg', '000002.JPEG')
    assert frame_files(visual) == {'000001': '000001.png', '000002': '000002.png'}
    assert list(frame_files(thermal)) == list(frame_files(visual))


def test_without_a_target_the_category_is_none_and_the_description_empty(tmp_path):
    main = subset(path.join(str(tmp_path), 'main'), '0001', '0002')
    assert read_targets(main) == {}  # no targets.json: nothing to report
    first = sequences_of(find_tabs(str(tmp_path))[0])['0001']
    assert (first.category, first.description) == (None, '')
    write_json(main, 'targets.json', {'0001': {'category': ' person ', 'description': None},
                                      '0002': {'category': '', 'description': ' The car. '}})
    assert read_targets(main) == {'0001': Target('person', ''), '0002': Target(None, 'The car.')}


def test_a_broken_targets_file_is_reported_and_leaves_the_texts_empty(tmp_path):
    main = subset(path.join(str(tmp_path), 'main'), '0001')
    write_json(main, 'targets.json', '{"0001": {"category": "car",')
    contents = read_tab(find_tabs(str(tmp_path))[0])
    assert contents.sequences[0].description == ''
    [problem] = contents.problems
    assert path.join(main, 'targets.json') in problem and 'start empty' in problem
    assert read_sequence(str(tmp_path), 'main', '0001').description == ''  # opens anyway
    write_json(main, 'targets.json', ['not', 'an', 'object'])
    with pytest.raises(InputError, match='keyed by sequence'):
        read_targets(main)
    write_json(main, 'targets.json', {'0001': 'car'})
    with pytest.raises(InputError, match='entry of 0001'):
        read_targets(main)


def test_targets_of_sequences_without_frames_are_reported(tmp_path):
    main = subset(path.join(str(tmp_path), 'main'), '0001')
    write_json(main, 'targets.json', {'0001': {'description': 'here'}, '0099': {'description': 'x'}})
    contents = read_tab(find_tabs(str(tmp_path))[0])
    assert contents.sequences[0].description == 'here'
    assert any('0099' in problem for problem in contents.problems)


def test_categories_are_a_list_of_names_and_colors(tmp_path):
    folder = str(tmp_path)
    with pytest.raises(InputError, match='categories.json is missing'):
        read_categories(folder)
    write_json(folder, 'categories.json', CATEGORIES)
    assert read_categories(folder) == [Category('car', '#00CF00'), Category('person', '#4800FF')]
    for payload, problem in (({'car': '#00CF00'}, 'not a list'),
                             ([{'name': 'car', 'color': 'green'}], 'not #RRGGBB'),
                             ([{'color': '#00CF00'}], 'has no name'),
                             (CATEGORIES + CATEGORIES[:1], 'car more than once'),
                             ('[{"name": ', 'cannot be read')):
        write_json(folder, 'categories.json', payload)
        with pytest.raises(InputError, match=problem):
            read_categories(folder)


def test_pregenerated_masks_come_from_annotation(tmp_path):
    main = subset(path.join(str(tmp_path), 'main'), '0001', '0002')
    add_frames(path.join(main, 'annotation', '0001'), '000002.png', '000001.png')
    first = read_sequence(str(tmp_path), 'main', '0001')
    assert first.annotation_dir == path.join(main, 'annotation', '0001')
    assert pregenerated_masks(first) == {
        '000001': path.join(main, 'annotation', '0001', '000001.png'),
        '000002': path.join(main, 'annotation', '0001', '000002.png')}
    second = read_sequence(str(tmp_path), 'main', '0002')
    assert second.annotation_dir is None and pregenerated_masks(second) == {}
    with pytest.raises(InputError, match='no frames of sequence 0003'):
        read_sequence(str(tmp_path), 'main', '0003')


def test_check_subset_names_what_is_missing_and_counts_the_rest(tmp_path):
    folder = subset(str(tmp_path / 'ok'), '0001', '0002')
    add_frames(path.join(folder, 'visual', '0003'), '000001.jpg')  # visual only
    add_frames(path.join(folder, 'annotation', '0001'), '000001.png')
    write_json(folder, 'targets.json', {'0001': {'category': 'car'}, '0002': {'category': 'boat'}})
    check = check_subset(folder)
    assert check.ok and check.has_targets
    assert (check.sequences, check.with_targets, check.with_masks, check.categories) == (3, 2, 1, 2)
    assert any('only one modality: 0003' in warning for warning in check.warnings)
    assert any('not in categories.json: boat' in warning for warning in check.warnings)

    bare = str(tmp_path / 'bare')
    add_frames(path.join(bare, 'visual', '0001'), '000001.jpg')
    check = check_subset(bare)
    assert not check.ok and not check.has_targets
    assert 'thermal/ is missing.' in check.errors
    assert any('categories.json is missing' in error for error in check.errors)
    assert check_subset(str(tmp_path / 'absent')).errors[0].startswith('There is no folder')


def test_a_category_file_may_list_names_only_until_import_colors_them(tmp_path):
    folder = str(tmp_path)
    write_json(folder, 'categories.json', ['car', {'name': 'person'}, CATEGORIES[0] | {'name': 'bus'}])
    assert read_categories(folder, require_colors=False) == [
        Category('car', None), Category('person', None), Category('bus', '#00CF00')]
    with pytest.raises(InputError, match='car has no color; Import picks one'):
        read_categories(folder)  # the labeling window needs the colors
    main = subset(path.join(folder, 'main'), '0001')
    write_json(main, 'categories.json', ['car', 'person'])
    check = check_subset(main)
    assert check.ok and (check.categories, check.uncolored) == (2, 2)
