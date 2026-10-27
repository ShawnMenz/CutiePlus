"""
The Import window (gui/import_dialog.py): with the input still in the input folder itself, it
opens with the row main, <input>/main, and Save moves the input there; + Add a subset adds a row
whose tab follows its folder's name; a row that fails its check (✗) disables Save; ✕ removes a
new row; the task browser shows a tab for every subset once Save has moved them.
"""
import json
import os
import time
from os import path

from tests.conftest import compose_gui_config


def subset(folder: str, categories: bool = True) -> str:
    for modality in ('visual', 'thermal'):
        os.makedirs(path.join(folder, modality, '0001'), exist_ok=True)
        open(path.join(folder, modality, '0001', '000001.jpg'), 'w').close()
    if categories:
        with open(path.join(folder, 'categories.json'), 'w', encoding='utf-8') as stream:
            json.dump([{'name': 'car', 'color': '#00CF00'}], stream)
    return folder


def save_and_wait(qapp, dialog, timeout_s: float = 10) -> None:
    dialog.save()
    deadline = time.monotonic() + timeout_s
    while dialog.moving and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not dialog.moving, dialog.status.text()


def test_the_default_row_moves_the_input_in_the_folder_into_main(tmp_path, qapp):
    from gui.import_dialog import ImportDialog
    root = subset(str(tmp_path / 'cutie_tasks'))
    dialog = ImportDialog(root)
    imported = []
    dialog.imported.connect(imported.append)
    [row] = dialog.rows
    assert (row.name_edit.text(), row.folder_edit.text()) == ('main', path.join(root, 'main'))
    text = row.check_label.text()
    assert '✓' in text and '1 sequences' in text and 'move from' in text
    assert 'descriptions empty, category null' in text  # no targets.json
    assert dialog.save_button.isEnabled()
    save_and_wait(qapp, dialog)
    assert imported == [f'Imported main into {root}.']
    assert sorted(os.listdir(root)) == ['main']
    assert dialog.result() == dialog.DialogCode.Accepted


def test_add_a_subset_checks_it_and_remove_takes_it_back(tmp_path, qapp):
    from gui.import_dialog import ImportDialog
    root = str(tmp_path / 'cutie_tasks')
    subset(path.join(root, 'main'))
    dialog = ImportDialog(root)
    try:
        [main] = dialog.rows
        assert main.imported and main.folder_edit.isReadOnly() and main.remove_button.isHidden()
        assert not dialog.save_button.isEnabled()  # nothing new to import
        row = dialog.add_row()
        row.folder_edit.setText(subset(str(tmp_path / 'incoming' / 'night')))
        assert row.name_edit.text() == 'night'  # the folder's name, until it is typed
        dialog.recheck()
        assert '✓' in row.check_label.text() and dialog.save_button.isEnabled()

        row.folder_edit.setText(subset(str(tmp_path / 'incoming' / 'bare'), categories=False))
        row.name_edit.setText('night')
        row.on_name_edited('night')
        dialog.recheck()
        assert '✗' in row.check_label.text() and 'categories.json is missing' in \
            row.check_label.text()
        assert not dialog.save_button.isEnabled()

        dialog.remove_row(row)
        dialog.recheck()
        assert dialog.rows == [main] and not dialog.save_button.isEnabled()
    finally:
        dialog.close()


def test_the_browser_shows_each_imported_subset_as_a_tab(tmp_path, qapp):
    from gui.import_dialog import ImportDialog
    from gui.task_browser import TaskBrowser
    root = subset(str(tmp_path / 'cutie_tasks'))
    night = subset(str(tmp_path / 'incoming' / 'night'))
    browser = TaskBrowser(compose_gui_config(device='cpu'), root)
    try:
        assert browser.tables == {}
        assert 'press Import, then Save' in browser.console.toPlainText()
        dialog = ImportDialog(root, browser)
        dialog.imported.connect(browser.on_imported)
        dialog.add_row().folder_edit.setText(night)
        dialog.recheck()
        save_and_wait(qapp, dialog)
        assert list(browser.tables) == ['main', 'night']
        assert [browser.tabs.tabText(i) for i in range(browser.tabs.count())] == \
            ['main (1)', 'night (1)']
        assert f'Imported main, night into {root}.' in browser.console.toPlainText()
        from gui.task_browser import COLUMNS, SEQUENCE_COLUMN, STATUS_COLUMN, TARGET_COLUMN
        assert 'Modality' not in COLUMNS
        row = browser.tables['main']
        assert row.item(0, SEQUENCE_COLUMN).text() == '0001'
        assert row.item(0, STATUS_COLUMN).text() == 'not started'
        assert row.item(0, TARGET_COLUMN).text() == 'null: '  # no targets.json
        assert not path.exists(path.join(str(tmp_path), 'cutie_output'))  # nothing opened yet
    finally:
        browser.close()


def test_save_colors_a_category_list_of_names_only(tmp_path, qapp):
    from gui.import_dialog import ImportDialog
    from gui.task_input import read_categories
    root = str(tmp_path / 'cutie_tasks')
    main = subset(path.join(root, 'main'))
    with open(path.join(main, 'categories.json'), 'w', encoding='utf-8') as stream:
        json.dump(['car', 'person'], stream)
    dialog = ImportDialog(root)  # main is imported: Save is there for the colors alone
    imported = []
    dialog.imported.connect(imported.append)
    [row] = dialog.rows
    assert 'colors for 2 picked on Save' in row.check_label.text()
    assert dialog.save_button.isEnabled()
    save_and_wait(qapp, dialog)
    assert imported == ['Picked the missing category colors of main.']
    assert all(category.color for category in read_categories(main))
