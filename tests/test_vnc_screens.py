"""
The annotators' VNC screens follow the annotator list: Save on Manage annotators starts the screen
of every new port and, after one confirmation, stops the screen of every removed one (never the
screen the page is on); a removed name becomes inactive and the task browser draws it red. The
screens are cutie-screen@N user services; here systemctl is a fake and no real screen starts.
"""
import json
import subprocess
from os import path

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QMessageBox

from gui import vnc_screens
from gui.annotators import load_annotators, load_inactive, save_annotators
from tests.conftest import compose_gui_config
from tests.helpers import make_task


class FakeScreens:
    """Stands in for gui.vnc_screens in the dialog."""

    def __init__(self):
        self.calls = []

    def sync(self, saved, removed):
        self.calls.append((list(saved), list(removed)))
        results = {port: ('stopped', None) for port in removed}
        results.update({port: ('did not start', 'screen :9 is taken') if port == 5909 else
                        ('started', None) if port == 5905 else ('running', None)
                        for port in saved})
        return results


def open_manager(root, screens):
    from gui.annotator_manager import AnnotatorManager
    return AnnotatorManager(root, screens=screens)


def remove_ports(dialog, *ports):
    for row in [row for row in dialog.rows if row.port_box.value() in ports]:
        dialog.remove_row(row)


def answer_with(monkeypatch, *answers):
    """The answers QMessageBox.question gives, in order; returns the questions asked."""
    answers, asked = list(answers), []

    def question(parent, title, text, *args):
        asked.append(text)
        return answers.pop(0)
    monkeypatch.setattr(QMessageBox, 'question', question)
    return asked


def test_save_starts_new_ports(tmp_path, qapp):
    root = str(tmp_path)
    save_annotators(root, [{'port': 5902, 'name': 'Ann'}])
    screens = FakeScreens()
    dialog = open_manager(root, screens)
    dialog.add_row(5905, 'Bea')
    dialog.add_row(5909, 'Cal')
    assert dialog.save()
    assert screens.calls == [([5902, 5905, 5909], [])]
    status = dialog.status.text()
    assert 'Started port 5905, ready to connect.' in status
    assert 'Port 5909 did not start: screen :9 is taken.' in status
    assert '5902' not in status  # already running: not mentioned
    assert [entry['port'] for entry in load_annotators(root)] == [5902, 5905, 5909]


def test_removal_asks_once_and_no_saves_nothing(tmp_path, qapp, monkeypatch):
    root = str(tmp_path)
    save_annotators(root, [{'port': 5902, 'name': 'Ann'}, {'port': 5903, 'name': 'Bea'}])
    monkeypatch.setenv('DISPLAY', ':2')  # the page is on Ann's screen
    asked = answer_with(monkeypatch, QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
    screens = FakeScreens()
    dialog = open_manager(root, screens)
    remove_ports(dialog, 5903)

    assert not dialog.save()  # No: nothing written, nothing stopped, the removal stays staged
    assert 'Port 5903: Bea' in asked[0]
    assert len(load_annotators(root)) == 2 and not screens.calls and dialog.is_dirty()

    assert dialog.save()
    assert screens.calls == [([5902], [5903])]
    assert 'Stopped port 5903.' in dialog.status.text()
    assert load_inactive(root) == {'Bea'}


def test_the_screen_the_page_is_on_cannot_be_removed(tmp_path, qapp, monkeypatch):
    root = str(tmp_path)
    save_annotators(root, [{'port': 5902, 'name': 'Ann'}, {'port': 5903, 'name': 'Bea'}])
    monkeypatch.setenv('DISPLAY', ':2')
    asked = answer_with(monkeypatch)
    screens = FakeScreens()
    dialog = open_manager(root, screens)
    remove_ports(dialog, 5902)
    assert not dialog.save()
    assert 'port 5902 is this screen' in dialog.status.text()
    assert not asked and not screens.calls and len(load_annotators(root)) == 2


def test_inactive_names_follow_removal_rename_and_return(tmp_path):
    from gui.annotators import next_inactive
    ann, bea = {'port': 5902, 'name': 'Ann'}, {'port': 5903, 'name': 'Bea'}
    assert next_inactive(set(), [ann, bea], [ann]) == {'Bea'}  # removed
    assert next_inactive({'Bea'}, [ann], [{'port': 5902, 'name': 'Anna'}]) == {'Bea', 'Ann'}
    assert next_inactive({'Bea', 'Ann'}, [], [{'port': 5904, 'name': 'Bea'}]) == {'Ann'}  # back
    root = str(tmp_path)
    save_annotators(root, [ann], inactive={'Bea'})
    save_annotators(root, [ann, bea])  # by default the file's inactive names, less the saved ones
    assert load_inactive(root) == set()


class FakeSystemd:
    """systemctl --user for cutie-screen@N units; a started unit's port listens at once."""

    def __init__(self, listening=()):
        self.listening = set(listening)
        self.enabled, self.running, self.commands = set(), set(), []

    def run(self, command):
        self.commands.append(command)
        if command[:2] != ['systemctl', '--user']:
            return subprocess.CompletedProcess(command, 0, '', '')
        action, unit = command[2], command[-1]
        display = int(unit.split('@')[1].split('.')[0]) if '@' in unit else None
        out, code = '', 0
        if action == 'show':
            out = 'running\n' if display in self.running else 'dead\n'
        elif action == 'is-enabled':
            code = 0 if display in self.enabled else 1
        elif action == 'enable':
            self.enabled.add(display)
            self.running.add(display)
            self.listening.add(5900 + display)
        elif action == 'disable':
            self.enabled.discard(display)
            if display in self.running:
                self.running.discard(display)
                self.listening.discard(5900 + display)
        return subprocess.CompletedProcess(command, code, out, '')


def fake_systemd(monkeypatch, tmp_path, listening=()):
    system = FakeSystemd(listening)
    monkeypatch.setattr(vnc_screens, 'run', system.run)
    monkeypatch.setattr(vnc_screens, 'port_listening', lambda port: port in system.listening)
    monkeypatch.setattr(vnc_screens.path, 'exists', lambda name: False)
    monkeypatch.setattr(vnc_screens, 'UNIT_FILE', str(tmp_path / 'cutie-screen@.service'))
    monkeypatch.setattr(vnc_screens, 'LOCK_FILE', str(tmp_path / 'screens.lock'))
    return system


def actions(system):
    return [command[2:] for command in system.commands if command[2] in ('enable', 'disable')]


def test_sync_enables_saved_and_disables_removed_once(tmp_path, monkeypatch):
    system = fake_systemd(monkeypatch, tmp_path)
    assert vnc_screens.sync([5905]) == {5905: ('started', None)}
    assert vnc_screens.sync([5905]) == {5905: ('running', None)}  # again: nothing to do
    assert actions(system) == [['enable', '--now', 'cutie-screen@5.service']]
    assert vnc_screens.sync([], [5905]) == {5905: ('stopped', None)}
    assert system.enabled == set() and 5905 not in system.listening
    assert vnc_screens.sync([], [5905]) == {5905: ('absent', None)}


def test_a_screen_cutie_does_not_manage_is_left_alone(tmp_path, monkeypatch):
    system = fake_systemd(monkeypatch, tmp_path, listening={5902})  # a system tigervncserver@:2
    what, problem = vnc_screens.sync([5902])[5902]
    assert what == 'did not start' and 'setup_annotator_screens.sh' in problem
    assert actions(system) == []


def test_the_unit_keeps_nothing_of_the_screen_that_wrote_it(tmp_path, monkeypatch):
    fake_systemd(monkeypatch, tmp_path)
    monkeypatch.setenv('DISPLAY', ':2')
    monkeypatch.setenv('CUTIE_LABELER', 'someone')
    vnc_screens.write_unit_file()
    text = (tmp_path / 'cutie-screen@.service').read_text()
    assert '/usr/bin/env -i ' in text and 'tigervncserver :%i -fg -localhost yes' in text
    assert 'DISPLAY' not in text and 'CUTIE_' not in text
    assert 'WantedBy=default.target' in text


def test_the_browser_draws_inactive_names_red(tmp_path, qapp):
    from gui.task_browser import ANNOTATORS_COLUMN, INACTIVE_COLOR, TaskBrowser
    task = make_task(str(tmp_path))
    with open(path.join(task.workspace, 'progress.json'), 'w') as stream:
        json.dump({'last_edited': '2026-10-18T12:00:00', 'last_edited_by': 'bob',
                   'edited_by': ['alice', 'bob']}, stream)
    save_annotators(task.root, [{'port': 5903, 'name': 'bob'}], inactive={'alice'})
    browser = TaskBrowser(compose_gui_config(device='cpu'), task.root)
    try:
        table = browser.tables['main']
        item = table.item(0, ANNOTATORS_COLUMN)
        assert item.text() == 'alice, bob'
        assert item.toolTip() == 'changed by alice (removed), bob'
        table.setColumnWidth(ANNOTATORS_COLUMN, 200)
        font = table.font()
        font.setStyleStrategy(QFont.StyleStrategy.NoAntialias)  # text pixels in the pen's color
        table.setFont(font)
        image = table.viewport().grab(table.visualItemRect(item)).toImage()
        red = INACTIVE_COLOR
        assert any(QColor(image.pixel(x, y)) == red
                   for x in range(image.width()) for y in range(image.height()))

        save_annotators(task.root, [{'port': 5902, 'name': 'alice'}, {'port': 5903, 'name': 'bob'}])
        browser.refresh_identity()  # alice added back
        assert item.toolTip() == 'changed by alice, bob'
        image = table.viewport().grab(table.visualItemRect(item)).toImage()
        assert not any(QColor(image.pixel(x, y)) == red
                       for x in range(image.width()) for y in range(image.height()))
    finally:
        browser.close()
