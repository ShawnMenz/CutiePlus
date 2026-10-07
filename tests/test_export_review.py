"""
Export and review through the shared service: every export is tied to the content revision it
exports and verified when the command ends; Mark done needs a review and an export of the current
content; any change reopens the task. The task browser runs the same flow in shared mode.
"""
import json
import subprocess
import time
from os import path

import numpy as np
import pytest

from cutieplus.client import ClientError
from gui.task_state import export_command, load_task_set, task_status
from tests.conftest import compose_gui_config
from tests.helpers import ellipse_mask, make_task
from tests.service_env import ServiceEnv, fake_overrides

pytestmark = pytest.mark.slow


@pytest.fixture
def env(tmp_path):
    task = make_task(str(tmp_path), frame_count=8, mask_anchors=(0, ))
    environment = ServiceEnv(task.root, str(tmp_path / 'state'), fake_overrides())
    environment.task = task
    yield environment
    environment.close()


def run_export(task) -> bool:
    command = export_command(load_task_set(task.root), task.split, task.sequence)
    return subprocess.run(command, capture_output=True).returncode == 0


def test_an_export_is_tied_to_the_content_it_exports(env):
    task = env.task
    client = env.client('alice', ':2')
    lease = client.call('open_task', workspace=task.workspace)
    ids = dict(task_id=lease['task_id'], lease_epoch=lease['lease_epoch'])
    client.call('commit_frame', frame='000003', mask=ellipse_mask(2, 8), anchor='edit', **ids)
    with pytest.raises(ClientError) as error:
        client.call('mark_done', **ids)
    assert error.value.code == 'invalid'  # not under review

    client.call('submit_for_review', target_objects=[1], **ids)
    assert task_status(task.workspace) == 'under review'
    status = client.call('export_status', task_id=lease['task_id'])
    assert status['review_current'] and not status['export_current']
    with pytest.raises(ClientError) as error:
        client.call('mark_done', **ids)
    assert error.value.code == 'export_needed'

    export = client.call('begin_export', **ids)
    time.sleep(1.1)  # exported_at has one-second resolution
    assert client.call('finish_export', export_id=export['export_id'],
                       exit_ok=run_export(task))['status'] == 'complete'
    assert client.call('export_status', task_id=lease['task_id'])['export_current']
    client.call('mark_done', **ids)
    assert task_status(task.workspace) == 'done'

    # a change reopens the task, and neither the review nor the export is current any more
    client.call('commit_frame', frame='000004', mask=ellipse_mask(3, 8), anchor='edit', **ids)
    assert task_status(task.workspace) == 'in progress'
    status = client.call('export_status', task_id=lease['task_id'])
    assert not status['review_current'] and not status['export_current']


def test_an_export_during_a_change_or_a_failed_one_is_not_accepted(env):
    task = env.task
    client = env.client('alice', ':2')
    lease = client.call('open_task', workspace=task.workspace)
    ids = dict(task_id=lease['task_id'], lease_epoch=lease['lease_epoch'])
    export = client.call('begin_export', **ids)
    client.call('commit_frame', frame='000002', mask=ellipse_mask(1, 8), anchor='edit', **ids)
    result = client.call('finish_export', export_id=export['export_id'], exit_ok=run_export(task))
    assert result['status'] == 'stale'
    failed = client.call('begin_export', **ids)
    assert client.call('finish_export', export_id=failed['export_id'],
                       exit_ok=False)['status'] == 'failed'
    missing = client.call('begin_export', **ids)
    # a command that exits 0 without writing the export is not a complete export either
    tree = env.service.tasks.export_tree(env.service.tasks.info(task.workspace))
    metadata = path.join(tree, 'metadata.json')
    if path.exists(metadata):
        import os
        os.remove(metadata)
    result = client.call('finish_export', export_id=missing['export_id'], exit_ok=True)
    assert result['status'] == 'failed' and 'metadata.json' in result['detail']
    rows = env.service.db.query('SELECT status, content_revision FROM exports ORDER BY started_at')
    assert [row['status'] for row in rows] == ['stale', 'failed', 'failed']


def test_the_browser_exports_and_marks_done_through_the_service(env, qapp):
    from gui.task_browser import TaskBrowser
    task = env.task
    client = env.client('alice', ':2')
    lease = client.call('open_task', workspace=task.workspace)
    ids = dict(task_id=lease['task_id'], lease_epoch=lease['lease_epoch'])
    client.call('commit_frame', frame='000003', mask=ellipse_mask(2, 8), anchor='edit', **ids)
    client.call('submit_for_review', target_objects=[1], **ids)
    client.call('close_task', **ids)
    time.sleep(1.1)

    cfg = compose_gui_config(device='cpu')
    browser = TaskBrowser(cfg, task.root, service=client)
    try:
        table = browser.tables['main']
        table.selectRow(0)
        browser.on_mark_done()  # exports first: no export of this content yet
        deadline = time.monotonic() + 30
        while (browser.exports or browser.service_exports) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert task_status(task.workspace) == 'done', browser.console.toPlainText()
        assert env.service.db.query("SELECT status FROM exports")[-1]['status'] == 'complete'
        assert env.service.tasks.leases == {}  # the browser's lease is released
    finally:
        browser.close()
