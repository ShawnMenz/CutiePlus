"""
The GPU queue window (the task browser's GPU queue button, every role): the shared GPU worker's
state, the propagation it runs and those waiting, in order, with their owners as far as the user
may see them (their own; a supervisor's annotators'; an admin everyone's), and the user's own
recent propagations with Cancel and Retry. Cancel shows where the service would allow it; it
checks again. Refreshed off the GUI thread every two seconds.
"""
import uuid
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget,
                               QVBoxLayout, QWidget)

from gui.gui_utils import fit_window
from gui.manage_annotators import cell, short_time, table
from gui.service_link import ServiceLink
from gui.style import Card

REFRESH_MS = 2000
STATES = {'idle': 'Idle', 'busy': 'Busy', 'starting': 'Starting', 'recovering': 'Recovering',
          'offline': 'Offline'}
STATUS = {'queued': 'Waiting', 'running': 'Running', 'cancel_requested': 'Cancelling',
          'succeeded': 'Done', 'failed': 'Failed', 'cancelled': 'Cancelled',
          'interrupted': 'Interrupted'}


def describe(job: dict) -> str:
    text = STATUS.get(job['status'], job['status'])
    if job['status'] == 'queued' and job['position'] is not None:
        text += f' ({job["position"]} ahead)'
    elif job['status'] in ('running', 'cancel_requested'):
        text += f' {round(100 * (job["progress"] or 0))} %'
    return text


class GpuQueueWindow(QDialog):

    def __init__(self, link: ServiceLink, parent=None):
        super().__init__(parent)
        self.link = link
        self.setWindowTitle('GPU queue')
        self.loading = False
        self.worker_label = QLabel('Loading…')
        self.worker_label.setTextFormat(Qt.TextFormat.PlainText)
        self.queue_table = table(('#', 'Owner', 'Task', 'Status', 'Submitted (UTC)', ''))
        self.mine_table = table(('Status', 'Task', 'Submitted (UTC)', 'Finished', 'Result',
                                 'Error', ''))
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(Card(self.worker_label, title='GPU worker'))
        layout.addWidget(Card(self.queue_table, title='Running and waiting propagations'), 1)
        layout.addWidget(Card(self.mine_table, title='Your propagations'), 1)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_MS)
        self.refresh()
        fit_window(self, QSize(900, 620))

    def refresh(self) -> None:
        if self.loading:
            return
        self.loading = True
        self.link.request('gpu_status', self.on_status)

    def on_status(self, status, error) -> None:
        self.loading = False
        if error is not None:
            self.worker_label.setText(f'The service cannot say: {error}')
            return
        worker = status['worker']
        text = f'{STATES.get(worker["state"], worker["state"])}'
        if worker.get('device'):
            text += f' on {worker["device"]}'
        if worker.get('memory_total_mb'):
            text += (f' · {worker["memory_used_mb"] / 1024:.1f} of '
                     f'{worker["memory_total_mb"] / 1024:.1f} GB in use')
        if worker.get('last_error') and worker['state'] != 'idle':
            text += f' · {worker["last_error"]}'
        self.worker_label.setText(text)
        self.fill(self.queue_table, status['queue'], queue=True)
        self.fill(self.mine_table, status['mine'], queue=False)

    def fill(self, widget: QTableWidget, jobs, queue: bool) -> None:
        widget.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            task = '/'.join(job['workspace'].split('/')[-2:]) if job.get('workspace') else '—'
            if queue:
                values = (str(row + 1), job['owner'] or 'another user', task, describe(job),
                          short_time(job['submitted_at']))
            else:
                values = (describe(job), task, short_time(job['submitted_at']),
                          short_time(job['finished_at']), job['result_state'],
                          job['error_summary'] or '')
            for column, value in enumerate(values):
                widget.setItem(row, column, cell(value))
            widget.setCellWidget(row, len(values), self.actions(job))

    def actions(self, job: dict) -> Optional[QWidget]:
        buttons = []
        # others' jobs show an owner only where the service lets this user act on them
        if job['status'] in ('queued', 'running') and (job['mine'] or job['owner']):
            cancel = QPushButton('Cancel')
            cancel.clicked.connect(lambda _=False, job=job: self.run(
                'Cancel', 'cancel_job', job_id=job['id']))
            buttons.append(cancel)
        if job['mine'] and job['status'] in ('failed', 'interrupted', 'cancelled'):
            retry = QPushButton('Retry')
            retry.setToolTip('A new job from the same inputs, at the end of the queue; its '
                             'result is applied the next time the task opens.')
            retry.clicked.connect(lambda _=False, job=job: self.run(
                'Retry', 'retry_job', job_id=job['id'], idempotency_key=uuid.uuid4().hex))
            buttons.append(retry)
        if not buttons:
            return None
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        for button in buttons:
            row.addWidget(button)
        return widget

    def run(self, what: str, op: str, **args) -> None:
        def done(result, error) -> None:
            self.status.setText(f'{what}: {error}' if error is not None else f'{what}: sent.')
            self.refresh()

        self.link.request(op, done, **args)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
