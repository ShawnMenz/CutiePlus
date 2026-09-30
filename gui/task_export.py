"""
Runs a task set's export command (see gui/task_state.py) without blocking the GUI.
The command comes from task_set.json, so Cutie never imports the labeling pipeline.
"""
from typing import Callable, List

from PySide6.QtCore import QProcess, QProcessEnvironment


def start_export(command: List[str], log: Callable[[str], None],
                 on_finished: Callable[[bool], None]) -> QProcess:
    """Keep a reference to the returned process until on_finished runs."""
    process = QProcess()
    environment = QProcessEnvironment.systemEnvironment()
    # the command runs in the pipeline's own environment, not this one
    for key in ('VIRTUAL_ENV', 'PYTHONPATH'):
        environment.remove(key)
    process.setProcessEnvironment(environment)
    process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

    def forward_output():
        for line in bytes(process.readAllStandardOutput()).decode(errors='replace').splitlines():
            if line.strip():
                log(line)

    def failed_to_start(error):
        if error == QProcess.ProcessError.FailedToStart:
            log(f'Export could not start: {" ".join(command)}')
            on_finished(False)

    process.readyReadStandardOutput.connect(forward_output)
    process.errorOccurred.connect(failed_to_start)
    process.finished.connect(
        lambda code, status: on_finished(code == 0 and status == QProcess.ExitStatus.NormalExit))
    process.start(command[0], command[1:])
    return process
