"""Keep benchmark evidence honest when clients poll late or a scenario fails."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts.benchmark_service import check_job, check_masks, job_summary, observe_recovery, save, wait_job
from scripts.bench.report import main as write_report


def finished_reply():
    return {'state': 'finished', 'stop_reason': 'end', 'next': 100, 'attempt': 1,
            'messages': [], 'effective': [],
            'metrics': {'elapsed_seconds': 20.0, 'run_seconds': 10.0, 'queue_seconds': 10.0,
                        'steps': 100, 'inference_seconds': 8.0, 'commit_seconds': 2.0,
                        'commit_wait_seconds': 2.1, 'replayed_frames': 0,
                        'replay_mismatches': 0, 'oom_retries': 0, 'worker_restarts': 0}}


def test_late_polling_includes_time_before_polling_in_throughput():
    class Client:
        def call(self, *args, **kwargs):
            return finished_reply()

    result = wait_job(Client(), 'already-finished')
    assert result['poll_seconds'] < result['wall_seconds']
    summary = job_summary(result)
    assert summary['wall_seconds'] == 20
    assert summary['fps'] == 5
    assert summary['queue_seconds'] == 10


@pytest.mark.parametrize('change', [{'state': 'failed'}, {'state': 'cancelled'},
                                   {'metrics': {'replay_mismatches': 1}}])
def test_failed_or_divergent_job_cannot_be_reported_as_success(change):
    reply = dict(finished_reply(), all_messages=[])
    reply.update(change)
    with pytest.raises(AssertionError):
        check_job(reply)


def test_missing_or_changed_masks_fail_equivalence_check():
    with pytest.raises(AssertionError):
        check_masks({'differing_files': 1, 'differing_pixels': 0})


def test_recovery_timer_ignores_a_precrash_result_committed_after_kill():
    killed = {'at': 100.0, 'attempt': 1, 'frames': 60}
    observe_recovery(killed, {'attempt': 1, 'next': 61}, 100.05)
    assert 'resumed' not in killed
    observe_recovery(killed, {'attempt': 2, 'next': 61}, 108.0)
    observe_recovery(killed, {'attempt': 2, 'next': 61}, 110.0)
    assert 'resumed' not in killed
    observe_recovery(killed, {'attempt': 2, 'next': 62}, 111.0)
    assert killed['resumed'] == 11.0


def test_failed_result_serialization_preserves_previous_evidence(tmp_path):
    args = SimpleNamespace(output=str(tmp_path), result_file=str(tmp_path / 'results.json'))
    original = {'part1': {'fps': 12.3}}
    save(args, original)
    broken = copy.deepcopy(original)
    broken['cycle'] = broken
    with pytest.raises(ValueError):
        save(args, broken)
    assert json.loads((tmp_path / 'results.json').read_text()) == original


def test_report_explicitly_labels_unrun_pressure_test(tmp_path):
    data = {'date': '2026-10-07', 'tasks': [],
            'device': {'name': 'test GPU', 'total_mib': 8192, 'free_mib': 4096, 'driver': 'test'},
            'scenarios': {'I2_memory_held_by_another_process': {
                'state': 'not_run', 'reason': 'Requires a maintenance window.'}}}
    source, target = tmp_path / 'result.json', tmp_path / 'report.md'
    source.write_text(json.dumps(data))
    write_report(str(source), str(target))
    assert 'I2 was not run: Requires a maintenance window.' in target.read_text()
