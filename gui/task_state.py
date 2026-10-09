"""
Task mode: a workspace created by RGBT_MOT's tools/cutie_tasks.py.

task.json     frame directories and file lists per modality (read in place), anchor modality,
              referring text
anchors.json  frame name -> human decision {kind: mask|empty, origin: ls|ls_refined|cutie, by}
prefill.json  frame name -> {source, path} of an existing model mask for untouched frames
interval.json the dataset clip(s) when set in Cutie (1-indexed, inclusive): {start, end} for one
              interval, {intervals: [{start, end}, ...]} (sorted, never overlapping) when the sequence
              keeps several, each a clip of its own; saving binary PNGs also writes them as
              interval.txt next to them
description.json {text, by, updated_at}: the referring text when tuned in Cutie, in place of
              task.json's (the curated CSV's); saving binary PNGs also writes it as description.txt
attention.json {at, by}: the sequence needs further attention (the window's Need Attention
              button, which also removes it); the task browser shows its status as Attention, and
              Mark done waits until it is removed. Edits and reviews leave it alone
progress.json the review workflow: {last_edited, last_edited_by, edited_by} is rewritten on every
              change (so a change always needs a new review); edited_by lists everyone who has
              changed the task, in order of their first change; saving binary PNGs adds
              {review_at, review_by} (under review); the task browser's Mark done adds
              {done_at, done_by} to a task under review, after exporting it
lock.json     who has the task open (see TaskLock); a task is open in one window at a time

"by" fields name the labeler: --labeler, else the annotator assigned to the screen's port
(annotators.json in the task root, edited in the task browser), else $CUTIE_LABELER, else the OS user.

Every human edit makes its frame an anchor. Anchors are never overwritten by
propagation; an empty anchor (target absent) stops propagation.

Tasks sit in <task root>/<split>/<sequence>/; <task root>/task_set.json says how to
export one (a command template) and where the export lands (a path template).
"""
import csv
import getpass
import json
import os
import socket
import time
from datetime import datetime, timezone
from os import path
from typing import Dict, List, Optional, Tuple

from gui.annotators import annotator_for_display

try:
    import fcntl
except ImportError:  # not on Windows: tasks are not locked there
    fcntl = None

TASK_FILE = 'task.json'
ANCHORS_FILE = 'anchors.json'
PREFILL_FILE = 'prefill.json'
PROGRESS_FILE = 'progress.json'
INTERVAL_FILE = 'interval.json'
INTERVAL_TEXT = 'interval.txt'
DESCRIPTION_FILE = 'description.json'
DESCRIPTION_TEXT = 'description.txt'
ATTENTION_FILE = 'attention.json'
ATTENTION_STATUS = 'Attention'  # the status the task browser shows for a flagged sequence
TASK_SET_FILE = 'task_set.json'
LOCK_FILE = 'lock.json'


def _read_json(file_path: str, default=None):
    if default is not None and not path.exists(file_path):
        return default
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def _write_json(file_path: str, payload) -> None:
    temporary = file_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(temporary, file_path)


Interval = Tuple[int, int]  # (start, end) of one clip, 1-indexed, inclusive


def intervals_payload(intervals: List[Interval]) -> dict:
    """interval.json: the one-interval shape stays {start, end}; several are a list, with no
    top-level start/end, so a reader that knows only one interval fails instead of dropping the
    others."""
    if len(intervals) == 1:
        return {'start': intervals[0][0], 'end': intervals[0][1]}
    return {'intervals': [{'start': start, 'end': end} for start, end in intervals]}


def intervals_from_payload(payload: dict) -> List[Interval]:
    if 'intervals' in payload:
        return sorted((int(item['start']), int(item['end'])) for item in payload['intervals'])
    return [(int(payload['start']), int(payload['end']))]


def check_intervals(intervals: List[Interval], frame_count: int) -> List[str]:
    """What is wrong with a list of intervals, in words; empty when it can be saved. Intervals may
    touch (a long clip split in two) but not overlap: a frame cannot be in two clips."""
    problems = []
    for start, end in intervals:
        if not 1 <= start <= end <= frame_count:
            problems.append(f'{start}-{end} is not an interval of frames 1-{frame_count}.')
    ordered = sorted(intervals)
    for (start_a, end_a), (start_b, end_b) in zip(ordered, ordered[1:]):
        if start_b <= end_a:
            problems.append(f'{start_a}-{end_a} and {start_b}-{end_b} overlap.')
    if not intervals:
        problems.append('There is no interval.')
    return problems


def format_intervals(intervals: List[Interval]) -> str:
    return ', '.join(f'{start}-{end}' for start, end in intervals)


def interval_index_from(intervals: List[Interval], frame: int, direction: int) -> int:
    """The interval whose start is the nearest after (direction 1) or before (-1) the 1-indexed
    frame, around the ends: inside an interval, back goes to its own start first. intervals are
    sorted."""
    starts = [start for start, _ in intervals]
    if direction > 0:
        return next((i for i, start in enumerate(starts) if start > frame), 0)
    return next((i for i in reversed(range(len(starts))) if starts[i] < frame), len(starts) - 1)


def write_interval_text(directory: str, intervals: List[Interval]) -> None:
    """interval.txt: interval_start / interval_end for one interval; interval_count and numbered
    keys for several (again no bare interval_start, so an old reader cannot keep only the first)."""
    if len(intervals) == 1:
        text = f'interval_start={intervals[0][0]}\ninterval_end={intervals[0][1]}\n'
    else:
        text = f'interval_count={len(intervals)}\n' + ''.join(
            f'interval_{number}_start={start}\ninterval_{number}_end={end}\n'
            for number, (start, end) in enumerate(intervals, start=1))
    with open(path.join(directory, INTERVAL_TEXT), 'w', encoding='utf-8') as f:
        f.write(text)


def write_description_text(directory: str, text: str) -> None:
    with open(path.join(directory, DESCRIPTION_TEXT), 'w', encoding='utf-8') as f:
        f.write(f'description={text}\n')


def saved_description(workspace: str) -> Optional[str]:
    # the text tuned in Cutie, None when the curated one stands
    saved = _read_json(path.join(workspace, DESCRIPTION_FILE), {})
    return saved.get('text') or None


def attention_of(workspace: str) -> Optional[dict]:
    # {at, by} while the sequence needs attention, else None
    return _read_json(path.join(workspace, ATTENTION_FILE), {}) or None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def labeler_name(argument: Optional[str] = None, task_root: Optional[str] = None) -> str:
    """Who is labeling: --labeler, else the annotator assigned to this screen's port (see
    gui/annotators.py, in the task root), else $CUTIE_LABELER, else the OS user."""
    if argument:
        return argument
    assigned = annotator_for_display(task_root) if task_root is not None else None
    return assigned or os.environ.get('CUTIE_LABELER') or getpass.getuser()


def describe_attention(entry: Optional[dict]) -> str:
    """e.g. 'flagged by alice at 10:32' (local time)."""
    if not entry:
        return ''
    text = f'flagged by {entry.get("by", "?")}'
    try:
        text += f' at {datetime.fromisoformat(entry["at"]).astimezone():%H:%M}'
    except (KeyError, TypeError, ValueError):
        pass
    return text


def describe_holder(holder: dict) -> str:
    """e.g. 'alice (:3, pid 1234) since 10:32'."""
    if not holder:
        return 'another window'
    text = f'{holder.get("labeler", "?")} ({holder.get("display") or "no display"}, ' \
           f'pid {holder.get("pid", "?")})'
    if holder.get('activity'):
        text += f' {holder["activity"]}'
    try:
        opened = datetime.fromisoformat(holder['opened_at']).astimezone()
        text += f' since {opened:%H:%M}'
    except (KeyError, TypeError, ValueError):
        pass
    return text


class TaskLocked(Exception):

    def __init__(self, workspace: str, holder: dict):
        self.holder = holder
        super().__init__(f'{path.basename(path.abspath(workspace))} is open by '
                         f'{describe_holder(holder)}.')


class TaskLock:
    """
    Holds a task for one window: an exclusive flock on <task>/lock.json, released on close, or by
    the kernel when the process dies, so a crashed session never leaves a task locked. The file
    only describes the holder and is emptied, never deleted, on release: a lock taken on a
    deleted file would not exclude one taken on its replacement.
    """

    def __init__(self, stream=None):
        self.stream = stream

    @classmethod
    def acquire(cls, workspace: str, labeler: str, activity: str = '',
                holder: Optional[dict] = None) -> 'TaskLock':
        """Raises TaskLocked when another window has the task. activity: what the holder does
        besides labeling, e.g. 'exporting'. holder: fields that describe the holder instead of
        this process (the shared service holds tasks for its windows)."""
        if fcntl is None:
            return cls()
        stream = open(path.join(workspace, LOCK_FILE), 'a+', encoding='utf-8')
        for attempt in range(3):
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # lock_holder's probe holds the lock for an instant
                time.sleep(0.05)
        else:
            stream.close()
            raise TaskLocked(workspace, lock_holder(workspace) or {})
        stream.seek(0)
        stream.truncate()
        description = {'labeler': labeler, 'display': os.environ.get('DISPLAY', ''),
                       'host': socket.gethostname(), 'pid': os.getpid(), 'activity': activity,
                       'opened_at': utc_now()}
        description.update(holder or {})
        json.dump(description, stream, ensure_ascii=False)
        stream.flush()
        return cls(stream)

    def release(self) -> None:
        if self.stream is None or self.stream.closed:
            return
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.flush()
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def lock_holder(workspace: str) -> Optional[dict]:
    """Who has the task open ({} when that is unreadable), or None when nobody has."""
    lock_path = path.join(workspace, LOCK_FILE)
    if fcntl is None:
        return None
    try:
        if path.getsize(lock_path) == 0:
            return None
        with open(lock_path, encoding='utf-8') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    return json.loads(stream.read() or '{}')
                except ValueError:
                    return {}
            # nobody holds it: the description is left over from a crashed window
            fcntl.flock(stream, fcntl.LOCK_UN)
            return None
    except FileNotFoundError:
        return None


def load_task_set(task_root: str) -> dict:
    return _read_json(path.join(task_root, TASK_SET_FILE))


def task_root_of(workspace: str) -> str:
    # <task root>/<split>/<sequence>
    return path.dirname(path.dirname(path.abspath(workspace)))


def export_command(task_set: dict, split: str, sequence: str) -> List[str]:
    return [part.format(split=split, sequence=sequence) for part in task_set['export_command']]


def task_annotators(progress: dict, anchors: Dict[str, dict]) -> List[str]:
    """Everyone who has changed a task (put it in progress), in order of their first change.
    Tasks last changed before edited_by existed fall back to their anchors' authors and the
    last editor."""
    if 'edited_by' in progress:
        recorded = list(progress['edited_by'])
    else:
        recorded = [anchor['by'] for anchor in
                    sorted(anchors.values(), key=lambda entry: entry.get('updated_at', ''))
                    if anchor.get('by')]
        if progress.get('last_edited_by'):
            recorded.append(progress['last_edited_by'])
    return list(dict.fromkeys(recorded))


def summarize_task(workspace: str) -> dict:
    # one row of the task browser
    task = _read_json(path.join(workspace, TASK_FILE))
    anchors = _read_json(path.join(workspace, ANCHORS_FILE), {})
    progress = _read_json(path.join(workspace, PROGRESS_FILE), {})
    kinds = [anchor['kind'] for anchor in anchors.values()]
    frame_files = task.get('frame_files', {}).get(task['anchor_modality'], [])
    return {
        'split': task['split'],
        'sequence': task['sequence_id'],
        'frames': task.get('frame_count', len(frame_files)),
        'mask_anchors': kinds.count('mask'),
        'empty_anchors': kinds.count('empty'),
        # a sequence that needs attention shows that instead of its review status
        'status': ATTENTION_STATUS if attention_of(workspace) else task_status(workspace),
        'attention': attention_of(workspace),
        'last_edited_by': progress.get('last_edited_by'),
        'annotators': task_annotators(progress, anchors),
        'open_by': lock_holder(workspace),
        'modality': task['anchor_modality'],
        'category': task.get('category', ''),
        'description': saved_description(workspace) or task.get('description', ''),
        'target': f'{task.get("category", "")}: '
                  f'{saved_description(workspace) or task.get("description", "")}',
    }


def task_status(workspace: str) -> str:
    # not started / in progress / under review / done
    return status_of(_read_json(path.join(workspace, PROGRESS_FILE), {}))


def export_is_current(workspace: str, task_set: dict) -> bool:
    # an export exists and is newer than the last change; the export is stamped when it starts
    # and timestamps have 1 s resolution, so a change in the export's second counts as newer
    task = _read_json(path.join(workspace, TASK_FILE))
    export_tree = task_set['export_tree'].format(modality=task['anchor_modality'],
                                                 sequence=task['sequence_id'])
    exported = _read_json(path.join(export_tree, 'metadata.json'), {}).get('exported_at')
    edited = _read_json(path.join(workspace, PROGRESS_FILE), {}).get('last_edited')
    return exported is not None and (edited is None or exported > edited)


def _update_progress(workspace: str, **fields) -> None:
    progress_path = path.join(workspace, PROGRESS_FILE)
    progress = _read_json(progress_path, {})
    progress.update(fields)
    _write_json(progress_path, progress)


def submit_for_review(workspace: str, labeler: str) -> None:
    # from any status, including done: the task waits for a new review
    progress = _read_json(path.join(workspace, PROGRESS_FILE), {})
    _write_json(path.join(workspace, PROGRESS_FILE), submitted_progress(progress, labeler, utc_now()))


def mark_done(workspace: str, labeler: str) -> None:
    # only for a task under review whose export is current
    _update_progress(workspace, done_at=utc_now(), done_by=labeler)


# the rules of anchors and of the review status, shared by TaskState (local mode) and the shared
# service's coordinator, so both record changes alike
def next_anchor_entry(previous: Optional[dict], kind: str, edited: bool, labeler: str,
                      now: str) -> Optional[dict]:
    """The anchor entry after a human decision on a frame; None when nothing changes (accepting
    the shown mask of an anchor of the same kind). An edit of a Label Studio anchor keeps its
    source and becomes ls_refined."""
    if previous is not None and not edited and previous['kind'] == kind:
        return None
    entry = {'kind': kind, 'updated_at': now, 'by': labeler}
    if previous is not None and previous['origin'] in ('ls', 'ls_refined'):
        entry['origin'] = 'ls_refined'
        for key in ('ls_file', 'ls_modality'):
            if key in previous:
                entry[key] = previous[key]
    else:
        entry['origin'] = 'cutie'
    return entry


def touched_progress(progress: dict, anchors: Dict[str, dict], labeler: str, now: str) -> dict:
    """progress.json after a change: back to in progress (any review or done mark dropped);
    edited_by keeps everyone who has changed the task, in order."""
    editors = task_annotators(progress, anchors)
    if labeler not in editors:
        editors.append(labeler)
    return {'last_edited': now, 'last_edited_by': labeler, 'edited_by': editors}


def submitted_progress(progress: dict, labeler: str, now: str) -> dict:
    """progress.json once the task is submitted for review (from any status, including done)."""
    progress = dict(progress)
    progress.pop('done_at', None)
    progress.pop('done_by', None)
    progress['review_at'] = now
    progress['review_by'] = labeler
    return progress


def status_of(progress: dict) -> str:
    if 'done_at' in progress:
        return 'done'
    if 'review_at' in progress:
        return 'under review'
    return 'in progress' if 'last_edited' in progress else 'not started'


class TaskState:

    def __init__(self, workspace: str, labeler: Optional[str] = None):
        self.workspace = workspace
        self.labeler = labeler_name(labeler, task_root_of(workspace))
        task = _read_json(path.join(workspace, TASK_FILE))
        self.sequence_id: str = task['sequence_id']
        self.split: str = task['split']
        self.anchor_modality: str = task['anchor_modality']
        self.frame_dirs: Dict[str, str] = task['frame_dirs']
        self.frame_files: Optional[Dict[str, List[str]]] = task.get('frame_files')
        self.frame_count: Optional[int] = task.get('frame_count')
        self.curated_interval: Optional[List[int]] = task.get('curated_interval')
        self.category: str = task.get('category', '')
        # the curated text, and the one shown: the text tuned in Cutie, else the curated one
        self.original_description: str = task.get('description', '')
        self.description: str = saved_description(workspace) or self.original_description
        self.anchors: Dict[str, dict] = _read_json(path.join(workspace, ANCHORS_FILE))
        prefill_path = path.join(workspace, PREFILL_FILE)
        self.prefill: Dict[str, dict] = _read_json(prefill_path) if path.exists(prefill_path) else {}
        self.attention: Optional[dict] = attention_of(workspace)

    @staticmethod
    def is_task(workspace: Optional[str]) -> bool:
        return workspace is not None and path.isfile(path.join(workspace, TASK_FILE))

    @property
    def other_modality(self) -> Optional[str]:
        others = [m for m in self.frame_dirs if m != self.anchor_modality]
        return others[0] if others else None

    def anchor(self, name: str) -> Optional[dict]:
        return self.anchors.get(name)

    def reset_block_reason(self) -> Optional[str]:
        """A curated done sequence must keep its masks until it leaves that split."""
        try:
            task_set = load_task_set(task_root_of(self.workspace))
            csv_path = task_set['curated_csv']
            with open(csv_path, encoding='utf-8-sig', newline='') as stream:
                rows = csv.DictReader(stream)
                if not rows.fieldnames or 'split' not in rows.fieldnames:
                    return f'Curated CSV has no split column: {csv_path}'
                for index, row in enumerate(rows, 1):
                    if f'{index:04d}' == self.sequence_id:
                        if (row.get('split') or '').strip() == 'done':
                            return f'{self.sequence_id} is done in the curated CSV.'
                        return None
            return f'{self.sequence_id} has no row in the curated CSV.'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return f'Cannot verify the curated split: {exc}'

    def export_directories(self) -> List[str]:
        task_set = load_task_set(task_root_of(self.workspace))
        directories = []
        for modality in ('visual', 'thermal'):
            directory = path.abspath(task_set['export_tree'].format(
                modality=modality, sequence=self.sequence_id))
            if (path.basename(directory) != self.sequence_id
                    or path.basename(path.dirname(directory)) != f'cutie_masks_{modality}'):
                raise ValueError(f'Unexpected Cutie export path: {directory}')
            directories.append(directory)
        return directories

    def clear_mask_references(self) -> None:
        _write_json(path.join(self.workspace, ANCHORS_FILE), {})
        _write_json(path.join(self.workspace, PREFILL_FILE), {})
        self.anchors.clear()
        self.prefill.clear()
        self.touch()

    def set_anchor(self, name: str, kind: str, edited: bool) -> None:
        """Record a human decision; edited=False accepts the shown mask without changing it."""
        entry = next_anchor_entry(self.anchors.get(name), kind, edited, self.labeler, utc_now())
        if entry is None:
            return
        self.anchors[name] = entry
        self.save()

    def remove_anchor(self, name: str) -> None:
        if self.anchors.pop(name, None) is not None:
            self.save()

    def restore_anchor(self, name: str, entry: Optional[dict]) -> None:
        # undo/redo: put back an earlier anchor entry, or no anchor at all
        if entry is None:
            self.anchors.pop(name, None)
        else:
            self.anchors[name] = dict(entry)
        self.save()

    def intervals(self, frame_count: int) -> Tuple[List[Interval], str]:
        """(intervals, source) of the dataset clips, sorted, 1-indexed and inclusive: set in
        Cutie, else the curated CSV's one, else the whole sequence."""
        saved = _read_json(path.join(self.workspace, INTERVAL_FILE), {})
        if saved:
            return intervals_from_payload(saved), 'cutie'
        if self.curated_interval:
            return [(self.curated_interval[0], self.curated_interval[1])], 'curated'
        return [(1, frame_count)], 'full'

    def save_intervals(self, intervals: List[Interval]) -> None:
        _write_json(path.join(self.workspace, INTERVAL_FILE), intervals_payload(sorted(intervals)))

    def save_interval(self, start: int, end: int) -> None:
        self.save_intervals([(start, end)])

    def save_description(self, text: str) -> None:
        _write_json(path.join(self.workspace, DESCRIPTION_FILE),
                    {'text': text, 'by': self.labeler, 'updated_at': utc_now()})
        self.description = text

    def save_attention(self, flagged: bool) -> None:
        # not a change of the task: the review status and progress.json stay as they are
        attention_path = path.join(self.workspace, ATTENTION_FILE)
        if flagged:
            self.attention = {'at': utc_now(), 'by': self.labeler}
            _write_json(attention_path, self.attention)
        else:
            self.attention = None
            if path.exists(attention_path):
                os.remove(attention_path)

    def save(self) -> None:
        _write_json(path.join(self.workspace, ANCHORS_FILE), dict(sorted(self.anchors.items())))
        self.touch()

    def touch(self) -> None:
        # a change: back to in progress, dropping any review or done mark; edited_by keeps
        # everyone who has changed the task
        progress_path = path.join(self.workspace, PROGRESS_FILE)
        _write_json(progress_path, touched_progress(_read_json(progress_path, {}), self.anchors,
                                                    self.labeler, utc_now()))
