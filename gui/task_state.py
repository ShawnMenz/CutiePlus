"""
Task mode: one sequence of the input (gui/task_input.py) and the work done on it, kept in its
work folder, <output root>/<tab>/<sequence>/ (work_folder; the output root is cutie_output beside
the input root). The folder holds two things:

masks/         the saved mask of every anchor and propagated frame, written as you label; a frame
               without one shows its pregenerated mask, if the input has one
metadata.json  everything else, one section each (a section is absent until it is set):

  source       {input, tab, sequence}: where its frames are, the input root relative to this
               folder; the frames, the target text and the pregenerated masks are read from there
  category     {name, by, updated_at}: the category chosen in Cutie (from the subset's
               categories.json), in place of the input's (targets.json); a change like any other
  description  {text, by, updated_at}: the referring text when tuned in Cutie, in place of the
               input's (targets.json)
  intervals    the dataset clip(s) when set in Cutie (1-indexed, inclusive): {start, end} for one
               interval, {intervals: [{start, end}, ...]} (sorted, never overlapping) when the
               sequence keeps several, each a clip of its own; an added interval's entry also
               carries its description and category (the first interval's has neither: the
               sequence's describe it)
  progress     the review workflow: {last_edited, last_edited_by, edited_by} is rewritten on
               every change (so a change always needs a new review); edited_by lists everyone who
               has changed the task, in order of their first change; Save (Ctrl+S) adds
               {review_at, review_by} (under review); the task browser's Mark done adds
               {done_at, done_by} to a task under review; the labeling window's Ctrl+Q (quit and
               mark done) saves and marks it done in one go, and also adds {quit_at}, when it was
               pressed
  attention    {at, by}: the sequence needs further attention (the window's Need Attention
               button, which also removes it); the task browser shows its status as Attention, and
               Mark done waits until it is removed. Edits and reviews leave it alone
  discard      {at, by}: the sequence is left out of the dataset (the task browser's Discard,
               which Restore undoes); its status is discarded, its done mark and Need Attention
               are removed, and Mark done refuses it
  anchors      frame name -> human decision {kind: mask|empty, updated_at, by}
  revision     a number raised with every change of the masks or anchors (an edit, a propagation
               saved, Reset): a GPU job remembers the one it started from, and its result is
               applied only to that same revision (cutieplus/backends/service.py)

Who has the task open is not output: the lock lives in a folder every window of the account shares
(lock_path), whichever desktop it runs on.
"by" fields name the labeler: --labeler, else $CUTIE_LABELER, else the OS user.

Every human edit makes its frame an anchor. Anchors are never overwritten by
propagation; an empty anchor (target absent) stops propagation. The work folder is made when the
sequence is first opened (or discarded); until then the sequence is not started. The dataset is
exported from the done sequences (gui/dataset_export.py).
"""
import getpass
import hashlib
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from os import path
from typing import Dict, List, Optional, Tuple

from gui.task_input import (NULL_CATEGORY, Category, InputError, SequenceInput, frame_files,
                            pregenerated_masks, read_categories, read_sequence)

try:
    import fcntl
except ImportError:  # not on Windows: tasks are not locked there
    fcntl = None

METADATA_FILE = 'metadata.json'
MASKS_DIR = 'masks'
# the sections of metadata.json, in the file's order
SOURCE, CATEGORY, DESCRIPTION, INTERVALS, PROGRESS, ATTENTION, DISCARD, ANCHORS, REVISION = (
    'source', 'category', 'description', 'intervals', 'progress', 'attention', 'discard',
    'anchors', 'revision')
SECTIONS = (SOURCE, CATEGORY, DESCRIPTION, INTERVALS, PROGRESS, ATTENTION, DISCARD, ANCHORS,
            REVISION)
ATTENTION_STATUS = 'Attention'  # the status the task browser shows for a flagged sequence
DISCARDED_STATUS = 'discarded'  # before any other status
# the task locks of every window of the account, on this desktop or a remote one (lock_path)
LOCK_DIR = path.join('.cache', 'cutieplus', 'task-locks')  # in the account's home folder
LOCK_DIR_VARIABLE = 'CUTIEPLUS_LOCK_DIR'  # another folder, for the tests
DONE_FIELDS = ('done_at', 'done_by', 'quit_at')  # the done mark in progress
OUTPUT_DIR = 'cutie_output'  # the work folders, beside the input root
PREGENERATED = 'pregenerated'  # the source of a frame's pregenerated mask


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


# the browser and its labeling window share a process: a section written by one (Discard) must
# survive the other's next write (an anchor), so every write reads the file again under this lock
_metadata_lock = threading.Lock()


def read_metadata(workspace: str) -> dict:
    # the work folder's metadata.json, {} before it exists
    return _read_json(path.join(workspace, METADATA_FILE), {})


def read_section(workspace: str, section: str, default=None):
    value = read_metadata(workspace).get(section)
    return default if value is None else value


def write_sections(workspace: str, **sections) -> None:
    """Sets sections of metadata.json, leaving the others as they are; a None value removes its
    section. One atomic write."""
    with _metadata_lock:
        _write_unlocked(workspace, **sections)


def _write_unlocked(workspace: str, **sections) -> None:
    # write_sections, for a caller that holds _metadata_lock (a read-modify-write of a section)
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f'unknown metadata sections: {", ".join(sorted(unknown))}')
    metadata = read_metadata(workspace)
    metadata.update(sections)
    _write_json(path.join(workspace, METADATA_FILE),
                {key: metadata[key] for key in SECTIONS if metadata.get(key) is not None})


Interval = Tuple[int, int]  # (start, end) of one clip, 1-indexed, inclusive


def intervals_payload(intervals: List[Interval],
                      descriptions: Optional[Dict[Interval, str]] = None,
                      categories: Optional[Dict[Interval, str]] = None) -> dict:
    """The intervals section: the one-interval shape stays {start, end}; several are a list, with no
    top-level start/end, so a reader that knows only one interval fails instead of dropping the
    others. descriptions, categories: the texts and categories of the added intervals; the first
    interval has neither (the sequence's describe it), and neither has a single interval."""
    if len(intervals) == 1:
        return {'start': intervals[0][0], 'end': intervals[0][1]}
    entries = []
    for start, end in sorted(intervals):
        entry = {'start': start, 'end': end}
        if descriptions and descriptions.get((start, end)):
            entry['description'] = descriptions[(start, end)]
        if categories and categories.get((start, end)):
            entry['category'] = categories[(start, end)]
        entries.append(entry)
    return {'intervals': entries}


def intervals_from_payload(payload: dict) -> List[Interval]:
    if 'intervals' in payload:
        return sorted((int(item['start']), int(item['end'])) for item in payload['intervals'])
    return [(int(payload['start']), int(payload['end']))]


def dataset_intervals(payload: dict, frame_count: int) -> Tuple[List[Interval], str]:
    """(intervals, source) of the dataset clips, sorted, 1-indexed and inclusive: set in Cutie
    (the intervals section, as saved), else the whole sequence."""
    if payload:
        return intervals_from_payload(payload), 'cutie'
    return [(1, frame_count)], 'full'


def first_interval(payload: dict) -> Optional[Interval]:
    """The interval the sequence's description describes: the entry without a description of its
    own, else the first; None when no interval is saved."""
    if not payload:
        return None
    if 'intervals' not in payload:
        return intervals_from_payload(payload)[0]
    entries = sorted(payload['intervals'], key=lambda item: (int(item['start']), int(item['end'])))
    plain = [item for item in entries if not item.get('description')] or entries
    return int(plain[0]['start']), int(plain[0]['end'])


def interval_descriptions(intervals: List[Interval], payload: dict,
                          sequence_description: str) -> Dict[Interval, str]:
    """Each interval's referring text: the first interval's is the sequence's; an added one has
    its own, else (saved before intervals had descriptions) the sequence's."""
    first = first_interval(payload) if payload else None
    own = {} if first is None or 'intervals' not in payload else {
        (int(item['start']), int(item['end'])): item['description']
        for item in payload['intervals'] if item.get('description')}
    return {interval: (sequence_description if interval == first
                       else own.get(interval) or sequence_description)
            for interval in intervals}


def added_interval_descriptions(intervals: List[Interval], payload: dict,
                                sequence_description: str) -> Dict[Interval, str]:
    """The texts saved with the intervals: every interval's but the first's (which the sequence's
    description is); an interval saved without one gets the sequence's, from then on its own."""
    first = first_interval(payload) or intervals[0]
    return {interval: text for interval, text in
            interval_descriptions(intervals, payload, sequence_description).items()
            if interval != first}


def interval_categories(intervals: List[Interval], payload: dict,
                        sequence_category: Optional[str]) -> Dict[Interval, Optional[str]]:
    """Each interval's category: the first interval's is the sequence's; an added one has its
    own, else (saved before intervals had categories) the sequence's."""
    first = first_interval(payload) if payload else None
    own = {} if first is None or 'intervals' not in payload else {
        (int(item['start']), int(item['end'])): item['category']
        for item in payload['intervals'] if item.get('category')}
    return {interval: (sequence_category if interval == first
                       else own.get(interval) or sequence_category)
            for interval in intervals}


def added_interval_categories(intervals: List[Interval], payload: dict,
                              sequence_category: Optional[str]) -> Dict[Interval, Optional[str]]:
    """The categories saved with the intervals: every interval's but the first's (the
    sequence's)."""
    first = first_interval(payload) or intervals[0]
    return {interval: category for interval, category in
            interval_categories(intervals, payload, sequence_category).items()
            if interval != first}


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


def saved_description(workspace: str, metadata: Optional[dict] = None) -> Optional[str]:
    # the text tuned in Cutie, None when the input's stands; metadata: the file, already read
    metadata = read_metadata(workspace) if metadata is None else metadata
    return (metadata.get(DESCRIPTION) or {}).get('text') or None


def saved_category(workspace: str, metadata: Optional[dict] = None) -> Optional[str]:
    # the category chosen in Cutie, None when the input's stands
    metadata = read_metadata(workspace) if metadata is None else metadata
    return (metadata.get(CATEGORY) or {}).get('name') or None


def attention_of(workspace: str, metadata: Optional[dict] = None) -> Optional[dict]:
    # {at, by} while the sequence needs attention, else None
    metadata = read_metadata(workspace) if metadata is None else metadata
    return metadata.get(ATTENTION) or None


def discard_of(workspace: str, metadata: Optional[dict] = None) -> Optional[dict]:
    # {at, by} while the sequence is discarded, else None
    metadata = read_metadata(workspace) if metadata is None else metadata
    return metadata.get(DISCARD) or None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def labeler_name(argument: Optional[str] = None) -> str:
    """Who is labeling: --labeler, else $CUTIE_LABELER, else the OS user."""
    return argument or os.environ.get('CUTIE_LABELER') or getpass.getuser()


def describe_attention(entry: Optional[dict], verb: str = 'flagged') -> str:
    """e.g. 'flagged by alice at 10:32' (local time); also for a discard."""
    if not entry:
        return ''
    text = f'{verb} by {entry.get("by", "?")}'
    try:
        text += f' at {datetime.fromisoformat(entry["at"]).astimezone():%H:%M}'
    except (KeyError, TypeError, ValueError):
        pass
    return text


def describe_holder(holder: dict) -> str:
    """e.g. 'alice (pid 1234) since 10:32'."""
    if not holder:
        return 'another window'
    text = f'{holder.get("labeler", "?")} (pid {holder.get("pid", "?")})'
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


def lock_folder() -> str:
    """The same folder for every process of the account: the home folder comes from the account,
    not from $HOME or $XDG_RUNTIME_DIR, which differ between your login and a VNC desktop's
    service (two windows would then both get the same task)."""
    folder = os.environ.get(LOCK_DIR_VARIABLE)
    if folder:
        return folder
    try:
        import pwd
        home = pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):  # not on Linux, or an account without an entry
        home = path.expanduser('~')
    return path.join(home, LOCK_DIR)


def lock_path(workspace: str, create_folder: bool = True) -> str:
    """The task's lock file, outside the output (lock_folder), named after the sequence and its
    work folder's path."""
    folder = lock_folder()
    if create_folder:
        os.makedirs(folder, mode=0o700, exist_ok=True)  # private to the account
    workspace = path.realpath(workspace)
    digest = hashlib.sha1(workspace.encode('utf-8')).hexdigest()[:16]
    return path.join(folder, f'{path.basename(workspace)}-{digest}.json')


class TaskLock:
    """
    Holds a task for one window (or the export): an exclusive flock on its lock file (lock_path),
    released on close, or by the kernel when the process dies, so a crashed window never leaves a
    task locked. The file only describes the holder and is emptied, never deleted, on release: a
    lock taken on a deleted file would not exclude one taken on its replacement.
    """

    def __init__(self, stream=None):
        self.stream = stream

    @classmethod
    def acquire(cls, workspace: str, labeler: str, activity: str = '') -> 'TaskLock':
        """Raises TaskLocked when another window has the task. activity: what the holder does
        besides labeling, e.g. 'exporting'."""
        if fcntl is None:
            return cls()
        # never through a symbolic link planted in its place
        descriptor = os.open(lock_path(workspace),
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        stream = os.fdopen(descriptor, 'r+', encoding='utf-8')
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
        description = {'labeler': labeler, 'host': socket.gethostname(), 'pid': os.getpid(),
                       'activity': activity, 'opened_at': utc_now()}
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
    if fcntl is None:
        return None
    lock_file = lock_path(workspace, create_folder=False)
    try:
        if path.getsize(lock_file) == 0:
            return None
        with open(lock_file, encoding='utf-8') as stream:
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


def output_root_of(input_root: str) -> str:
    # where the work folders go: cutie_output beside the input root
    return path.join(path.dirname(path.abspath(input_root)), OUTPUT_DIR)


def work_folder(input_root: str, tab: str, sequence: str) -> str:
    return path.join(output_root_of(input_root), tab, sequence)


def ensure_work_folder(input_root: str, source: SequenceInput) -> str:
    """The sequence's work folder, made with its metadata.json (the source section) the first
    time."""
    folder = work_folder(input_root, source.tab, source.sequence)
    if not read_section(folder, SOURCE):
        os.makedirs(folder, exist_ok=True)
        write_sections(folder, source={'input': path.relpath(path.abspath(input_root), folder),
                                       'tab': source.tab, 'sequence': source.sequence})
    return folder


def source_of(workspace: str) -> SequenceInput:
    # the input sequence of a work folder, read again from the input
    source = read_metadata(workspace)[SOURCE]
    input_root = path.normpath(path.join(path.abspath(workspace), source['input']))
    return read_sequence(input_root, source['tab'], source['sequence'])


def category_text(category: Optional[str]) -> str:
    return category or NULL_CATEGORY


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


def summarize_task(source: SequenceInput, workspace: str,
                   frame_count: Optional[int] = None) -> dict:
    """One row of the task browser: the input sequence and its work folder (which need not
    exist yet). frame_count: the frames of its anchor modality, counted here when not given."""
    metadata = read_metadata(workspace)
    anchors = metadata.get(ANCHORS) or {}
    progress = metadata.get(PROGRESS) or {}
    attention = attention_of(workspace, metadata)
    discard = discard_of(workspace, metadata)
    kinds = [anchor['kind'] for anchor in anchors.values()]
    if frame_count is None:
        frame_count = len(frame_files(source.frame_dirs[source.anchor_modality]))
    clips, _ = dataset_intervals(metadata.get(INTERVALS) or {}, frame_count)
    description = saved_description(workspace, metadata) or source.description
    category = category_text(saved_category(workspace, metadata) or source.category)
    return {
        'split': source.tab,
        'sequence': source.sequence,
        'frames': frame_count,
        'mask_anchors': kinds.count('mask'),
        'empty_anchors': kinds.count('empty'),
        # a discarded sequence shows that, else one that needs attention, else its review status
        'status': (DISCARDED_STATUS if discard else
                   ATTENTION_STATUS if attention else status_of(progress)),
        'attention': attention,
        'discard': discard,
        # marked done (also when flagged later; a discard removes the mark), and its clips: the
        # saved intervals, else the whole sequence
        'done': status_of(progress) == 'done',
        'clips': clips,
        'intervals': len(clips),
        'last_edited_by': progress.get('last_edited_by'),
        'annotators': task_annotators(progress, anchors),
        'open_by': lock_holder(workspace),
        'modality': source.anchor_modality,
        'category': category,
        'description': description,
        'target': f'{category}: {description}',
    }


def task_status(workspace: str) -> str:
    # not started / in progress / under review / done
    return status_of(read_section(workspace, PROGRESS, {}))


def submit_for_review(workspace: str, labeler: str) -> None:
    # from any status, including done: the task waits for a new review
    with _metadata_lock:  # one read-modify-write: nothing may change progress in between
        progress = read_section(workspace, PROGRESS, {})
        _write_unlocked(workspace, progress=submitted_progress(progress, labeler, utc_now()))


def mark_done(workspace: str, labeler: str, quit_at: Optional[str] = None) -> None:
    # only for a task under review; quit_at: when the labeling window's Ctrl+Q asked for it
    with _metadata_lock:
        progress = read_section(workspace, PROGRESS, {})
        _write_unlocked(workspace, progress=done_progress(progress, labeler, utc_now(), quit_at))


def discard_task(workspace: str, labeler: str, discarded: bool) -> None:
    """Discard: the sequence is left out of the dataset. Its done mark goes (the dataset's done
    list reads it) and Need Attention is turned off; masks, anchors and the review stay. Restore
    (discarded False) only removes the discard."""
    if not discarded:
        write_sections(workspace, discard=None)
        return
    with _metadata_lock:
        progress = read_section(workspace, PROGRESS)
        _write_unlocked(workspace, discard={'at': utc_now(), 'by': labeler}, attention=None,
                        progress=None if progress is None else discarded_progress(progress))


# the rules of anchors and of the review status, as functions of the previous entry
def next_anchor_entry(previous: Optional[dict], kind: str, edited: bool, labeler: str,
                      now: str) -> Optional[dict]:
    """The anchor entry after a human decision on a frame; None when nothing changes (accepting
    the shown mask of an anchor of the same kind)."""
    if previous is not None and not edited and previous['kind'] == kind:
        return None
    return {'kind': kind, 'updated_at': now, 'by': labeler}


def touched_progress(progress: dict, anchors: Dict[str, dict], labeler: str, now: str) -> dict:
    """The progress after a change: back to in progress (any review or done mark dropped);
    edited_by keeps everyone who has changed the task, in order."""
    editors = task_annotators(progress, anchors)
    if labeler not in editors:
        editors.append(labeler)
    return {'last_edited': now, 'last_edited_by': labeler, 'edited_by': editors}


def submitted_progress(progress: dict, labeler: str, now: str) -> dict:
    """The progress once the task is submitted for review (from any status, including done)."""
    progress = without_done_mark(progress)
    progress['review_at'] = now
    progress['review_by'] = labeler
    return progress


def done_progress(progress: dict, labeler: str, now: str, quit_at: Optional[str] = None) -> dict:
    """The progress once the task is marked done; quit_at: when the labeling window's Ctrl+Q
    (quit and mark done) was pressed, else none is kept."""
    progress = dict(without_done_mark(progress), done_at=now, done_by=labeler)
    if quit_at:
        progress['quit_at'] = quit_at
    return progress


def without_done_mark(progress: dict) -> dict:
    return {key: value for key, value in progress.items() if key not in DONE_FIELDS}


def discarded_progress(progress: dict) -> dict:
    """The progress of a discarded sequence: no done mark."""
    return without_done_mark(progress)


def status_of(progress: dict) -> str:
    if 'done_at' in progress:
        return 'done'
    if 'review_at' in progress:
        return 'under review'
    return 'in progress' if 'last_edited' in progress else 'not started'


class TaskState:

    def __init__(self, workspace: str, labeler: Optional[str] = None):
        self.workspace = workspace
        self.labeler = labeler_name(labeler)
        self.source: SequenceInput = source_of(workspace)
        self.sequence_id: str = self.source.sequence
        self.split: str = self.source.tab
        self.anchor_modality: str = self.source.anchor_modality
        self.frame_dirs: Dict[str, str] = dict(self.source.frame_dirs)
        # the chosen category, else the input's; None: not chosen yet
        self.category: Optional[str] = saved_category(workspace) or self.source.category
        # the ones to choose from: the subset's categories.json (empty when it cannot be read)
        try:
            self.categories: List[Category] = read_categories(self.source.folder)
        except InputError:
            self.categories = []
        # the input's text, and the one shown: the text tuned in Cutie, else the input's
        self.original_description: str = self.source.description
        self.description: str = saved_description(workspace) or self.original_description
        self.anchors: Dict[str, dict] = read_section(workspace, ANCHORS, {})
        # frame name -> {source, path} of a model mask shown on the frame until it has a mask
        self.prefill: Dict[str, dict] = {
            name: {'source': PREGENERATED, 'path': mask_path}
            for name, mask_path in pregenerated_masks(self.source).items()}
        self.attention: Optional[dict] = attention_of(workspace)

    @staticmethod
    def is_task(workspace: Optional[str]) -> bool:
        return workspace is not None and bool(read_section(workspace, SOURCE))

    @property
    def other_modality(self) -> Optional[str]:
        others = [m for m in self.frame_dirs if m != self.anchor_modality]
        return others[0] if others else None

    def anchor(self, name: str) -> Optional[dict]:
        return self.anchors.get(name)

    def clear_mask_references(self) -> None:
        # Reset all: no anchor is left; the pregenerated masks show again (they are input)
        self.anchors.clear()
        self.save()

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
        Cutie, else the whole sequence."""
        return dataset_intervals(self.interval_payload(), frame_count)

    def interval_descriptions(self, frame_count: int) -> Dict[Interval, str]:
        # each interval's referring text (see interval_descriptions)
        return interval_descriptions(self.intervals(frame_count)[0], self.interval_payload(),
                                     self.description)

    def interval_payload(self) -> dict:
        # the intervals section as saved, {} when it is not
        return read_section(self.workspace, INTERVALS, {})

    def added_interval_descriptions(self, frame_count: int) -> Dict[Interval, str]:
        # the texts saved with the intervals (see added_interval_descriptions)
        return added_interval_descriptions(self.intervals(frame_count)[0], self.interval_payload(),
                                           self.description)

    def interval_categories(self, frame_count: int) -> Dict[Interval, Optional[str]]:
        # each interval's category (see interval_categories)
        return interval_categories(self.intervals(frame_count)[0], self.interval_payload(),
                                   self.category)

    def added_interval_categories(self, frame_count: int) -> Dict[Interval, Optional[str]]:
        # the categories saved with the intervals (see added_interval_categories)
        return added_interval_categories(self.intervals(frame_count)[0], self.interval_payload(),
                                         self.category)

    def first_interval(self, frame_count: int) -> Interval:
        # the interval the sequence's description describes
        return first_interval(self.interval_payload()) or self.intervals(frame_count)[0][0]

    def save_intervals(self, intervals: List[Interval],
                       descriptions: Optional[Dict[Interval, str]] = None,
                       categories: Optional[Dict[Interval, str]] = None) -> None:
        write_sections(self.workspace,
                       intervals=intervals_payload(sorted(intervals), descriptions, categories))

    def save_interval(self, start: int, end: int) -> None:
        self.save_intervals([(start, end)])

    def save_description(self, text: str) -> None:
        write_sections(self.workspace,
                       description={'text': text, 'by': self.labeler, 'updated_at': utc_now()})
        self.description = text

    def save_category(self, name: str) -> None:
        write_sections(self.workspace,
                       category={'name': name, 'by': self.labeler, 'updated_at': utc_now()})
        self.category = name

    def save_attention(self, flagged: bool) -> None:
        # not a change of the task: the review status and progress stay as they are
        self.attention = {'at': utc_now(), 'by': self.labeler} if flagged else None
        write_sections(self.workspace, attention=self.attention)

    def save(self) -> None:
        # the anchors, and the change they are (touch), in one write
        self.touch(changes_masks=True, anchors=dict(sorted(self.anchors.items())))

    def touch(self, changes_masks: bool = False, **sections) -> None:
        # a change: back to in progress, dropping any review or done mark; edited_by keeps
        # everyone who has changed the task. changes_masks: of the masks or anchors (the
        # revision goes up). sections: written with it
        with _metadata_lock:
            metadata = read_metadata(self.workspace)
            if changes_masks:
                sections[REVISION] = int(metadata.get(REVISION) or 0) + 1
            _write_unlocked(self.workspace, **sections,
                            progress=touched_progress(metadata.get(PROGRESS) or {}, self.anchors,
                                                      self.labeler, utc_now()))

    def revision(self) -> int:
        """The revision of the masks and anchors, as the file has it now."""
        return int(read_section(self.workspace, REVISION, 0) or 0)
