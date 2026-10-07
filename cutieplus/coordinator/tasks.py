"""
Tasks in the shared service: their committed state (SQLite rows and immutable mask objects), the
legacy files kept as views, leases (one editing session per task, with an increasing epoch, plus
the original per-task file lock so the original labeling window and the service never write the
same task), revisions, and the commit protocol of every change of official annotation data:

1. encode the mask and publish it as an immutable object (unique temporary file, fsync, rename);
2. in one short database transaction: replay a known request id, validate the lease and the
   revisions, update the mask reference, the anchor, the edit history and the revisions;
3. after the commit, update the legacy files and record their size and modification time;
4. only then answer the request.

No GPU work happens here and no transaction is held during inference.
"""
import io
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from os import path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from cutie.utils.palette import davis_palette
from cutieplus.engine import policies
from cutieplus.engine.images import read_mask_source
from cutieplus.storage.db import Database, dumps, loads
from cutieplus.storage.objects import ObjectStore
from cutieplus.storage.views import JSON_FILES, TaskViews
from gui.exporter import convert_mask_to_binary
from gui.task_state import (ANCHORS_FILE, DESCRIPTION_FILE, INTERVAL_FILE, PREFILL_FILE,
                            PROGRESS_FILE, TASK_FILE, TaskLock, TaskLocked, intervals_from_payload,
                            load_task_set,
                            intervals_payload, next_anchor_entry, status_of, submitted_progress,
                            touched_progress, utc_now, write_description_text, write_interval_text)

log = logging.getLogger(__name__)


class TaskError(Exception):
    """code: 'invalid' | 'locked' | 'lease_lost' | 'stale' | 'conflict' | 'storage'."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def encode_mask(mask: np.ndarray) -> bytes:
    """The palette PNG the labeling window writes (ResourceManager.save_mask)."""
    image = Image.fromarray(np.ascontiguousarray(mask.astype(np.uint8)))
    image.putpalette(davis_palette)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def decode_mask(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)))


@dataclass
class TaskInfo:
    """What task.json says; does not change while the service runs."""
    task_id: str
    workspace: str
    split: str
    sequence: str
    anchor_modality: str
    frame_dirs: Dict[str, str]
    frame_files: Dict[str, Dict[str, str]]  # modality -> frame name -> file
    names: List[str]
    height: int
    width: int
    curated_interval: Optional[List[int]]
    original_description: str

    @property
    def name_to_ti(self) -> Dict[str, int]:
        return {name: ti for ti, name in enumerate(self.names)}

    def image_path(self, frame: str, modality: Optional[str] = None) -> str:
        """A frame's image; a modality that lacks the frame shows the anchor modality's, as the
        labeling window does."""
        modality = modality or self.anchor_modality
        if frame not in self.frame_files.get(modality, {}):
            modality = self.anchor_modality
        return path.join(self.frame_dirs[modality], self.frame_files[modality][frame])


@dataclass
class LeaseContext:
    """What the service keeps in memory about an open task."""
    task_id: str
    session_id: str
    epoch: int
    labeler: str
    resolution: policies.Resolution
    flock: Optional[TaskLock] = None


@dataclass
class _Changes:
    """What one commit changes, for the views written after it."""
    frames: Dict[str, Tuple[Optional[str], Optional[dict]]] = field(default_factory=dict)
    json_files: Dict[str, object] = field(default_factory=dict)  # file -> payload (None: remove)
    anchors_changed: bool = False


class TaskService:

    def __init__(self, db: Database, objects: ObjectStore, task_root: str,
                 default_internal_size: int = 480, clock: Callable[[], str] = utc_now):
        self.db = db
        self.objects = objects
        self.task_root = path.abspath(task_root)
        self.default_internal_size = default_internal_size
        self.clock = clock
        self.infos: Dict[str, TaskInfo] = {}
        self.locks: Dict[str, threading.RLock] = {}
        self.locks_guard = threading.Lock()
        self.leases: Dict[str, LeaseContext] = {}

    # task identity ------------------------------------------------------------------------
    def info(self, workspace: str) -> TaskInfo:
        workspace = path.abspath(workspace)
        if not workspace.startswith(self.task_root + os.sep):
            raise TaskError('invalid', f'{workspace} is not a task of {self.task_root}')
        task_file = path.join(workspace, TASK_FILE)
        if not path.isfile(task_file):
            raise TaskError('invalid', f'{workspace} has no {TASK_FILE}')
        with open(task_file, encoding='utf-8') as f:
            task = json.load(f)
        task_id = f'{task["split"]}/{task["sequence_id"]}'
        cached = self.infos.get(task_id)
        if cached is not None:
            return cached
        modality = task['anchor_modality']
        frame_dirs = dict(task['frame_dirs'])
        if task.get('frame_files'):
            frame_files = {m: {path.splitext(f)[0]: f for f in files}
                           for m, files in task['frame_files'].items()}
        else:
            from gui.resource_manager import _frame_files
            frame_files = {m: _frame_files(directory) for m, directory in frame_dirs.items()}
        names = list(frame_files[modality])
        height, width = task.get('height'), task.get('width')
        if not height or not width:
            first = path.join(frame_dirs[modality], frame_files[modality][names[0]])
            width, height = Image.open(first).size
        info = TaskInfo(task_id, workspace, task['split'], task['sequence_id'], modality, frame_dirs,
                        frame_files, names, int(height), int(width), task.get('curated_interval'),
                        task.get('description', ''))
        self.infos[task_id] = info
        return info

    def info_by_id(self, task_id: str) -> TaskInfo:
        info = self.infos.get(task_id)
        if info is None:
            split, sequence = task_id.split('/', 1)
            info = self.info(path.join(self.task_root, split, sequence))
        return info

    def lock(self, task_id: str) -> threading.RLock:
        with self.locks_guard:
            return self.locks.setdefault(task_id, threading.RLock())

    # committed state ------------------------------------------------------------------------
    def task_row(self, task_id: str) -> Optional[dict]:
        row = self.db.query_one('SELECT * FROM tasks WHERE task_id = ?', (task_id, ))
        if row is None:
            return None
        row = dict(row)
        row['state'] = loads(row['state'])
        row['view_files'] = loads(row['view_files'])
        return row

    def frames(self, task_id: str) -> Dict[str, dict]:
        """frame -> {'mask_object', 'anchor', 'view_stat', 'updated_revision'} of every frame
        that has a mask or is an anchor."""
        rows = self.db.query('SELECT * FROM frames WHERE task_id = ?', (task_id, ))
        return {row['frame']: {'mask_object': row['mask_object'], 'anchor': loads(row['anchor']),
                               'view_stat': loads(row['view_stat']),
                               'updated_revision': row['updated_revision']} for row in rows}

    def anchors(self, task_id: str) -> Dict[str, dict]:
        return {frame: entry['anchor'] for frame, entry in self.frames(task_id).items()
                if entry['anchor'] is not None}

    def mask_source(self, info: TaskInfo, frame: str, frames: Optional[dict] = None,
                    state: Optional[dict] = None) -> Optional[dict]:
        """Where a frame's mask comes from: its committed mask, else its prefill (binary), else
        nothing (the labeling window's fallback)."""
        frames = self.frames(info.task_id) if frames is None else frames
        entry = frames.get(frame)
        if entry is not None and entry['mask_object'] is not None:
            return {'path': self.objects.path(entry['mask_object'])}
        state = self.task_row(info.task_id)['state'] if state is None else state
        prefill = state['prefill'].get(frame)
        if prefill is not None:
            return {'path': prefill['path'], 'binary': True}
        return None

    def mask(self, info: TaskInfo, frame: str) -> Optional[np.ndarray]:
        return read_mask_source(self.mask_source(info, frame))

    # import, external changes, views -------------------------------------------------------
    def import_task(self, info: TaskInfo, reason: str) -> None:
        """Reads the legacy files into the database: the files are authoritative."""
        views = TaskViews(info.workspace)
        anchors = views.read_json(ANCHORS_FILE, {}) or {}
        state = {
            'prefill': views.read_json(PREFILL_FILE, {}) or {},
            'interval': views.read_json(INTERVAL_FILE),
            'description': views.read_json(DESCRIPTION_FILE),
            'progress': views.read_json(PROGRESS_FILE, {}) or {},
        }
        mask_stats = views.scan_masks()
        known = set(info.names)
        objects = {}
        for frame in mask_stats:
            if frame in known:
                with open(views.mask_path(frame), 'rb') as f:
                    objects[frame] = self.objects.put(f.read())
        view_files = dict(views.json_stats())
        view_files['masks_other'] = {frame: stat for frame, stat in mask_stats.items()
                                     if frame not in known}
        now = self.clock()
        with self.db.transaction() as conn:
            row = conn.execute('SELECT * FROM tasks WHERE task_id = ?', (info.task_id, )).fetchone()
            input_revision = (row['input_revision'] + 1) if row else 1
            content_revision = (row['content_revision'] + 1) if row else 1
            epoch = row['lease_epoch'] if row else 0
            conn.execute('DELETE FROM tasks WHERE task_id = ?', (info.task_id, ))
            conn.execute('INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                         (info.task_id, info.workspace, input_revision, content_revision,
                          content_revision, epoch, dumps(state), dumps(view_files), now, now))
            conn.execute('DELETE FROM frames WHERE task_id = ?', (info.task_id, ))
            for frame in sorted(set(objects) | (set(anchors) & known)):
                conn.execute('INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?)',
                             (info.task_id, frame, objects.get(frame), dumps(anchors.get(frame)),
                              dumps(mask_stats.get(frame)), content_revision))
            conn.execute('INSERT INTO edits (task_id, op, at, after_anchor, input_revision, '
                         'content_revision) VALUES (?, ?, ?, ?, ?, ?)',
                         (info.task_id, 'import', now, dumps({'reason': reason}), input_revision,
                          content_revision))
            conn.execute('INSERT INTO events (at, kind, detail) VALUES (?, ?, ?)',
                         (now, 'import', dumps({'task_id': info.task_id, 'reason': reason})))
        log.info('imported %s (%s)', info.task_id, reason)

    def external_changes(self, info: TaskInfo) -> List[str]:
        """Differences between the legacy files and what the service last wrote to them."""
        row = self.task_row(info.task_id)
        views = TaskViews(info.workspace)
        changes = []
        recorded = row['view_files']
        for name, stat in views.json_stats().items():
            if stat != recorded.get(name):
                changes.append(f'{name} changed')
        expected = {frame: entry['view_stat'] for frame, entry in self.frames(info.task_id).items()
                    if entry['view_stat'] is not None}
        expected.update(recorded.get('masks_other', {}))
        current = views.scan_masks()
        for frame in sorted(set(expected) | set(current)):
            if expected.get(frame) != current.get(frame):
                changes.append(f'masks/{frame}.png changed')
                if len(changes) > 8:
                    break
        return changes

    def sync_views(self, info: TaskInfo, full: bool = False) -> int:
        """Writes the legacy files from the committed state (after a crash between a commit and
        its view update); frames changed since the files were last written, or all with full.
        Returns the number of files written."""
        row = self.task_row(info.task_id)
        frames = self.frames(info.task_id)
        views = TaskViews(info.workspace)
        written = 0
        stats = {}
        on_disk = views.scan_masks()
        for frame, entry in frames.items():
            if full or entry['updated_revision'] > row['view_revision'] or \
                    entry['view_stat'] != on_disk.get(frame):
                data = self.objects.get(entry['mask_object']) if entry['mask_object'] else None
                stats[frame] = views.write_mask(frame, data)
                written += 1
        for frame in set(on_disk) - set(frames):
            if frame in info.name_to_ti:
                stats[frame] = views.write_mask(frame, None)
                written += 1
        json_stats = self._write_json_views(info, row['state'], frames, JSON_FILES)
        written += len(json_stats)
        self._record_views(info.task_id, stats, json_stats, row['content_revision'])
        return written

    def _json_payloads(self, info: TaskInfo, state: dict, frames: Dict[str, dict]) -> dict:
        anchors = {frame: entry['anchor'] for frame, entry in sorted(frames.items())
                   if entry['anchor'] is not None}
        return {
            ANCHORS_FILE: anchors,
            PREFILL_FILE: state['prefill'],
            PROGRESS_FILE: state['progress'] or None,
            INTERVAL_FILE: state['interval'],
            DESCRIPTION_FILE: state['description'],
        }

    def _write_json_views(self, info: TaskInfo, state: dict, frames: Dict[str, dict],
                          names) -> Dict[str, Optional[list]]:
        views = TaskViews(info.workspace)
        payloads = self._json_payloads(info, state, frames)
        return {name: views.write_json(name, payloads[name]) for name in names}

    def _record_views(self, task_id: str, mask_stats: Dict[str, Optional[list]],
                      json_stats: Dict[str, Optional[list]], revision: int) -> None:
        with self.db.transaction() as conn:
            for frame, stat in mask_stats.items():
                conn.execute('UPDATE frames SET view_stat = ? WHERE task_id = ? AND frame = ?',
                             (dumps(stat), task_id, frame))
            row = conn.execute('SELECT view_files, view_revision FROM tasks WHERE task_id = ?',
                               (task_id, )).fetchone()
            view_files = loads(row['view_files'])
            view_files.update(json_stats)
            conn.execute('UPDATE tasks SET view_files = ?, view_revision = MAX(view_revision, ?) '
                         'WHERE task_id = ?', (dumps(view_files), revision, task_id))

    def ensure_current(self, info: TaskInfo) -> str:
        """Before the service writes a task again: import it the first time; re-import it when
        another program changed its files since; finish an interrupted view update."""
        with self.lock(info.task_id):
            row = self.task_row(info.task_id)
            if row is None:
                self.import_task(info, 'first use by the service')
                return 'imported'
            changes = self.external_changes(info)
            if changes:
                if row['view_revision'] < row['content_revision']:
                    log.warning('%s: changed outside the service while service commits were not '
                                'yet in its files; the files win (the commits stay in the edit '
                                'history)', info.task_id)
                self.import_task(info, 'changed outside the service: ' + ', '.join(changes[:3]))
                return 'reimported'
            if row['view_revision'] < row['content_revision']:
                self.sync_views(info)
                return 'resynced'
            return 'current'

    # leases ------------------------------------------------------------------------------------
    def _session(self, conn, session_id: str):
        return conn.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id, )).fetchone()

    def open_task(self, workspace: str, session_id: str) -> dict:
        """The lease of a task for a session; raises TaskError('locked') when another live
        session or another program has it. A session reopening its own task keeps its lease."""
        info = self.info(workspace)
        with self.lock(info.task_id):
            session = self.db.query_one('SELECT * FROM sessions WHERE session_id = ?', (session_id, ))
            if session is None or session['status'] != 'active':
                raise TaskError('lease_lost', 'the session is not active; reconnect first')
            lease = self.db.query_one('SELECT * FROM leases WHERE task_id = ?', (info.task_id, ))
            context = self.leases.get(info.task_id)
            if lease is not None and lease['status'] == 'active':
                if lease['session_id'] == session_id and context is not None:
                    return self._open_reply(info, context, 'reopened')
                holder = self.db.query_one('SELECT * FROM sessions WHERE session_id = ?',
                                           (lease['session_id'], ))
                if lease['session_id'] != session_id and holder is not None and \
                        holder['status'] == 'active':
                    raise TaskError('locked', f'{info.sequence} is open by '
                                    f'{holder["annotator_name"]} ({holder["vnc_display"] or "?"})',
                                    holder={'labeler': holder['annotator_name'],
                                            'display': holder['vnc_display'],
                                            'pid': holder['process_id']})
            # a lease of a session that is gone (or of this session, before a restart of the
            # service) is taken over; its file lock is given up first
            old = self.leases.pop(info.task_id, None)
            if old is not None and old.flock is not None:
                old.flock.release()
            try:
                flock = TaskLock.acquire(info.workspace, session['annotator_name'], holder={
                    'display': session['vnc_display'] or '', 'pid': session['process_id'],
                    'host': session['host'], 'via': 'CutiePlus service'})
            except TaskLocked as exc:
                raise TaskError('locked', str(exc), holder=exc.holder) from None
            try:
                sync = self.ensure_current(info)
                now = self.clock()
                with self.db.transaction() as conn:
                    epoch = conn.execute('SELECT lease_epoch FROM tasks WHERE task_id = ?',
                                         (info.task_id, )).fetchone()['lease_epoch'] + 1
                    conn.execute('UPDATE tasks SET lease_epoch = ? WHERE task_id = ?',
                                 (epoch, info.task_id))
                    conn.execute('INSERT OR REPLACE INTO leases VALUES (?, ?, ?, ?, ?, ?)',
                                 (info.task_id, session_id, epoch, now, now, 'active'))
                resolution = self.resolution(info)
            except BaseException:
                flock.release()
                raise
            context = LeaseContext(info.task_id, session_id, epoch, session['annotator_name'],
                                   resolution, flock)
            self.leases[info.task_id] = context
            return self._open_reply(info, context, sync)

    def _open_reply(self, info: TaskInfo, context: LeaseContext, sync: str) -> dict:
        row = self.task_row(info.task_id)
        return {'task_id': info.task_id, 'lease_epoch': context.epoch,
                'input_revision': row['input_revision'],
                'content_revision': row['content_revision'], 'sync': sync,
                'resolution': {'max_internal_size': context.resolution.max_internal_size,
                               'upscale_to_internal_size': context.resolution.upscale_to_internal_size,
                               'message': context.resolution.message()}}

    def resolution(self, info: TaskInfo) -> policies.Resolution:
        """The small-target policy, from the committed masks (as the window decides at open)."""
        row = self.task_row(info.task_id)
        frames = self.frames(info.task_id)
        anchors = {frame: entry['anchor'] for frame, entry in frames.items()
                   if entry['anchor'] is not None}
        names = policies.target_area_frames(anchors, row['state']['prefill'])
        known = info.name_to_ti
        area = policies.median_target_area(
            names, lambda name: read_mask_source(self.mask_source(info, name, frames, row['state']))
            if name in known else None)
        return policies.choose_resolution(area, info.height, info.width,
                                          self.default_internal_size)

    def close_task(self, task_id: str, session_id: str, epoch: int, status: str = 'released') -> bool:
        with self.lock(task_id):
            context = self.leases.get(task_id)
            if context is None or context.session_id != session_id or context.epoch != epoch:
                return False
            with self.db.transaction() as conn:
                conn.execute('UPDATE leases SET status = ?, renewed_at = ? WHERE task_id = ? AND '
                             'session_id = ? AND epoch = ?',
                             (status, self.clock(), task_id, session_id, epoch))
            self.leases.pop(task_id, None)
            if context.flock is not None:
                context.flock.release()
            return True

    def leases_of(self, session_id: str) -> List[LeaseContext]:
        return [context for context in self.leases.values() if context.session_id == session_id]

    def check_lease(self, conn, task_id: str, session_id: str, epoch: int) -> None:
        lease = conn.execute('SELECT * FROM leases WHERE task_id = ?', (task_id, )).fetchone()
        if lease is None or lease['status'] != 'active' or lease['session_id'] != session_id \
                or lease['epoch'] != epoch:
            raise TaskError('lease_lost', 'this window no longer holds the task (its lease '
                            'expired or another window opened it); open it again')
        session = self._session(conn, session_id)
        if session is None or session['status'] != 'active':
            raise TaskError('lease_lost', 'the session expired; open the task again')

    def context(self, task_id: str, session_id: str, epoch: int) -> LeaseContext:
        context = self.leases.get(task_id)
        if context is None or context.session_id != session_id or context.epoch != epoch:
            raise TaskError('lease_lost', 'this window no longer holds the task; open it again')
        return context

    # commits -------------------------------------------------------------------------------
    def _stored_response(self, conn, request_id: Optional[str]) -> Optional[dict]:
        if request_id is None:
            return None
        row = conn.execute('SELECT response FROM requests WHERE request_id = ?',
                           (request_id, )).fetchone()
        return None if row is None else loads(row['response'])

    def _bump(self, conn, task_id: str, inputs: bool) -> Tuple[int, int]:
        row = conn.execute('SELECT input_revision, content_revision FROM tasks WHERE task_id = ?',
                           (task_id, )).fetchone()
        input_revision = row['input_revision'] + (1 if inputs else 0)
        content_revision = row['content_revision'] + 1
        conn.execute('UPDATE tasks SET input_revision = ?, content_revision = ?, updated_at = ? '
                     'WHERE task_id = ?', (input_revision, content_revision, self.clock(), task_id))
        return input_revision, content_revision

    def _state(self, conn, task_id: str) -> dict:
        return loads(conn.execute('SELECT state FROM tasks WHERE task_id = ?',
                                  (task_id, )).fetchone()['state'])

    def _set_state(self, conn, task_id: str, state: dict) -> None:
        conn.execute('UPDATE tasks SET state = ? WHERE task_id = ?', (dumps(state), task_id))

    def _all_anchors(self, conn, task_id: str) -> Dict[str, dict]:
        rows = conn.execute('SELECT frame, anchor FROM frames WHERE task_id = ? AND anchor IS NOT '
                            'NULL', (task_id, )).fetchall()
        return {row['frame']: loads(row['anchor']) for row in rows}

    def _touch(self, conn, task_id: str, labeler: str, changes: _Changes) -> None:
        state = self._state(conn, task_id)
        state['progress'] = touched_progress(state['progress'], self._all_anchors(conn, task_id),
                                             labeler, self.clock())
        self._set_state(conn, task_id, state)
        changes.json_files[PROGRESS_FILE] = state['progress']

    def _frame_row(self, conn, task_id: str, frame: str) -> Tuple[Optional[str], Optional[dict]]:
        row = conn.execute('SELECT mask_object, anchor FROM frames WHERE task_id = ? AND frame = ?',
                           (task_id, frame)).fetchone()
        return (None, None) if row is None else (row['mask_object'], loads(row['anchor']))

    def _set_frame(self, conn, task_id: str, frame: str, mask_object: Optional[str],
                   anchor: Optional[dict], revision: int, changes: _Changes) -> None:
        if mask_object is None and anchor is None:
            conn.execute('DELETE FROM frames WHERE task_id = ? AND frame = ?', (task_id, frame))
        else:
            conn.execute('INSERT INTO frames (task_id, frame, mask_object, anchor, view_stat, '
                         'updated_revision) VALUES (?, ?, ?, ?, NULL, ?) ON CONFLICT (task_id, '
                         'frame) DO UPDATE SET mask_object = excluded.mask_object, anchor = '
                         'excluded.anchor, updated_revision = excluded.updated_revision',
                         (task_id, frame, mask_object, dumps(anchor), revision))
        changes.frames[frame] = (mask_object, anchor)

    def _finish(self, conn, request_id: Optional[str], session_id: Optional[str], op: str,
                response: dict) -> dict:
        if request_id is not None:
            conn.execute('INSERT INTO requests VALUES (?, ?, ?, ?, ?)',
                         (request_id, session_id, op, self.clock(), dumps(response)))
        return response

    def _write_views(self, info: TaskInfo, changes: _Changes, revision: int) -> bool:
        """After a commit: the legacy files of what changed. Returns False when they could not be
        written (the commit stands; the files are rewritten at the next start or open)."""
        try:
            views = TaskViews(info.workspace)
            mask_stats = {}
            for frame, (mask_object, _) in changes.frames.items():
                data = self.objects.get(mask_object) if mask_object else None
                mask_stats[frame] = views.write_mask(frame, data)
            names = set(changes.json_files)
            if changes.anchors_changed:
                names.add(ANCHORS_FILE)
            json_stats = {}
            if names:
                row = self.task_row(info.task_id)
                payloads = self._json_payloads(info, row['state'], self.frames(info.task_id))
                for name in names:
                    json_stats[name] = views.write_json(name, payloads[name])
            self._record_views(info.task_id, mask_stats, json_stats, revision)
            return True
        except OSError:
            log.exception('%s: committed, but its legacy files were not updated', info.task_id)
            return False

    def commit_frame(self, task_id: str, session_id: str, epoch: int, frame: str, *,
                     mask: Optional[np.ndarray] = None, mask_bytes: Optional[bytes] = None,
                     remove_mask: bool = False, anchor: Optional[str] = None,
                     restore_anchor: Optional[dict] = None, request_id: Optional[str] = None,
                     op: str = 'edit') -> dict:
        """A human change of one frame.
        mask / mask_bytes: the frame's new mask (array, or PNG bytes kept as they are);
        remove_mask: the frame has no mask file afterwards.
        anchor: 'edit' (a human edit makes the frame an anchor), 'accept' (accept the shown mask
        as an anchor), 'remove', 'restore' (restore_anchor, None for no anchor), or None."""
        info = self.info_by_id(task_id)
        if frame not in info.name_to_ti:
            raise TaskError('invalid', f'{task_id} has no frame {frame}')
        data = mask_bytes if mask_bytes is not None else (encode_mask(mask) if mask is not None
                                                           else None)
        with self.lock(task_id):
            sha = self.objects.put(data) if data is not None else None  # published before the DB
            changes = _Changes()
            with self.db.transaction() as conn:
                stored = self._stored_response(conn, request_id)
                if stored is not None:
                    return dict(stored, replayed=True)
                self.check_lease(conn, task_id, session_id, epoch)
                labeler = self._session(conn, session_id)['annotator_name']
                before_object, before_anchor = self._frame_row(conn, task_id, frame)
                after_object = sha if data is not None else (None if remove_mask else before_object)
                if anchor in ('edit', 'accept'):
                    if mask is not None:
                        shown = mask
                    else:
                        shown = self.objects.mask(after_object) if after_object else None
                    kind = 'mask' if shown is not None and (shown > 0).any() else 'empty'
                    entry = next_anchor_entry(before_anchor, kind, anchor == 'edit', labeler,
                                              self.clock())
                    after_anchor = before_anchor if entry is None else entry
                elif anchor == 'remove':
                    after_anchor = None
                elif anchor == 'restore':
                    after_anchor = None if restore_anchor is None else dict(restore_anchor)
                else:
                    after_anchor = before_anchor
                if after_object == before_object and after_anchor == before_anchor and \
                        anchor != 'edit':
                    # nothing changes (e.g. C on a frame that is already this anchor)
                    response = {'frame': frame, 'mask_object': after_object,
                                'anchor': after_anchor, 'changed': False,
                                **self._revisions(conn, task_id)}
                    return self._finish(conn, request_id, session_id, op, response)
                input_revision, content_revision = self._bump(conn, task_id, inputs=True)
                self._set_frame(conn, task_id, frame, after_object, after_anchor, content_revision,
                                changes)
                changes.anchors_changed = after_anchor != before_anchor
                self._touch(conn, task_id, labeler, changes)
                conn.execute('INSERT INTO edits (task_id, frame, op, request_id, session_id, '
                             'author, at, before_object, after_object, before_anchor, after_anchor, '
                             'input_revision, content_revision) VALUES '
                             '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (task_id, frame, op, request_id, session_id, labeler, self.clock(),
                              before_object, after_object, dumps(before_anchor),
                              dumps(after_anchor), input_revision, content_revision))
                response = {'frame': frame, 'mask_object': after_object, 'anchor': after_anchor,
                            'changed': True, 'input_revision': input_revision,
                            'content_revision': content_revision}
                self._finish(conn, request_id, session_id, op, response)
            response['views_written'] = self._write_views(info, changes, content_revision)
            return response

    def _revisions(self, conn, task_id: str) -> dict:
        row = conn.execute('SELECT input_revision, content_revision FROM tasks WHERE task_id = ?',
                           (task_id, )).fetchone()
        return {'input_revision': row['input_revision'],
                'content_revision': row['content_revision']}

    def commit_propagated(self, task_id: str, session_id: str, epoch: int, input_revision: int,
                          frame: str, mask: np.ndarray, job_id: str, attempt_id: str,
                          is_current_attempt: Callable[[], bool]) -> dict:
        """A mask written by propagation. Rejected (TaskError 'stale') when the lease, the task's
        human inputs or the job attempt are no longer the ones the result was computed from, or
        when the frame became an anchor: a propagated mask never replaces a human one."""
        info = self.info_by_id(task_id)
        data = encode_mask(mask)
        with self.lock(task_id):
            sha = self.objects.put(data)
            changes = _Changes()
            with self.db.transaction() as conn:
                self.check_lease(conn, task_id, session_id, epoch)
                revisions = self._revisions(conn, task_id)
                if revisions['input_revision'] != input_revision:
                    raise TaskError('stale', f'the task\'s human inputs changed (revision '
                                    f'{revisions["input_revision"]}, the job used {input_revision})')
                if not is_current_attempt():
                    raise TaskError('stale', f'attempt {attempt_id} of {job_id} is obsolete')
                before_object, before_anchor = self._frame_row(conn, task_id, frame)
                if before_anchor is not None:
                    raise TaskError('stale', f'{frame} is an anchor; propagation never writes it')
                labeler = self._session(conn, session_id)['annotator_name']
                _, content_revision = self._bump(conn, task_id, inputs=False)
                self._set_frame(conn, task_id, frame, sha, None, content_revision, changes)
                self._touch(conn, task_id, labeler, changes)
                conn.execute('INSERT INTO edits (task_id, frame, op, session_id, author, at, '
                             'before_object, after_object, input_revision, content_revision, '
                             'job_id, attempt_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (task_id, frame, 'propagate', session_id, labeler, self.clock(),
                              before_object, sha, input_revision, content_revision, job_id,
                              attempt_id))
            views_written = self._write_views(info, changes, content_revision)
            return {'frame': frame, 'mask_object': sha, 'content_revision': content_revision,
                    'views_written': views_written}

    def _task_change(self, task_id: str, session_id: str, epoch: int, request_id: Optional[str],
                     op: str, mutate: Callable, inputs: bool = False, touch: bool = True) -> dict:
        """A change of task-level data (intervals, description, review, reset)."""
        info = self.info_by_id(task_id)
        with self.lock(task_id):
            changes = _Changes()
            with self.db.transaction() as conn:
                stored = self._stored_response(conn, request_id)
                if stored is not None:
                    return dict(stored, replayed=True)
                self.check_lease(conn, task_id, session_id, epoch)
                labeler = self._session(conn, session_id)['annotator_name']
                state = self._state(conn, task_id)
                result = mutate(conn, state, labeler, changes)
                if touch:
                    input_revision, content_revision = self._bump(conn, task_id, inputs)
                else:
                    revisions = self._revisions(conn, task_id)
                    input_revision = revisions['input_revision']
                    content_revision = revisions['content_revision']
                self._set_state(conn, task_id, state)
                if touch:
                    self._touch(conn, task_id, labeler, changes)
                conn.execute('INSERT INTO edits (task_id, op, request_id, session_id, author, at, '
                             'after_anchor, input_revision, content_revision) VALUES '
                             '(?, ?, ?, ?, ?, ?, ?, ?, ?)',
                             (task_id, op, request_id, session_id, labeler, self.clock(),
                              dumps(result), input_revision, content_revision))
                response = dict(result or {}, input_revision=input_revision,
                                content_revision=content_revision)
                self._finish(conn, request_id, session_id, op, response)
            response['views_written'] = self._write_views(info, changes, content_revision)
            return response

    def set_intervals(self, task_id, session_id, epoch, intervals, request_id=None) -> dict:
        intervals = sorted((int(start), int(end)) for start, end in intervals)

        def mutate(conn, state, labeler, changes):
            state['interval'] = intervals_payload(intervals)
            changes.json_files[INTERVAL_FILE] = state['interval']
            return {'intervals': intervals}

        return self._task_change(task_id, session_id, epoch, request_id, 'intervals', mutate)

    def set_description(self, task_id, session_id, epoch, text, request_id=None) -> dict:
        text = text.strip()
        if not text:
            raise TaskError('invalid', 'the description cannot be empty')

        def mutate(conn, state, labeler, changes):
            state['description'] = {'text': text, 'by': labeler, 'updated_at': self.clock()}
            changes.json_files[DESCRIPTION_FILE] = state['description']
            return {'description': text}

        return self._task_change(task_id, session_id, epoch, request_id, 'description', mutate)

    def intervals(self, info: TaskInfo, state: dict) -> List[tuple]:
        """As TaskState.intervals: set in Cutie, else curated, else the whole sequence."""
        if state['interval']:
            return intervals_from_payload(state['interval'])
        if info.curated_interval:
            return [(info.curated_interval[0], info.curated_interval[1])]
        return [(1, len(info.names))]

    def description(self, info: TaskInfo, state: dict) -> str:
        saved = (state['description'] or {}).get('text')
        return saved or info.original_description

    def submit_for_review(self, task_id, session_id, epoch, target_objects,
                          request_id=None) -> dict:
        """Save binary PNGs: binary masks of the committed masks, the intervals and description
        beside them, and the task under review at its current content revision."""
        info = self.info_by_id(task_id)
        with self.lock(task_id):
            row = self.task_row(task_id)
            if row['view_revision'] < row['content_revision']:
                self.sync_views(info)
            intervals = self.intervals(info, row['state'])
            folder = path.join(info.workspace, 'binary_masks')
            os.makedirs(folder, exist_ok=True)
            convert_mask_to_binary(path.join(info.workspace, 'masks'), folder, list(target_objects))
            write_interval_text(folder, intervals)
            write_description_text(folder, self.description(info, row['state']))

            def mutate(conn, state, labeler, changes):
                # the intervals are saved with every submission, as in the window
                state['interval'] = intervals_payload(intervals)
                changes.json_files[INTERVAL_FILE] = state['interval']
                state['progress'] = submitted_progress(state['progress'], labeler, self.clock())
                revision = self._revisions(conn, task_id)['content_revision']
                state['review_revision'] = revision
                changes.json_files[PROGRESS_FILE] = state['progress']
                return {'folder': folder, 'intervals': intervals, 'review_revision': revision}

            return self._task_change(task_id, session_id, epoch, request_id, 'submit_for_review',
                                     mutate, touch=False)

    def reset_all(self, task_id, session_id, epoch, export_dirs, request_id=None) -> dict:
        """Reset all frame masks: every mask, anchor and prefill reference, the binary and soft
        masks, and the task's export trees. Not undoable (as in the window)."""
        info = self.info_by_id(task_id)

        def mutate(conn, state, labeler, changes):
            frames = conn.execute('SELECT frame FROM frames WHERE task_id = ?',
                                  (task_id, )).fetchall()
            for row in frames:
                changes.frames[row['frame']] = (None, None)
            conn.execute('DELETE FROM frames WHERE task_id = ?', (task_id, ))
            state['prefill'] = {}
            changes.json_files[PREFILL_FILE] = {}
            changes.anchors_changed = True
            return {'frames_cleared': len(frames)}

        with self.lock(task_id):
            response = self._task_change(task_id, session_id, epoch, request_id, 'reset_all', mutate,
                                         inputs=True)
            if response.get('replayed'):
                return response
            removed = 0
            for directory in (path.join(info.workspace, 'binary_masks'),
                              path.join(info.workspace, 'soft_masks'),
                              path.join(info.workspace, 'masks')):
                for root, _, files in os.walk(directory):
                    for name in files:
                        if name.lower().endswith('.png'):
                            os.remove(path.join(root, name))
                            removed += 1
            removed_exports = 0
            for directory in export_dirs:
                if path.isdir(directory):
                    shutil.rmtree(directory)
                    removed_exports += 1
            response.update(removed=removed, removed_exports=removed_exports)
            return response

    def status(self, task_id: str) -> str:
        return status_of(self.task_row(task_id)['state']['progress'])

    # export and review ----------------------------------------------------------------------
    def export_tree(self, info: TaskInfo) -> str:
        task_set = load_task_set(self.task_root)
        return task_set['export_tree'].format(modality=info.anchor_modality,
                                              sequence=info.sequence)

    def begin_export(self, task_id: str, session_id: str, epoch: int) -> dict:
        """Before the external export command runs: its files are brought up to date and the
        export is recorded with the content revision it exports."""
        info = self.info_by_id(task_id)
        with self.lock(task_id):
            with self.db.transaction() as conn:
                self.check_lease(conn, task_id, session_id, epoch)
            row = self.task_row(task_id)
            if row['view_revision'] < row['content_revision']:
                self.sync_views(info)
            export_id = uuid.uuid4().hex[:16]
            started_at = self.clock()
            with self.db.transaction() as conn:
                conn.execute('INSERT INTO exports VALUES (?, ?, ?, ?, ?, NULL, ?, NULL)',
                             (export_id, task_id, session_id, row['content_revision'], started_at,
                              'running'))
        return {'export_id': export_id, 'content_revision': row['content_revision'],
                'started_at': started_at}

    def finish_export(self, export_id: str, session_id: str, exit_ok: bool) -> dict:
        """After the command: complete only when it succeeded, wrote a new metadata.json, and the
        task did not change meanwhile; failed or stale otherwise."""
        export = self.db.query_one('SELECT * FROM exports WHERE export_id = ?', (export_id, ))
        if export is None or export['session_id'] != session_id:
            raise TaskError('invalid', f'no export {export_id}')
        if export['status'] != 'running':
            return {'status': export['status'], 'detail': export['detail'],
                    'content_revision': export['content_revision']}
        info = self.info_by_id(export['task_id'])
        status, detail = 'complete', None
        if not exit_ok:
            status, detail = 'failed', 'the export command failed'
        else:
            metadata_path = path.join(self.export_tree(info), 'metadata.json')
            try:
                with open(metadata_path, encoding='utf-8') as f:
                    exported_at = json.load(f).get('exported_at')
            except (OSError, ValueError):
                exported_at = None
            if exported_at is None:
                status, detail = 'failed', f'{metadata_path} is missing or unreadable'
            elif exported_at < export['started_at']:
                status, detail = 'failed', f'{metadata_path} is older than this export'
            else:
                current = self.task_row(info.task_id)['content_revision']
                if current != export['content_revision']:
                    status, detail = 'stale', (f'the task changed during the export (revision '
                                               f'{export["content_revision"]} -> {current})')
        with self.db.transaction() as conn:
            conn.execute('UPDATE exports SET finished_at = ?, status = ?, detail = ? WHERE '
                         'export_id = ?', (self.clock(), status, detail, export_id))
        return {'status': status, 'detail': detail,
                'content_revision': export['content_revision']}

    def export_status(self, task_id: str) -> dict:
        row = self.task_row(task_id)
        state = row['state']
        latest = self.db.query_one("SELECT * FROM exports WHERE task_id = ? AND status = "
                                   "'complete' ORDER BY finished_at DESC, rowid DESC LIMIT 1",
                                   (task_id, ))
        status = status_of(state['progress'])
        revision = row['content_revision']
        return {'status': status, 'content_revision': revision,
                'review_revision': state.get('review_revision'),
                'review_current': status == 'under review' and
                state.get('review_revision') == revision,
                'export_revision': latest['content_revision'] if latest else None,
                'export_current': latest is not None and latest['content_revision'] == revision}

    def mark_done(self, task_id: str, session_id: str, epoch: int,
                  request_id: Optional[str] = None) -> dict:
        """Accepts a task under review whose review and export are both of its current
        content."""

        def mutate(conn, state, labeler, changes):
            revision = self._revisions(conn, task_id)['content_revision']
            status = status_of(state['progress'])
            if status != 'under review':
                raise TaskError('invalid', f'the task is {status}; save its binary PNGs to submit '
                                'it for review first')
            if state.get('review_revision') != revision:
                raise TaskError('stale', 'the task changed after it was submitted for review')
            latest = conn.execute("SELECT content_revision FROM exports WHERE task_id = ? AND "
                                  "status = 'complete' ORDER BY finished_at DESC, rowid DESC "
                                  "LIMIT 1", (task_id, )).fetchone()
            if latest is None or latest['content_revision'] != revision:
                raise TaskError('export_needed', 'export the task first: its export is missing or '
                                'older than its content')
            state['progress'] = dict(state['progress'], done_at=self.clock(), done_by=labeler)
            state['done_revision'] = revision
            changes.json_files[PROGRESS_FILE] = state['progress']
            return {'done_revision': revision}

        return self._task_change(task_id, session_id, epoch, request_id, 'mark_done', mutate,
                                 touch=False)
