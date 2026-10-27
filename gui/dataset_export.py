"""
The dataset export (the task browser's Export): every interval of every done sequence, in every
tab, becomes one clip of cutie_export/, beside the input root, laid out as a subset of the input:

    cutie_export/
    ├── visual/<clip>/000001.<ext> ...    the interval's frames, copied, numbered from 1
    ├── thermal/<clip>/...
    ├── annotation/<clip>/000001.png ...  its masks, L 0/255: each frame's saved mask, else its
    │                                     pregenerated mask (shown when it was marked done), else
    │                                     an empty one
    └── seqinfo/<clip>_seqinfo.ini        where the clip comes from, its target and its frames

The clips are numbered from 1 in the browser's order (the tabs, then the sequences by name, then
the intervals by start), zero-padded to the digits of their count: 3000 clips are 0001 ... 3000.
originalSource (<tab>/<sequence>) names the sequence a clip was cut from: clips cut from one
sequence share it, and intervalStart / intervalEnd are frames of that sequence. The export is
built in .cutie_export.partial/ and replaces the previous one only once it is complete.
"""
import configparser
import os
import shutil
from dataclasses import dataclass, field
from os import path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from gui.task_input import (ANNOTATION_DIR, MODALITIES, SEQINFO_DIR, SEQINFO_SUFFIX, InputError,
                            SequenceInput, find_tabs, frame_files, pregenerated_masks, read_tab)
from gui.task_state import (DISCARD, INTERVALS, MASKS_DIR, PROGRESS, TaskLock, TaskLocked,
                            attention_of, check_intervals, dataset_intervals, interval_categories,
                            interval_descriptions, read_metadata, saved_category,
                            saved_description, status_of, work_folder)

EXPORT_DIR = 'cutie_export'  # beside the input root
SECTION = 'Sequence'
FRAME_DIGITS = 6  # a clip's frames: 000001, 000002, ...


class ExportCancelled(Exception):
    """Cancel was pressed: the previous export stays as it was."""


@dataclass
class Clip:
    """One interval of a done sequence: a clip of the dataset."""
    source: SequenceInput
    workspace: str
    start: int  # 1-indexed, inclusive, in the frames of the source sequence
    end: int
    category: Optional[str]
    description: str
    frames: List[str]  # the frame names of the interval, in order

    @property
    def original_source(self) -> str:
        return f'{self.source.tab}/{self.source.sequence}'


@dataclass
class ExportPlan:
    clips: List[Clip] = field(default_factory=list)
    sequences: int = 0  # the done sequences the clips come from
    bytes: int = 0  # the frames to copy
    problems: List[str] = field(default_factory=list)  # stop the export
    warnings: List[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return sum(len(clip.frames) for clip in self.clips)

    def names(self) -> List[str]:
        return clip_names(len(self.clips))


@dataclass
class ExportReport:
    clips: int = 0
    frames: int = 0
    pregenerated: int = 0  # mask frames taken from the pregenerated masks
    empty: int = 0  # mask frames without any mask: exported empty


def export_root_of(input_root: str) -> str:
    return path.join(path.dirname(path.abspath(input_root)), EXPORT_DIR)


def clip_names(total: int) -> List[str]:
    """1 ... total, zero-padded to the digits of total: 12 clips are 01 ... 12."""
    width = len(str(total))
    return [f'{number:0{width}d}' for number in range(1, total + 1)]


def plan_export(input_root: str) -> ExportPlan:
    """The clips of every done sequence; the problems found stop the export before anything is
    written."""
    plan = ExportPlan()
    try:
        tabs = find_tabs(input_root)
    except InputError as exc:
        plan.problems.append(str(exc))
        return plan
    flagged = []
    for tab in tabs:
        contents = read_tab(tab)
        plan.warnings.extend(contents.problems)
        for source in contents.sequences:
            workspace = work_folder(input_root, tab.name, source.sequence)
            metadata = read_metadata(workspace)
            if metadata.get(DISCARD) or status_of(metadata.get(PROGRESS) or {}) != 'done':
                continue
            plan.sequences += 1
            if attention_of(workspace, metadata):
                flagged.append(f'{tab.name}/{source.sequence}')
            plan.clips.extend(sequence_clips(source, workspace, metadata, plan.problems))
    if flagged:
        plan.warnings.append(f'Done but flagged Need Attention (exported): {", ".join(flagged)}.')
    unnamed = [f'{clip.original_source} {clip.start}-{clip.end}' for clip in plan.clips
               if not clip.category]
    if unnamed:
        plan.warnings.append(f'No category (exported with an empty one): {", ".join(unnamed)}.')
    if not plan.problems:
        plan.bytes = sum(path.getsize(path.join(directory, files[name]))
                         for clip in plan.clips
                         for directory, files in frame_listings(clip.source).values()
                         for name in clip.frames)
    return plan


def sequence_clips(source: SequenceInput, workspace: str, metadata: dict,
                   problems: List[str]) -> List[Clip]:
    # a done sequence's intervals, each with its category and description
    listings = frame_listings(source)
    names = list(listings[source.anchor_modality][1])
    payload = metadata.get(INTERVALS) or {}
    intervals, _ = dataset_intervals(payload, len(names))
    sequence = f'{source.tab}/{source.sequence}'
    wrong = check_intervals(intervals, len(names))
    if wrong:
        problems.append(f'{sequence}: {" ".join(wrong)}')
        return []
    descriptions = interval_descriptions(
        intervals, payload, saved_description(workspace, metadata) or source.description)
    categories = interval_categories(
        intervals, payload, saved_category(workspace, metadata) or source.category)
    clips = []
    for start, end in intervals:
        frames = names[start - 1:end]
        for modality, (_, files) in listings.items():
            missing = [name for name in frames if name not in files]
            if missing:
                problems.append(f'{sequence} {start}-{end}: {len(missing)} frame(s) have no '
                                f'{modality} image, e.g. {missing[0]}.')
        clips.append(Clip(source, workspace, start, end, categories[(start, end)],
                          descriptions[(start, end)], frames))
    return clips


def frame_listings(source: SequenceInput) -> Dict[str, Tuple[str, Dict[str, str]]]:
    # modality -> (its folder, frame name -> file name)
    return {modality: (directory, frame_files(directory))
            for modality, directory in source.frame_dirs.items()}


def free_space_problem(plan: ExportPlan, export_root: str) -> Optional[str]:
    # the new export is written before the previous one is removed
    free = shutil.disk_usage(path.dirname(path.abspath(export_root))).free
    if plan.bytes <= free:
        return None
    return (f'The export needs {plan.bytes / 2**30:.1f} GB of frames and {free / 2**30:.1f} GB are '
            f'free: delete the previous export ({export_root}) or make room first.')


def lock_clips(plan: ExportPlan, labeler: str) -> List[TaskLock]:
    """Holds every exported sequence while it is read (Open by shows 'exporting'); raises
    TaskLocked, holding none, when a window has one open."""
    locks = []
    try:
        for workspace in dict.fromkeys(clip.workspace for clip in plan.clips):
            locks.append(TaskLock.acquire(workspace, labeler, activity='exporting'))
    except TaskLocked:
        release_locks(locks)
        raise
    return locks


def release_locks(locks: List[TaskLock]) -> None:
    for lock in locks:
        lock.release()


def write_export(plan: ExportPlan, export_root: str,
                 progress: Callable[[int, int], None] = lambda done, total: None,
                 cancelled: Callable[[], bool] = lambda: False) -> ExportReport:
    """Writes every clip into <export_root>.partial, then puts it in place of the previous
    export. progress(frames written, frames): after each clip. Raises ExportCancelled when
    cancelled() turns true; then, as on any failure, the previous export stays."""
    export_root = path.abspath(export_root)
    parent, name = path.split(export_root)
    partial = path.join(parent, f'.{name}.partial')
    previous = path.join(parent, f'.{name}.previous')
    if path.exists(previous) and not path.exists(export_root):
        os.rename(previous, export_root)  # an export stopped between its two renames
    for leftover in (partial, previous):
        if path.exists(leftover):
            shutil.rmtree(leftover)
    report = ExportReport()
    total = plan.frames
    try:
        for clip, clip_name in zip(plan.clips, plan.names()):
            if cancelled():
                raise ExportCancelled()
            write_clip(clip, clip_name, partial, report)
            progress(report.frames, total)
        os.makedirs(partial, exist_ok=True)  # an export of no clip is an empty folder
        if path.exists(export_root):
            os.rename(export_root, previous)
        os.rename(partial, export_root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        if path.exists(previous) and not path.exists(export_root):
            os.rename(previous, export_root)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return report


def write_clip(clip: Clip, name: str, root: str, report: ExportReport) -> None:
    sizes = {}  # modality -> (extension, width, height) of its first frame
    for modality, (directory, files) in frame_listings(clip.source).items():
        target = path.join(root, modality, name)
        os.makedirs(target)
        for number, frame in enumerate(clip.frames, start=1):
            file_name = files[frame]
            extension = path.splitext(file_name)[1]
            shutil.copy2(path.join(directory, file_name),
                         path.join(target, f'{number:0{FRAME_DIGITS}d}{extension}'))
        first = files[clip.frames[0]]
        with Image.open(path.join(directory, first)) as image:
            sizes[modality] = (path.splitext(first)[1], *image.size)
    _, width, height = sizes[clip.source.anchor_modality]
    write_masks(clip, path.join(root, ANNOTATION_DIR, name), (height, width), report)
    seqinfo_dir = path.join(root, SEQINFO_DIR)
    os.makedirs(seqinfo_dir, exist_ok=True)
    with open(path.join(seqinfo_dir, f'{name}{SEQINFO_SUFFIX}'), 'w', encoding='utf-8') as stream:
        stream.write(seqinfo_text(name, clip, sizes, source_dataset(clip.source)))
    report.clips += 1
    report.frames += len(clip.frames)


def write_masks(clip: Clip, target: str, shape: Tuple[int, int], report: ExportReport) -> None:
    # each frame's saved mask, else its pregenerated one, else an empty one; as L 0/255
    os.makedirs(target)
    saved_dir = path.join(clip.workspace, MASKS_DIR)
    pregenerated = pregenerated_masks(clip.source)
    for number, frame in enumerate(clip.frames, start=1):
        saved = path.join(saved_dir, frame + '.png')
        if path.exists(saved):
            mask = binary_mask(saved)
        elif frame in pregenerated:
            mask = binary_mask(pregenerated[frame])
            report.pregenerated += 1
        else:
            mask = np.zeros(shape, dtype=bool)
            report.empty += 1
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            path.join(target, f'{number:0{FRAME_DIGITS}d}.png'))


def binary_mask(file_path: str) -> np.ndarray:
    # the target: every nonzero pixel (an index mask, or a 0/255 one)
    with Image.open(file_path) as image:
        mask = np.array(image)
    return (mask > 0).any(axis=-1) if mask.ndim == 3 else mask > 0


def source_dataset(source: SequenceInput) -> str:
    # the dataset named by the input's own seqinfo of the sequence, when its subset has one
    file_path = path.join(source.folder, SEQINFO_DIR, f'{source.sequence}{SEQINFO_SUFFIX}')
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with open(file_path, encoding='utf-8') as stream:
            parser.read_file(stream)
        return parser.get(SECTION, 'dataset', fallback='').strip()
    except (OSError, UnicodeDecodeError, configparser.Error):
        return ''


def seqinfo_text(name: str, clip: Clip, sizes: Dict[str, Tuple[str, int, int]],
                 dataset: str = '') -> str:
    """The clip's seqinfo, in the format of the RGBT_VOS seqinfo files; a modality the sequence
    lacks has no keys."""
    entries = [('name', name), ('dataset', dataset), ('originalSource', clip.original_source),
               ('sourceLsId', clip.source.sequence), ('category', clip.category or ''),
               ('description', ' '.join(clip.description.splitlines())), ('problem', ''),
               ('seqLength', len(clip.frames)), ('intervalStart', clip.start),
               ('intervalEnd', clip.end)]
    modalities = [modality for modality in MODALITIES if modality in sizes]
    entries += [(f'imExt{modality.capitalize()}', sizes[modality][0]) for modality in modalities]
    for modality in modalities:
        entries += [(f'imWidth{modality.capitalize()}', sizes[modality][1]),
                    (f'imHeight{modality.capitalize()}', sizes[modality][2])]
    return f'[{SECTION}]\n' + ''.join(f'{key}={value}\n' for key, value in entries)
