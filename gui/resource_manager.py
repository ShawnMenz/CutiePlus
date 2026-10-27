import os
from os import path
import shutil
import collections
import logging
from dataclasses import dataclass
from queue import Queue
import threading
from threading import Thread
from omegaconf import DictConfig, open_dict
from typing import Optional, Tuple, Literal, Union
import cv2
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import numpy as np

from cutie.utils.palette import davis_palette
from gui.task_input import frame_files
from gui.task_state import MASKS_DIR, TaskState
from tqdm import tqdm

log = logging.getLogger()


# https://bugs.python.org/issue28178
# ah python ah why
class LRU:

    def __init__(self, func, maxsize=128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize

    def __call__(self, *args):
        cache = self.cache
        if args in cache:
            cache.move_to_end(args)
            return cache[args]
        result = self.func(*args)
        cache[args] = result
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def invalidate(self, key):
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()


@dataclass
class SaveItem:
    type: Literal['mask', 'visualization', 'soft_mask']
    data: Union[Image.Image, np.ndarray]
    name: str = None  # only used for soft_mask


class SaveError(Exception):
    """One or more background saves failed; the message lists the files and the reasons."""


def write_atomically(file_path: str, write) -> None:
    """write(temporary path) writes the whole file; it then replaces file_path in one step, so a
    failure leaves the previous file (or none) rather than half a file."""
    directory, name = path.split(file_path)
    temporary = path.join(directory, f'.{name}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        write(temporary)
        os.replace(temporary, file_path)
    except BaseException:
        if path.exists(temporary):
            os.remove(temporary)
        raise


def imwrite_checked(file_path: str, image: np.ndarray) -> None:
    extension = path.splitext(file_path)[1]

    def write(temporary: str) -> None:
        ok, encoded = cv2.imencode(extension, image)
        if not ok:
            raise OSError(f'cannot encode {file_path}')
        with open(temporary, 'wb') as f:
            f.write(encoded.tobytes())

    write_atomically(file_path, write)


class ResourceManager:

    def __init__(self, cfg: DictConfig):
        # determine inputs
        images = cfg['images']
        video = cfg['video']
        self.workspace = cfg['workspace']
        self.max_size = cfg['max_overall_size']
        self.palette = davis_palette

        # create temporary workspace if not specified
        if self.workspace is None:
            if images is not None:
                basename = path.basename(images)
            elif video is not None:
                basename = path.basename(video)[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            self.workspace = path.join('./workspace', basename)

        print(f'Workspace is in: {self.workspace}')
        with open_dict(cfg):
            cfg['workspace'] = self.workspace

        # task mode reads the frames in place from the input (named in the work folder's
        # metadata.json)
        labeler = cfg['labeler'] if 'labeler' in cfg else None
        self.task = TaskState(self.workspace, labeler) if TaskState.is_task(self.workspace) else None

        # determine the location of input images
        need_decoding = False
        need_resizing = False
        if self.task is not None or path.exists(path.join(self.workspace, 'images')):
            pass
        elif images is not None:
            need_resizing = True
        elif video is not None:
            # will decode video into frames later
            need_decoding = True

        # create workspace subdirectories
        if self.task is not None:
            self.image_dirs = dict(self.task.frame_dirs)
            self.default_modality = self.task.anchor_modality
        else:
            self.image_dirs = {'default': path.join(self.workspace, 'images')}
            self.default_modality = 'default'
            os.makedirs(self.image_dirs['default'], exist_ok=True)
        self.image_dir = self.image_dirs[self.default_modality]
        self.mask_dir = path.join(self.workspace, MASKS_DIR)
        self.visualization_dir = path.join(self.workspace, 'visualization')
        self.soft_mask_dir = path.join(self.workspace, 'soft_masks')
        os.makedirs(self.mask_dir, exist_ok=True)
        # a task's work folder holds masks/ and metadata.json only: the visualizations and soft
        # masks are a plain workspace's
        if self.task is None:
            os.makedirs(self.visualization_dir, exist_ok=True)
            os.makedirs(self.soft_mask_dir, exist_ok=True)

            # create all soft mask sub-directories
            for i in range(1, cfg['num_objects'] + 1):
                os.makedirs(path.join(self.soft_mask_dir, f'{i}'), exist_ok=True)

        # convert read functions to be buffered
        self.get_image = LRU(self._get_image_unbuffered, maxsize=cfg['buffer_size'])
        self.get_mask = LRU(self._get_mask_unbuffered, maxsize=cfg['buffer_size'])

        # extract frames from video
        if need_decoding:
            self._extract_frames(video)

        # copy/resize existing images to the workspace
        if need_resizing:
            self._copy_resize_frames(images)

        # read all frame names; each modality maps a frame name to its file
        self.frame_files = {modality: frame_files(directory)
                            for modality, directory in self.image_dirs.items()}
        self.names = list(self.frame_files[self.default_modality])
        self.name_to_ti = {name: ti for ti, name in enumerate(self.names)}
        self.length = len(self.names)
        for modality, files in self.frame_files.items():
            missing = len(set(self.names) - set(files))
            if missing:
                print(f'{missing} frames have no {modality} image; they show the '
                      f'{self.default_modality} image instead.')

        assert self.length > 0, f'No images found! Check {self.workspace}/images. Remove folder if necessary.'

        print(f'{self.length} images found.')

        self.height, self.width = self.get_image(0).shape[:2]

        # create the saver threads for saving the masks/visualizations; a failed save is kept in
        # save_errors (check_saves reports it) and never stops a thread
        self.save_errors = []
        self.save_errors_lock = threading.Lock()
        self.save_queue = Queue(maxsize=cfg['save_queue_size'])
        self.num_save_threads = cfg['num_save_threads']
        self.save_threads = [
            Thread(target=self.save_thread, args=(self.save_queue, ))
            for _ in range(self.num_save_threads)
        ]
        for t in self.save_threads:
            t.daemon = True
            t.start()
        self.closed = False

    def close(self):
        # finish pending saves and stop the save threads; safe to call more than once
        if getattr(self, 'closed', True):
            return
        self.closed = True
        for _ in range(self.num_save_threads):
            self.save_queue.put(None)
        self.save_queue.join()
        for t in self.save_threads:
            t.join()

    def __del__(self):
        self.close()

    def save_thread(self, queue: Queue):
        while True:
            args: SaveItem = queue.get()
            if args is None:
                queue.task_done()
                break
            try:
                self.save_item(args)
            except Exception as exc:  # reported by check_saves; the thread keeps serving
                log.exception('Saving %s %s failed', args.type, args.name)
                with self.save_errors_lock:
                    self.save_errors.append(f'{args.type} {args.name}: {exc}')
            finally:
                # always, or save_queue.join() would wait forever
                queue.task_done()

    def save_item(self, args: SaveItem) -> None:
        if args.type == 'mask':
            # PIL image
            mask_path = path.join(self.mask_dir, args.name + '.png')
            try:
                write_atomically(mask_path,
                                 lambda temporary: args.data.save(temporary, format='PNG'))
            except Exception as exc:
                raise OSError(f'{path.basename(mask_path)}: {exc}') from exc
        elif args.type.startswith('visualization'):
            # numpy array, save with cv2
            vis_mode = args.type.split('_')[-1]
            os.makedirs(path.join(self.visualization_dir, vis_mode), exist_ok=True)
            if vis_mode == 'rgba':
                data = cv2.cvtColor(args.data, cv2.COLOR_RGBA2BGRA).copy()
                imwrite_checked(path.join(self.visualization_dir, vis_mode, args.name + '.png'),
                                data)
            else:
                data = cv2.cvtColor(args.data, cv2.COLOR_RGB2BGR)
                imwrite_checked(path.join(self.visualization_dir, vis_mode, args.name + '.jpg'),
                                data)
        elif args.type == 'soft_mask':
            # numpy array, save each channel with cv2
            num_channels = args.data.shape[0]
            # first channel is background -- ignore
            for i in range(1, num_channels):
                data = args.data[i]
                data = (data * 255).astype(np.uint8)
                imwrite_checked(path.join(self.soft_mask_dir, f'{i}', args.name + '.png'), data)
        else:
            raise NotImplementedError(args.type)

    def check_saves(self) -> None:
        """Raises SaveError naming the saves that failed since the last check (each is reported
        once). Call it after save_queue.join() to know that everything queued was written."""
        with self.save_errors_lock:
            errors, self.save_errors = self.save_errors, []
        if errors:
            raise SaveError('; '.join(errors))

    def _extract_frames(self, video: str):
        cap = cv2.VideoCapture(video)
        frame_index = 0
        print(f'Extracting frames from {video} into {self.image_dir}...')
        with tqdm() as bar:
            while (cap.isOpened()):
                _, frame = cap.read()
                if frame is None:
                    break
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, f'{frame_index:07d}.jpg'), frame)
                frame_index += 1
                bar.update()
        print('Done!')

    def _copy_resize_frames(self, images: str):
        image_list = os.listdir(images)
        print(f'Copying/resizing frames into {self.image_dir}...')
        for image_name in tqdm(image_list):
            if self.max_size < 0:
                # just copy
                shutil.copy2(path.join(images, image_name), self.image_dir)
            else:
                frame = cv2.imread(path.join(images, image_name))
                h, w = frame.shape[:2]
                if self.max_size > 0 and min(h, w) > self.max_size:
                    new_w = (w * self.max_size // min(w, h))
                    new_h = (h * self.max_size // min(w, h))
                    frame = cv2.resize(frame, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path.join(self.image_dir, image_name), frame)
        print('Done!')

    def add_to_queue_with_warning(self, item: SaveItem):
        if self.save_queue.full():
            print(
                'The save queue is full! You need more threads or faster IO. Program might pause.')
        self.save_queue.put(item)

    def save_mask(self, ti: int, mask: np.ndarray):
        # mask should be uint8 H*W without channels
        assert 0 <= ti < self.length
        assert isinstance(mask, np.ndarray)

        mask = Image.fromarray(mask)
        mask.putpalette(self.palette)
        self.invalidate(ti)
        self.add_to_queue_with_warning(SaveItem('mask', mask, self.names[ti]))

    def save_visualization(self, ti: int, vis_mode: str, image: np.ndarray):
        # image should be uint8 3*H*W
        assert 0 <= ti < self.length
        assert isinstance(image, np.ndarray)

        self.add_to_queue_with_warning(SaveItem(f'visualization_{vis_mode}', image, self.names[ti]))

    def save_soft_mask(self, ti: int, prob: np.ndarray):
        # mask should be float (num_objects+1)*H*W np array
        assert 0 <= ti < self.length
        assert isinstance(prob, np.ndarray)

        self.add_to_queue_with_warning(SaveItem('soft_mask', prob, self.names[ti]))

    def _get_image_unbuffered(self, ti: int, modality: Optional[str] = None):
        # returns H*W*3 uint8 array; modality selects a task's other frame directory
        assert 0 <= ti < self.length

        name = self.names[ti]
        modality = modality or self.default_modality
        if name not in self.frame_files[modality]:
            modality = self.default_modality
        image_path = path.join(self.image_dirs[modality], self.frame_files[modality][name])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        return image

    def _get_mask_unbuffered(self, ti: int):
        # returns H*W uint8 array; a task falls back to its pregenerated mask for untouched frames
        assert 0 <= ti < self.length

        name = self.names[ti]
        mask_path = path.join(self.mask_dir, name + '.png')
        if path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = np.array(mask)
            return mask
        if self.task is not None and name in self.task.prefill:
            prefill = np.array(Image.open(self.task.prefill[name]['path']))
            return (prefill > 0).astype(np.uint8)
        return None

    def mask_source(self, ti: int) -> Optional[str]:
        # where the shown mask of a task frame comes from; None outside task mode
        if self.task is None:
            return None
        name = self.names[ti]
        anchor = self.task.anchor(name)
        if anchor is not None:
            return 'empty anchor' if anchor['kind'] == 'empty' else 'anchor'
        if path.exists(path.join(self.mask_dir, name + '.png')):
            return 'propagated'
        if name in self.task.prefill:
            return self.task.prefill[name]['source']
        return 'no mask'

    def frame_parts(self, ti: int, modality: Optional[str] = None
                    ) -> Tuple[str, Optional[str], Optional[str]]:
        """(file name, modality shown, where the mask comes from); the last two are None outside
        task mode."""
        modality = modality or self.default_modality
        name = self.names[ti]
        label = self.frame_files[modality].get(name, self.frame_files[self.default_modality][name])
        if self.task is None:
            return label, None, None
        return label, modality, self.mask_source(ti)

    def delete_mask(self, ti: int) -> None:
        # wait for pending saves so a queued write cannot recreate the file
        self.save_queue.join()
        mask_path = path.join(self.mask_dir, self.names[ti] + '.png')
        if path.exists(mask_path):
            os.remove(mask_path)
        self.invalidate(ti)

    def clear_sequence_masks(self) -> int:
        """Remove only generated mask PNGs from this sequence's workspace."""
        self.save_queue.join()
        directories = (self.mask_dir, path.join(self.workspace, 'binary_masks'),
                       self.soft_mask_dir)
        removed = 0
        for directory in directories:
            if not path.isdir(directory):
                continue
            for root, _, files in os.walk(directory):
                for name in files:
                    if name.lower().endswith('.png'):
                        os.remove(path.join(root, name))
                        removed += 1
        self.get_mask.clear()
        return removed

    def import_mask(self, file_name: str, size: Optional[Tuple[int, int]] = None):
        # read an mask file and resize it to exactly match the canvas size
        image = Image.open(file_name)
        if size is not None:
            # PIL uses (width, height)
            image = image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
        image = np.array(image)
        return image

    def import_layer(self, file_name: str, size: Tuple[int, int]):
        # read a RGBA/RGB file and resize it such that the entire layer is visible in the canvas
        # and then pad it to the canvas size (h, w)
        image = Image.open(file_name).convert('RGBA')
        im_w, im_h = image.size
        im_ratio = im_w / im_h
        canvas_ratio = size[1] / size[0]
        if im_ratio < canvas_ratio:
            # fit height
            new_h = size[0]
            new_w = int(new_h * im_ratio)
        else:
            # fit width
            new_w = size[1]
            new_h = int(new_w / im_ratio)
        image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        image = np.array(image)
        # padding
        pad_h = (size[0] - new_h) // 2
        pad_w = (size[1] - new_w) // 2
        image = np.pad(image,
                       ((pad_h, size[0] - new_h - pad_h), (pad_w, size[1] - new_w - pad_w), (0, 0)),
                       mode='constant',
                       constant_values=0)

        return image

    def invalidate(self, ti: int):
        # the image buffer is never invalidated
        self.get_mask.invalidate((ti, ))

    def __len__(self):
        return self.length

    @property
    def T(self) -> int:
        return self.length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width
