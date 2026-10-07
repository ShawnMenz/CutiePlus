"""Frame and mask access, and the background mask saving of gui/resource_manager.py."""
import os
import stat
import threading
from os import path

import numpy as np
import pytest
from PIL import Image

from gui.resource_manager import ResourceManager, SaveError
from tests.conftest import compose_gui_config
from tests.helpers import ellipse_mask, make_task, read_mask


def manager(task) -> ResourceManager:
    return ResourceManager(compose_gui_config(task.workspace))


def join_within(res_man: ResourceManager, seconds: float) -> bool:
    """Whether every queued save finished in time (Queue.join itself cannot time out)."""
    waiter = threading.Thread(target=res_man.save_queue.join, daemon=True)
    waiter.start()
    waiter.join(seconds)
    return not waiter.is_alive()


def test_masks_fall_back_to_prefill(tmp_path):
    task = make_task(str(tmp_path), mask_anchors=(0, ), empty_anchors=(3, ), prefill=(2, ))
    res_man = manager(task)
    try:
        assert res_man.length == task.frame_count
        assert np.array_equal(res_man.get_mask(0), ellipse_mask(0, task.frame_count))
        assert np.array_equal(res_man.get_mask(2), ellipse_mask(2, task.frame_count))  # binary
        assert res_man.get_mask(1) is None
        assert res_man.mask_source(0) == 'anchor: ls'
        assert res_man.mask_source(2) == 'prefill: sam3'
        assert res_man.mask_source(3) == 'empty anchor: ls'
        assert res_man.mask_source(1) == 'no mask'
        # thermal frames are read from their own directory
        assert not np.array_equal(res_man.get_image(1, 'thermal'), res_man.get_image(1))
    finally:
        res_man.close()


def test_saved_mask_is_read_back(tmp_path):
    task = make_task(str(tmp_path))
    res_man = manager(task)
    try:
        mask = ellipse_mask(5, task.frame_count)
        assert res_man.get_mask(5) is None
        res_man.save_mask(5, mask)
        res_man.save_queue.join()
        assert np.array_equal(res_man.get_mask(5), mask)
        assert Image.open(path.join(task.masks_dir, '000006.png')).mode == 'P'
        assert res_man.mask_source(5) == 'propagated'
        res_man.delete_mask(5)
        assert res_man.get_mask(5) is None
    finally:
        res_man.close()


def test_failed_save_does_not_block_join(tmp_path):
    task = make_task(str(tmp_path))
    res_man = manager(task)
    os.chmod(task.masks_dir, stat.S_IRUSR | stat.S_IXUSR)  # read-only directory
    try:
        res_man.save_mask(1, ellipse_mask(1, task.frame_count))
        assert join_within(res_man, 10), 'join() waits forever after a failed save'
        with pytest.raises(SaveError) as error:
            res_man.check_saves()
        assert '000002.png' in str(error.value)
        res_man.check_saves()  # reported once
        # the save threads survive: a later save works once the directory is writable again
        os.chmod(task.masks_dir, stat.S_IRWXU)
        res_man.save_mask(1, ellipse_mask(1, task.frame_count))
        assert join_within(res_man, 10)
        res_man.check_saves()
        assert np.array_equal(read_mask(path.join(task.masks_dir, '000002.png')),
                              ellipse_mask(1, task.frame_count))
    finally:
        os.chmod(task.masks_dir, stat.S_IRWXU)
        res_man.close()


def test_failed_save_keeps_the_previous_file(tmp_path, monkeypatch):
    task = make_task(str(tmp_path), mask_anchors=(0, ))
    res_man = manager(task)
    try:
        original = read_mask(path.join(task.masks_dir, '000001.png'))

        def broken_save(image, fp, *args, **kwargs):
            with open(fp, 'wb') as f:
                f.write(b'\x89PNG half')
            raise OSError('disk full')

        monkeypatch.setattr(Image.Image, 'save', broken_save)
        res_man.save_mask(0, np.zeros_like(original))
        assert join_within(res_man, 10)
        with pytest.raises(SaveError):
            res_man.check_saves()
        monkeypatch.undo()
        assert np.array_equal(read_mask(path.join(task.masks_dir, '000001.png')), original)
        assert not [f for f in os.listdir(task.masks_dir) if not f.endswith('.png')]
    finally:
        res_man.close()


def test_close_finishes_pending_saves(tmp_path):
    task = make_task(str(tmp_path))
    res_man = manager(task)
    for ti in range(task.frame_count):
        res_man.save_mask(ti, ellipse_mask(ti, task.frame_count))
    res_man.close()
    res_man.close()  # twice is harmless
    assert len(os.listdir(task.masks_dir)) == task.frame_count
