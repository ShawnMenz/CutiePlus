"""
The demo videos' frames and files (gui/demo_video.py): the frame counter in a band above, the RGB
and thermal panels side by side with the mask in the category's color and nothing else drawn on
them, the caption band below with the category in its color; every visualization mode; one MP4 per clip, written whole or not
at all, and a sequence's older demos of the same mode removed.
"""
import os
from os import path

import av
import numpy as np
import pytest
from PIL import Image

from gui.demo_video import (BAND_COLOR, DemoCancelled, DemoClip, DemoFrame, compose_frame,
                            top_band_height, write_clip, write_demos)

H, W = 48, 64
GREEN = (0, 207, 0)
MODES = ('mask', 'davis', 'fade', 'light', 'popup', 'layer', 'rgba')


def inputs():
    rgb = np.full((H, W, 3), 120, np.uint8)
    thermal = np.full((H, W, 3), 60, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    mask[20:40, 20:40] = 1
    return rgb, thermal, mask


def test_counter_band_panels_and_caption(qapp):
    rgb, thermal, mask = inputs()
    frame = compose_frame([rgb, thermal], mask, 'davis', GREEN, 'car', 'the red car', 3, 9)
    top = top_band_height(H)
    assert frame.shape[1] == 2 * W and frame.shape[0] > top + H
    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0  # for yuv420p
    counter = frame[:top]
    assert (counter > 200).all(-1).any() and (counter == BAND_COLOR).all(-1).mean() > 0.5
    body = frame[top:top + H]
    # davis: half the frame, half the category's color, on both panels; nothing else drawn
    assert tuple(body[30, 30]) == (60, 163, 60) and tuple(body[30, W + 30]) == (30, 133, 30)
    assert (body[:20, :20] == 120).all() and tuple(body[H - 2, W - 2]) == (120, 120, 120)
    band = frame[top + H:]
    assert (band == BAND_COLOR).all(-1).mean() > 0.5
    category = (band[..., 1] > 150) & (band[..., 0] < 90) & (band[..., 2] < 90)
    assert category.any() and (band > 200).all(-1).any()  # the category green, the rest white


@pytest.mark.parametrize('mode', MODES)
def test_every_visualization_mode_gives_an_rgb_frame(qapp, mode):
    rgb, thermal, mask = inputs()
    frame = compose_frame([rgb, thermal], mask, mode, GREEN, 'car', 'x', 1, 1)
    assert frame.dtype == np.uint8 and frame.shape[2] == 3 and frame.shape[1] == 2 * W
    body = frame[top_band_height(H):]
    if mode == 'rgba':  # the target on black
        assert tuple(body[30, 30]) == (120, 120, 120) and tuple(body[H - 2, 2]) == (0, 0, 0)


def test_one_modality_a_missing_frame_and_a_smaller_thermal(qapp):
    rgb, thermal, mask = inputs()
    assert compose_frame([rgb], mask, 'davis', GREEN, 'car', 'x', 1, 1).shape[1] == W
    top = top_band_height(H)
    missing = compose_frame([rgb, None], mask, 'davis', GREEN, 'car', 'x', 1, 1)
    assert missing.shape[1] == 2 * W and not missing[top:top + H, W:].any()  # black, no mask
    small = np.full((H // 2, W // 2, 3), 60, np.uint8)
    scaled = compose_frame([rgb, small], mask, 'davis', GREEN, 'car', 'x', 1, 1)
    assert scaled.shape[1] == 2 * W and tuple(scaled[top + 30, W + 30]) == (30, 133, 30)


def test_a_long_description_wraps_into_a_taller_band(qapp):
    rgb, thermal, mask = inputs()
    one = compose_frame([rgb, thermal], mask, 'davis', GREEN, 'car', 'short', 1, 1)
    long = compose_frame([rgb, thermal], mask, 'davis', GREEN, 'car', 'word ' * 40, 1, 1)
    assert long.shape[1] == one.shape[1] and long.shape[0] > one.shape[0]


def clip_of(tmp_path, name: str = 's_1-3_davis.mp4', frames: int = 3) -> DemoClip:
    rgb, thermal, mask = inputs()
    files = []
    for array, file_name in ((rgb, 'rgb.png'), (thermal, 'thermal.png'), (mask, 'mask.png')):
        Image.fromarray(array).save(tmp_path / file_name)
        files.append(str(tmp_path / file_name))
    return DemoClip(str(tmp_path / 'demo' / name), ['visual', 'thermal'],
                    [DemoFrame(files[:2], files[2])] * frames, 'car', 'the red car', GREEN)


def test_a_clip_is_one_playable_mp4(tmp_path, qapp):
    clip = clip_of(tmp_path)
    steps = []
    write_clip(clip, 5, 1, lambda: steps.append(1), lambda: False)
    assert len(steps) == 3 and os.listdir(tmp_path / 'demo') == ['s_1-3_davis.mp4']
    with av.open(clip.output) as video:
        stream = video.streams.video[0]
        decoded = list(video.decode(stream))
        assert len(decoded) == 3 and stream.average_rate == 5
        assert decoded[0].width == 2 * W and decoded[0].height % 2 == 0


def test_a_cancelled_clip_leaves_no_file(tmp_path, qapp):
    clip = clip_of(tmp_path)
    steps = []
    with pytest.raises(DemoCancelled):
        write_clip(clip, 5, 1, lambda: steps.append(1), lambda: len(steps) == 1)
    assert os.listdir(tmp_path / 'demo') == []


def test_the_older_demos_of_the_sequence_and_mode_go(tmp_path, qapp):
    clip = clip_of(tmp_path)
    os.makedirs(tmp_path / 'demo')
    for stale in ('s_1-9_davis.mp4', 's_1-9_fade.mp4', 't_1-9_davis.mp4', 's2_1-9_davis.mp4'):
        (tmp_path / 'demo' / stale).write_bytes(b'')
    progress = []
    videos = write_demos([clip], 5, 1, lambda written, total: progress.append((written, total)),
                         lambda: False)
    assert videos == [clip.output] and progress[-1] == (3, 3)
    assert sorted(os.listdir(tmp_path / 'demo')) == ['s2_1-9_davis.mp4', 's_1-3_davis.mp4',
                                                     's_1-9_fade.mp4', 't_1-9_davis.mp4']
    assert path.getsize(clip.output) > 0
