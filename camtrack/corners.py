#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import copy

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def create_mask_by_corners(image, corners):
    mask = np.full(image.shape, 255).astype(np.uint8)
    points = corners.points.astype(int)
    for i in range(points.shape[0]):
        cv2.circle(mask, (points[i, 0], points[i, 1]), corners.sizes[i, 0], 0, -1)
    return mask


def add_corners(image, feature_params, corners=None, max_pyramid_lvl=4):
    if corners is not None:
        mask = create_mask_by_corners(image, corners)
    else:
        mask = None
        corners = FrameCorners.empty_frame()

    small_img = image

    new_p = cv2.goodFeaturesToTrack(small_img, mask=mask, **feature_params)
    if new_p is not None:
        new_p = new_p.reshape((-1, 2))
        n = new_p.shape[0]
        new_sizes = np.full(n, feature_params['blockSize']).reshape((-1, 1))

        corners.add_points(None, new_p, new_sizes)

    return corners


def track_corners(image_0, image_1, corners):
    small_img0 = image_0
    small_img1 = image_1
    new_corners = FrameCorners.empty_frame()

    curr_ids = corners.ids.reshape((-1, 1))
    curr_points = corners.points.reshape((-1, 2))
    curr_sizes = corners.sizes.reshape((-1, 1))

    new_p, st, err = cv2.calcOpticalFlowPyrLK(small_img0,
                                              small_img1,
                                              curr_points,
                                              None)

    if new_p is not None:
        new_p = new_p[np.hstack((st, st)) == 1].reshape((-1, 2))
        new_corners.add_points(curr_ids[st == 1],
                               new_p,
                               curr_sizes[st == 1])

    return new_corners


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.06,
                          minDistance=16,
                          blockSize=10)


    image_0 = (frame_sequence[0] * 255.0).astype(np.uint8)
    corners = add_corners(image_0, feature_params)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = (image_1 * 255.0).astype(np.uint8)
        corners = track_corners(image_0, image_1, corners)

        corners = add_corners(image_1, feature_params, corners)

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
