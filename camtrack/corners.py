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
    create_cli
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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    prev_image = frame_sequence[0]
    MAX_CORNERS = 700
    QUALITY_LEVEL = 0.08
    MIN_DISTANCE = 4


    new_poses = cv2.goodFeaturesToTrack(image=prev_image,
                                        maxCorners=MAX_CORNERS,
                                        qualityLevel=QUALITY_LEVEL,
                                        minDistance=MIN_DISTANCE,
                                        mask=None)
    max_ind = new_poses.shape[0] + 1
    LEN_OF_CORNER = 10
    frame_corners = FrameCorners(
        ids=np.arange(len(new_poses)),
        points=new_poses,
        sizes=np.full(len(new_poses), LEN_OF_CORNER)
    )

    builder.set_corners_at_frame(0, frame_corners)
    lk_params = dict()
    for i_frame, image in enumerate(frame_sequence[1:], 1):
        new_poses, statuses, error = cv2.calcOpticalFlowPyrLK(prevImg=np.uint8(prev_image * 255),
                                                              nextImg=np.uint8(image * 255),
                                                              prevPts=frame_corners.points,
                                                              nextPts=None,
                                                              **lk_params)

        statuses = statuses.ravel()
        ids = frame_corners.ids.ravel()

        new_poses = new_poses[statuses == 1]
        ids = ids[statuses == 1]

        count_extra_points = MAX_CORNERS - new_poses.shape[0]
        # 1 := empty
        mask = np.ones(shape=image.shape, dtype=np.uint8)
        for x, y in new_poses:
            cv2.circle(img=mask,
                       center=(np.uint8(x), np.uint8(y)),
                       radius=MIN_DISTANCE,
                       color=0,
                       thickness=-1)

        #print(mask)

        extra_points = cv2.goodFeaturesToTrack(image=image,
                                               maxCorners=count_extra_points,
                                               qualityLevel=QUALITY_LEVEL,
                                               minDistance=MIN_DISTANCE,
                                               mask=mask)

        # append extra_points to new_points
        extra_points = extra_points.reshape(extra_points.shape[0], 2)





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
