#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
import random

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq

from scipy.spatial.transform import Rotation

from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    to_camera_center,
    solve_PnP,
    SolvePnPParameters,
    Correspondences,
    rodrigues_and_translation_to_view_mat3x4
)


def find_and_add_points3d(point_cloud_builder: PointCloudBuilder,
                          view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                          intrinsic_mat: np.ndarray,
                          corners_1: FrameCorners, corners_2: FrameCorners,
                          params: TriangulationParameters) -> PointCloudBuilder:
    correspondence = build_correspondences(corners_1, corners_2)
    points3d, ids, median_cos = triangulate_correspondences(correspondence,
                                                            view_mat_1,
                                                            view_mat_2,
                                                            intrinsic_mat,
                                                            params)

    # print(f'point_cloud_builder.ids={point_cloud_builder.ids}\n point_cloud_builder._ids={point_cloud_builder._ids}')

    if median_cos < params.max_reprojection_error:
        point_cloud_builder.add_points(ids, points3d)
        # print(f"ADED POINTS TO CLOUD erorr={median_cos}")
    else:
        # print(f"DONT ADED POINTS TO CLOUD error={median_cos}")
        pass
    return point_cloud_builder


def frame_by_frame_calc(point_cloud_builder: PointCloudBuilder,
                        corner_storage: CornerStorage,
                        view_mats: np.array,
                        known_views: list,
                        intrinsic_mat: np.ndarray,
                        params: TriangulationParameters):
    # random.seed(42)
    n_frames = len(corner_storage)

    flag_camera = [False] * n_frames
    for frame in known_views:
        flag_camera[frame] = True

    for find_new_frame in range(2, n_frames):
        next_ids_3d, next_points_3d, _ = point_cloud_builder.build_point_cloud()

        is_find_frame = False
        next_frame = None
        # sort out confidence for next frame
        for per_confidence in range(99, 0, -1):
            if is_find_frame:
                break
            confidence = per_confidence / 100.0

            for next_frame in range(n_frames):
                if flag_camera[next_frame]:
                    continue
                now_corner = corner_storage[next_frame]
                now_ids, (now_id_1, now_id_2) = \
                    snp.intersect(now_corner.ids.ravel(),
                                  next_ids_3d.ravel(),
                                  indices=True)
                if len(now_ids) < 4:
                    continue
                # print(f"Confidence={confidence}")
                is_find_frame, rot_vec, t_vec, inliners = \
                    cv2.solvePnPRansac(objectPoints=next_points_3d[now_id_2],
                                       imagePoints=now_corner.points[now_id_1],
                                       cameraMatrix=intrinsic_mat,
                                       distCoeffs=None,
                                       confidence=confidence,
                                       iterationsCount=270,
                                       reprojectionError=1
                                       )

                if is_find_frame:
                    break

        # print(find_new_frame)
        if not is_find_frame:
            raise Exception('Confidence step 0.1 is not enough')

        view_mats[next_frame] = rodrigues_and_translation_to_view_mat3x4(rot_vec, t_vec)
        flag_camera[next_frame] = True

        for prepared_frame in range(n_frames):
            if not flag_camera[prepared_frame]:
                continue
            if prepared_frame == next_frame:
                continue

            point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                        view_mats[prepared_frame],
                                                        view_mats[next_frame],
                                                        intrinsic_mat,
                                                        corner_storage[prepared_frame],
                                                        corner_storage[next_frame],
                                                        params)

    return view_mats


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    # ('max_reprojection_error', 'min_triangulation_angle_deg', 'min_depth')
    params = TriangulationParameters(1, 0.9, 0.4)

    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()
    # corners_0 = corner_storage[0]
    # point_cloud_builder = PointCloudBuilder(corners_0.ids[:1],
    #                                         np.zeros((1, 3)))

    known_id1, known_id2 = known_view_1[0], known_view_2[0]

    point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                pose_to_view_mat3x4(known_view_1[1]),
                                                pose_to_view_mat3x4(known_view_2[1]),
                                                intrinsic_mat,
                                                corner_storage[known_id1],
                                                corner_storage[known_id2],
                                                params
                                                )

    # view_mats[best_id] = calc_camera_pose(point_cloud_builder, corner_storage[best_id], intrinsic_mat)

    view_mats = frame_by_frame_calc(point_cloud_builder,
                                    corner_storage,
                                    view_mats,
                                    [known_id1, known_id2],
                                    intrinsic_mat,
                                    params)

    #
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
