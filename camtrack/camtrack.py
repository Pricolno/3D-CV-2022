#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from tqdm import tqdm

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
import random
from itertools import combinations

from pims import FramesSequence

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq

from scipy.spatial.transform import Rotation
import scipy

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
    rodrigues_and_translation_to_view_mat3x4,
    _calc_triangulation_angle_mask,
    eye3x4
)


def find_and_add_points3d(point_cloud_builder: PointCloudBuilder,
                          view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                          intrinsic_mat: np.ndarray,
                          corners_1: FrameCorners, corners_2: FrameCorners,
                          params: TriangulationParameters) -> PointCloudBuilder:
    correspondence = build_correspondences(corners_1, corners_2)
    points3d, ids, _, median_cos = triangulate_correspondences(correspondence,
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


def check_distance_between_pair_cameras(view_mat_1: np.array, view_mat_2: np.array, min_dist=0.2) -> bool:
    pose_center_1 = to_camera_center(view_mat_1)
    pose_center_2 = to_camera_center(view_mat_2)
    return np.linalg.norm(pose_center_1 - pose_center_2) > min_dist


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

    confidence = 0.999

    N_COUNT_RETRANGULATION_FRAMES = 4

    for find_new_frame in tqdm(range(2, n_frames), desc="Frame_by_frame_calc"):
        # next_ids_3d, next_points_3d, _ = point_cloud_builder.build_point_cloud()
        next_ids_3d, next_points_3d = point_cloud_builder.ids, point_cloud_builder.points

        # best_frames = []

        best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = None, None, None, None, None
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

            retval, rot_vec, t_vec, inliers = \
                cv2.solvePnPRansac(objectPoints=next_points_3d[now_id_2],
                                   imagePoints=now_corner.points[now_id_1],
                                   cameraMatrix=intrinsic_mat,
                                   distCoeffs=None,
                                   confidence=confidence,
                                   iterationsCount=3000,
                                   reprojectionError=5
                                   )

            if not retval:
                continue

            if best_inliers is None:
                best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = inliers, now_id_2, next_frame, rot_vec, t_vec
            elif len(best_inliers) < len(inliers):
                best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = inliers, now_id_2, next_frame, rot_vec, t_vec

            # best_frames.append((inliers, now_id_2, next_frame, rot_vec, t_vec))

        # best_inliers, best_ids, best_next_frame, rot_vec, t_vec = sorted(best_frames, key=lambda x: len(x[0]))[-1]

        outliers = np.setdiff1d(np.arange(0, len(next_points_3d[best_ids])), best_inliers.T, assume_unique=True)
        point_cloud_builder.remove_points(outliers)
        print(f"Found {len(best_inliers)} inliners for next_frame={best_next_frame}. Remove {len(outliers)} outliers")

        view_mats[best_next_frame] = rodrigues_and_translation_to_view_mat3x4(best_rot_vec, best_t_vec)
        flag_camera[best_next_frame] = True

        add_new_cloud_point_step = 1
        add_new_cloud_point_cur = -1
        for prepared_frame in range(n_frames):
            if not flag_camera[prepared_frame]:
                continue
            if prepared_frame == best_next_frame:
                continue
            if not check_distance_between_pair_cameras(view_mats[prepared_frame], view_mats[best_next_frame],
                                                       min_dist=0.05):
                #print(f"DEBUG | MIN DIST prepared_frame={prepared_frame} best_next_frame={best_next_frame}")
                continue
            #print(f"DEBUG | BIG DIST prepared_frame={prepared_frame} best_next_frame={best_next_frame}")

            add_new_cloud_point_cur += 1
            if not(add_new_cloud_point_cur % add_new_cloud_point_step == 0):
                continue

            point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                        view_mats[prepared_frame],
                                                        view_mats[best_next_frame],
                                                        intrinsic_mat,
                                                        corner_storage[prepared_frame],
                                                        corner_storage[best_next_frame],
                                                        params)


        # RETRANGULATION


    return view_mats


def find_start_position_camers(corner_storage: CornerStorage,
                               intrinsic_mat: np.ndarray):
    params_findEssentialMat = {
        'method': cv2.RANSAC,
        'prob': 0.999,
        # 'prob': 0.8,
        'threshold': 1,
        'maxIters': 3000
        # 'maxIters': 2000
    }
    params_findHomography = {
        "method": cv2.RANSAC,
        "ransacReprojThreshold": 0.9
    }

    THRESHHOLD_COUNT_FRAME = 100
    BIG_FRAME_STEP = 10
    SMALL_FRAME_STEP = 2

    count_frames = len(corner_storage)
    frame_step = BIG_FRAME_STEP
    if count_frames < THRESHHOLD_COUNT_FRAME:
        frame_step = SMALL_FRAME_STEP

    max_reproj_error = 5
    min_triangulation_angle_deg = 1
    min_depth = 0.1
    triangulation_parameters = TriangulationParameters(max_reproj_error, min_triangulation_angle_deg, min_depth)
    best_frame1, best_frame2, best_median_cos = None, None, None

    for frame1 in tqdm(range(0, count_frames, frame_step), desc="\nFind_start_position_camers\n"):
        for frame2 in range(frame1 + frame_step, count_frames, frame_step):
            correspondence = build_correspondences(
                corner_storage[frame1],
                corner_storage[frame2]
            )

            if len(correspondence.ids) < 35:
                continue

            essential_matrix, mask_essential = cv2.findEssentialMat(
                correspondence.points_1,
                correspondence.points_2,
                intrinsic_mat,
                **params_findEssentialMat
            )

            _, mask_homography = cv2.findHomography(
                correspondence.points_1,
                correspondence.points_2,
                **params_findHomography
            )

            cnt_mask_essential = np.count_nonzero(mask_essential)
            cnt_mask_homography = np.count_nonzero(mask_homography)
            if cnt_mask_essential < 2.3 * cnt_mask_homography:
                continue

            _, R, t, _ = cv2.recoverPose(
                essential_matrix,
                correspondence.points_1,
                correspondence.points_2,
                intrinsic_mat
            )

            # _calc_triangulation_angle_mask()

            # print()
            # print(f"R={R}, t={t}")

            # print(f"\nrodrigues_and_translation_to_view_mat3x4(R, t)={rodrigues_and_translation_to_view_mat3x4(R, t)}")
            _, ids, _, median_cos = triangulate_correspondences(
                correspondence,
                eye3x4(),
                rodrigues_and_translation_to_view_mat3x4(t_vec=t,
                                                         rot_mat=R),
                intrinsic_mat,
                triangulation_parameters
            )

            if len(ids) < 220:
                continue

            # 0 = median_cos a => a = 0 it's bad, 1/2 = median_cos a => a = 30 degree it's good
            # sort_key = (median_cos,)
            # print(f"sort_key(median_cos)={sort_key}")

            if best_frame1 is None:
                best_frame1, best_frame2, best_median_cos = frame1, frame2, median_cos
            elif best_median_cos > median_cos:
                best_frame1, best_frame2, best_median_cos = frame1, frame2, median_cos

            # ratio.append((frame1, frame2, sort_key))

    # frame1, frame2, _ = sorted(ratio, key=lambda x: (-x[2][0],))[-1]

    correspondence = build_correspondences(
        corner_storage[best_frame1],
        corner_storage[best_frame2],
    )

    essential_matrix, _ = cv2.findEssentialMat(
        correspondence.points_1,
        correspondence.points_2,
        intrinsic_mat,
        **params_findEssentialMat
    )

    _, R, t, _ = cv2.recoverPose(
        essential_matrix,
        correspondence.points_1,
        correspondence.points_2,
        intrinsic_mat
    )

    known_view_1 = (best_frame1, Pose(r_mat=np.eye(3), t_vec=np.zeros(3)))
    known_view_2 = (best_frame2, Pose(R.T, R.T @ -t))

    return known_view_1, known_view_2


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_1 is None:
        print("There are not start's views. Start finding two first position:")
        known_view_1, known_view_2 = find_start_position_camers(corner_storage, intrinsic_mat)
        print(f"known_view is None | known_view_1 = {known_view_1}, \n known_view_2={known_view_2}")
    else:
        print("There are start's views")

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
