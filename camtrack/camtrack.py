#! /usr/bin/env python3
from collections import OrderedDict

__all__ = [
    'track_and_calc_colors',
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
    eye3x4,
    compute_reprojection_errors,
    calc_inlier_indices
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


def get_reproj_threshold(sz):
    reproj_threshold = 2.0
    if sz > 700:
        reproj_threshold = 4.0
    if sz > 1100:
        reproj_threshold = 5.0
    if sz > 1600:
        reproj_threshold = 5.5
    if sz > 2100:
        reproj_threshold = 6.0

    return reproj_threshold


def get_step_by_frame(frame_count):
    step_by_frame = 1
    if frame_count > 30:
        step_by_frame = 2
    if frame_count > 100:
        step_by_frame = 3
    if frame_count > 200:
        step_by_frame = 4
    if frame_count > 400:
        step_by_frame = 5
    if frame_count > 800:
        step_by_frame = 7
    return step_by_frame


def mass_retriangulation(corner_list_storage: CornerStorage,
                         point_cloud_builder: PointCloudBuilder,
                         view_mats: list, intrinsic_mat: np.ndarray,
                         max_reproj_error: float,
                         COUNT_RECALCULATED_PAIR_FRAMES=10) -> PointCloudBuilder:
    max_id = max(corners.ids.max() for corners in corner_list_storage)
    params = TriangulationParameters(max_reproj_error, 0.1, 0.1)
    n_frames = len(corner_list_storage)

    # COUNT_RECALCULATED_PAIR_FRAMES = 10

    for _ in range(COUNT_RECALCULATED_PAIR_FRAMES):
        frame_1 = random.randint(0, n_frames - 1)
        frame_2 = random.randint(0, n_frames - 1)
        correspondence = build_correspondences(corner_list_storage[frame_1],
                                               corner_list_storage[frame_2])
        if correspondence.ids.shape[0] == 0:
            continue
        points3d, ids, _, _ = triangulate_correspondences(
            correspondence,
            view_mats[frame_1],
            view_mats[frame_2],
            intrinsic_mat,
            params
        )
        inliers_cnt = np.zeros((max_id + 1), int)
        for corners, view_mat in zip(corner_list_storage, view_mats):
            _, (idx_3d, idx_2d) = snp.intersect(
                point_cloud_builder.ids.flatten(), corners.ids.flatten(),
                indices=True)
            inliers = calc_inlier_indices(
                point_cloud_builder.points[idx_3d],
                corners.points[idx_2d],
                intrinsic_mat @ view_mat, max_reproj_error)
            inliers_ids = corners.ids[idx_2d[inliers]]
            inliers_cnt[inliers_ids] += 1

        point_cloud_builder.add_points(ids, points3d, inliers_cnt[ids] * -1)

    # print(f"errors = {point_cloud_builder.errors.flatten()}")
    return point_cloud_builder


def frame_by_frame_calc(point_cloud_builder: PointCloudBuilder,
                        corner_storage: CornerStorage,
                        view_mats: np.array,
                        known_views: list,
                        intrinsic_mat: np.ndarray,
                        params: TriangulationParameters):
    print(f"Start frame_by_frame_calc | known_views={known_views} | len(corner_storage)={len(corner_storage)}")

    np.random.seed(42)
    n_frames = len(corner_storage)

    flag_camera = np.full(shape=n_frames, fill_value=False)
    for frame in known_views:
        flag_camera[frame] = True

    def gen_iter_ready_frames(center_frame: int, flag_camera, is_find_ready_frames=True):
        nonlocal n_frames
        assert flag_camera[center_frame]

        frame_shift = {1, 2, 3, 4, 5, 10, 20, 30}
        len_already_frame = np.sum(flag_camera)
        # for choose not only ready frames but uncalculcaled frames (this we can stage in same function)
        if not is_find_ready_frames:
            len_already_frame = len(flag_camera) - len_already_frame

        large_shift = set(list(range(2, len_already_frame, get_step_by_frame(len_already_frame))))
        frame_shift.union(large_shift)

        cur_less_frame = 0
        cur_gr_frame = 0

        # find ready frame in left hand
        for cur_ind_frame in range(center_frame - 1, -1, -1):
            if not (flag_camera[cur_ind_frame] ^ (not is_find_ready_frames)):
                continue
            cur_less_frame += 1
            if cur_less_frame in frame_shift:
                yield cur_ind_frame
        # find ready frame in right hand
        for cur_ind_frame in range(center_frame + 1, n_frames, 1):
            if not (flag_camera[cur_ind_frame] ^ (not is_find_ready_frames)):
                continue
            cur_gr_frame += 1
            if cur_gr_frame in frame_shift:
                yield cur_ind_frame

    confidence = 0.999

    N_COUNT_RETRANGULATION_FRAMES = 4
    COUNT_READY_FRAMES_FOR_CHOOSE_NEXT_FRAME = 10

    for _ in tqdm(range(2, n_frames), desc="Frame_by_frame_calc"):
        next_ids_3d, next_points_3d = point_cloud_builder.ids, point_cloud_builder.points

        best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = None, None, None, None, None

        #
        def choose_iter_for_next_frame(flag_camera):
            nonlocal COUNT_READY_FRAMES_FOR_CHOOSE_NEXT_FRAME
            count_ready_frame = np.sum(flag_camera)
            shift_ready_frames = np.random.randint(low=0,
                                                   high=count_ready_frame,
                                                   size=COUNT_READY_FRAMES_FOR_CHOOSE_NEXT_FRAME)
            cur_shift = 0
            # print(f"choose_iter_for_next_frame | shift_ready_frames={shift_ready_frames}")
            for cur_frame in range(len(flag_camera)):
                if not flag_camera[cur_frame]:
                    continue
                if cur_shift in shift_ready_frames:
                    # print(f"choose_iter_for_next_frame | cur_shift={cur_shift} | cur_frame={cur_frame}")

                    for iter_frame in gen_iter_ready_frames(center_frame=cur_frame, flag_camera=flag_camera,
                                                            is_find_ready_frames=False):
                        # print(f"choose_iter_for_next_frame | iter_frame={iter_frame}")
                        yield iter_frame

                cur_shift += 1

        len_of_set_of_choosed_frames = 0
        for next_frame in choose_iter_for_next_frame(flag_camera=flag_camera):
            len_of_set_of_choosed_frames += 1
            # for next_frame in range(n_frames):

            if flag_camera[next_frame]:
                continue
            now_corner = corner_storage[next_frame]
            now_ids, (now_id_1, now_id_2) = \
                snp.intersect(now_corner.ids.ravel(),
                              next_ids_3d.ravel(),
                              indices=True)
            if len(now_ids) < 4:
                continue
            reprojectionError = get_reproj_threshold(len(now_id_1))

            retval, rot_vec, t_vec, inliers = \
                cv2.solvePnPRansac(objectPoints=next_points_3d[now_id_2],
                                   imagePoints=now_corner.points[now_id_1],
                                   cameraMatrix=intrinsic_mat,
                                   distCoeffs=None,
                                   confidence=confidence,
                                   iterationsCount=3000,
                                   reprojectionError=reprojectionError
                                   )

            if not retval:
                continue

            if best_inliers is None:
                best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = inliers, now_id_2, next_frame, rot_vec, t_vec
            elif len(best_inliers) < len(inliers):
                best_inliers, best_ids, best_next_frame, best_rot_vec, best_t_vec = inliers, now_id_2, next_frame, rot_vec, t_vec

        outliers = np.setdiff1d(np.arange(0, len(next_points_3d[best_ids])), best_inliers.T, assume_unique=True)
        point_cloud_builder.remove_points(outliers)
        print(
            f"Found {len(best_inliers)} inliners for next_frame={best_next_frame}. Remove {len(outliers)} outliers | Find between len_of_set_of_choosed_frames={len_of_set_of_choosed_frames}/{n_frames}")

        view_mats[best_next_frame] = rodrigues_and_translation_to_view_mat3x4(best_rot_vec, best_t_vec)
        flag_camera[best_next_frame] = True

        add_new_cloud_point_step = 1
        add_new_cloud_point_cur = -1

        for prepared_frame in gen_iter_ready_frames(best_next_frame, flag_camera):
            # if not flag_camera[prepared_frame]:
            #    continue
            # if prepared_frame == best_next_frame:
            #    continue

            if not check_distance_between_pair_cameras(view_mats[prepared_frame], view_mats[best_next_frame],
                                                       min_dist=0.0001):
                # print(f"DEBUG | MIN DIST prepared_frame={prepared_frame} best_next_frame={best_next_frame}")
                continue
            # print(f"DEBUG | BIG DIST prepared_frame={prepared_frame} best_next_frame={best_next_frame}")

            add_new_cloud_point_cur += 1
            if not (add_new_cloud_point_cur % add_new_cloud_point_step == 0):
                continue

            point_cloud_builder = find_and_add_points3d(point_cloud_builder,
                                                        view_mats[prepared_frame],
                                                        view_mats[best_next_frame],
                                                        intrinsic_mat,
                                                        corner_storage[prepared_frame],
                                                        corner_storage[best_next_frame],
                                                        params)

        # if _ % 5 == 0:
        #     # RETRANGULATION
        #     mass_retriangulation(corner_list_storage=corner_storage,
        #                          point_cloud_builder=point_cloud_builder,
        #                          view_mats=view_mats,
        #                          intrinsic_mat=intrinsic_mat,
        #                          max_reproj_error=params.max_reprojection_error,
        #                          COUNT_READY_FRAMES_FOR_CHOOSE_NEXT_FRAME=10)

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

    count_frames = len(corner_storage)

    frame_step = get_step_by_frame(count_frames)
    # frame_step = 1
    # if count_frames > 20:
    #     frame_step = 2
    # if count_frames > 80:
    #     frame_step = 3
    # if count_frames > 140:
    #     frame_step = 4
    # if count_frames > 250:
    #     frame_step = 5

    max_reproj_error = get_reproj_threshold(count_frames)

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
                print(f"FAILED1 | frame1={frame1} frame2={frame2}")
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
                print(f"FAILED2 | frame1={frame1} frame2={frame2}")
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
                print(f"FAILED3 | frame1={frame1} frame2={frame2}")
                continue

            # this is double bound very important, it's finding have taked ~5days ()
            LOW_BOUND_ON_ANGLE = 1.5
            HEIH_BOUND_ON_ANGLE = 12

            # if a few frames =>  delta by angle so big, and in other case angle need to get so small
            # if count_frames <= 30:
            #     HEIH_BOUND_ON_ANGLE = 13
            #     LOW_BOUND_ON_ANGLE = 2.5
            # elif count_frames <= 80:
            #     HEIH_BOUND_ON_ANGLE = 8
            #     LOW_BOUND_ON_ANGLE = 2
            # elif count_frames <= 150:
            #     HEIH_BOUND_ON_ANGLE = 6
            #     LOW_BOUND_ON_ANGLE = 1.5

            if median_cos > np.cos(LOW_BOUND_ON_ANGLE * (np.pi / 180)):
                print(f"FAILED4 | frame1={frame1} frame2={frame2}")
                continue

            if median_cos < np.cos(HEIH_BOUND_ON_ANGLE * (np.pi / 180)):
                print(f"FAILED5 | frame1={frame1} frame2={frame2}")
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
    print(f"find_start_position_camers | best_frame1={best_frame1} best_frame2={best_frame2} best_median_cos={best_median_cos}")
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
        print(f"known_view is None | known_view_1 = {known_view_1}, \n known_view_2={known_view_2}")

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

    print(f"track_and_calc_colors | frame_count={frame_count} | ")
    known_id1, known_id2 = known_view_1[0], known_view_2[0]
    print("Start track_and_calc_colors | first find_and_add_points3d")
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
