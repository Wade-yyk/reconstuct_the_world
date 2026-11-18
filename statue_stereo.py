"""Optional photo-based refinement of the statue point cloud."""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.io import imread


def _rectify_image(image, intrinsics, rotation, projection, image_size):
    maps = cv2.initUndistortRectifyMap(
        intrinsics, None, rotation, projection, image_size, cv2.CV_32FC1,
    )
    return cv2.remap(image, *maps, cv2.INTER_LINEAR)


def _stereo_disparities(left, right, minimum, count):
    block_size = 7
    settings = dict(
        numDisparities=count,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=6,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    forward = cv2.StereoSGBM_create(minDisparity=minimum, **settings)
    reverse = cv2.StereoSGBM_create(minDisparity=1 - minimum - count, **settings)
    return (
        forward.compute(left, right).astype(np.float32) / 16,
        reverse.compute(right, left).astype(np.float32) / 16,
    )


def refine_statue_cloud(points, colors, image_paths, K, motion, dense_matches):
    """Supplement the statue reconstruction using consistent photo matches."""
    images = [imread(path) for path in image_paths]
    height, width = images[0].shape[:2]
    image_size = (width, height)
    rows, cols = np.indices((height, width))
    origin = np.array([width / 2, height / 2])
    intrinsics = K.copy()
    intrinsics[:2, 2] = origin
    # Input match coordinates increase opposite to both JPG pixel axes.
    pixel_basis = np.diag([-1.0, -1.0, 1.0])
    support = cKDTree(points)
    disparity_count = 64
    point_sets = [points]
    color_sets = [colors]
    for index, matches in enumerate(dense_matches):
        first_rotation, second_rotation = motion[index:index + 2, :, :3]
        first_translation, second_translation = motion[index:index + 2, :, 3]
        relative_rotation = second_rotation @ first_rotation.T
        relative_translation = second_translation - relative_rotation @ first_translation
        rotate_left, rotate_right, project_left, project_right, Q, _, _ = cv2.stereoRectify(
            intrinsics, None, intrinsics, None, image_size,
            pixel_basis @ relative_rotation @ pixel_basis, pixel_basis @ relative_translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=1,
        )
        left_rgb = _rectify_image(images[index], intrinsics, rotate_left, project_left, image_size)
        right_rgb = _rectify_image(images[index + 1], intrinsics, rotate_right, project_right, image_size)
        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

        # Center the disparity search using the provided matches near rectified scanlines.
        pixels = origin - matches.T.reshape(-1, 2, 2)
        left_samples = cv2.undistortPoints(
            pixels[:, 0, None], intrinsics, None, R=rotate_left, P=project_left,
        )[:, 0]
        right_samples = cv2.undistortPoints(
            pixels[:, 1, None], intrinsics, None, R=rotate_right, P=project_right,
        )[:, 0]
        difference = left_samples - right_samples
        near_scanline = (np.abs(difference[:, 1]) < 2) & (difference[:, 0] > 0)
        if not near_scanline.any():
            continue
        center_disparity = int(np.median(difference[near_scanline, 0]))
        min_disparity = max(1, center_disparity - disparity_count // 2)
        disparity, reverse = _stereo_disparities(left, right, min_disparity, disparity_count)

        matched_col = cols - np.rint(disparity).astype(int)
        safe_col = matched_col.clip(0, width - 1)
        right_disparity = reverse[rows, safe_col]
        pixel_left = left_rgb.astype(np.float32)
        pixel_right = right_rgb[rows, safe_col].astype(np.float32)
        color_change = np.mean(np.abs(pixel_left - pixel_right), axis=2)
        valid = (
            (matched_col == safe_col)
            & (disparity >= min_disparity)
            & (right_disparity >= 1 - min_disparity - disparity_count)
            & (np.abs(disparity + right_disparity) < 0.9)
            & (color_change < 30)
            & (left > 0)
            & (right[rows, safe_col] > 0)
        )

        rectified = cv2.reprojectImageTo3D(disparity, Q)[valid]
        first_camera = rectified @ rotate_left @ pixel_basis
        second_depth = first_camera @ relative_rotation[2] + relative_translation[2]
        in_front = (first_camera[:, 2] > 0) & (second_depth > 0) & np.isfinite(first_camera).all(axis=1)
        scene = (first_camera[in_front] - first_translation) @ first_rotation
        rgb = (pixel_left[valid][in_front] + pixel_right[valid][in_front]) / 2
        near_surface = support.query(scene)[0] < 0.35

        # Keep new points on the pale bust instead of adding surrounding furniture.
        red, green, blue = rgb.T
        pale_surface = (
            (red > 0.42 * 255) & (green > 0.36 * 255)
            & (green > 0.75 * red) & (blue > 0.53 * red) & (blue < 1.10 * red)
        )
        keep = near_surface & pale_surface
        point_sets.append(scene[keep])
        color_sets.append(np.rint(rgb[keep]).astype(np.uint8))

    refined = np.vstack(point_sets)
    refined_colors = np.vstack(color_sets)
    # Retain a measured point per small voxel, preferring supplied matches.
    cells = np.floor(refined / 0.028).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return refined[indices], refined_colors[indices]
