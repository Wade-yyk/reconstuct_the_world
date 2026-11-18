"""Foreground masks and measured stereo points for the figurine capture."""

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from statue_stereo import _stereo_disparities


# Approximate contours in percent of the source width and height. GrabCut
# adjusts the boundaries using the original image. Inner polygons exclude
# tabletop visible through the open parts of the dress and between the legs.
FIGURINE_CONTOURS = {
    "2f95ce": (
        [(42, 20), (48, 20), (50, 23), (48, 29), (46, 32), (49, 34), (52, 37), (60, 39), (62, 39), (63, 42), (59, 47), (55, 52), (48, 57), (43, 58), (49, 59), (55, 61), (57, 65), (55, 69), (50, 72), (43, 74), (35, 73), (29, 71), (26, 68), (25, 65), (27, 61), (33, 59), (36, 58), (33, 56), (28, 52), (25, 48), (22, 39), (24, 37), (28, 39), (33, 37), (34, 34), (32, 32), (34, 28), (35, 24), (38, 21)],
        [(28, 41), (34, 40), (36, 37), (37, 40), (34, 44)],
        [(45, 40), (48, 37), (52, 39), (58, 41), (52, 45), (47, 45)],
    ),
    "42fc70": (
        [(46, 11), (53, 11), (57, 14), (60, 17), (58, 23), (56, 25), (62, 28), (64, 31), (69, 33), (74, 32), (75, 37), (73, 46), (70, 51), (65, 56), (57, 59), (57, 62), (64, 63), (69, 66), (71, 70), (70, 74), (66, 78), (58, 80), (48, 80), (38, 78), (32, 75), (30, 70), (33, 67), (36, 65), (35, 61), (37, 56), (33, 51), (31, 44), (30, 35), (32, 33), (36, 35), (41, 33), (41, 30), (43, 27), (40, 25), (40, 22), (43, 17), (44, 14)],
        [(36, 38), (42, 35), (45, 34), (46, 38), (45, 41), (41, 42)],
        [(59, 35), (64, 35), (70, 36), (68, 40), (65, 42), (60, 39)],
        [(40, 58), (47, 60), (52, 60), (52, 62), (40, 64)],
    ),
    "70e5b9": (
        [(42, 22), (47, 22), (49, 23), (49, 29), (52, 32), (52, 34), (55, 37), (59, 35), (61, 36), (60, 40), (58, 45), (55, 49), (53, 52), (54, 57), (59, 59), (62, 62), (63, 65), (60, 68), (55, 70), (46, 70), (39, 69), (35, 66), (34, 62), (37, 59), (43, 57), (43, 54), (35, 53), (31, 47), (29, 41), (30, 37), (33, 36), (37, 37), (40, 35), (38, 32), (39, 28), (37, 25)],
        [(47, 54), (50, 52), (52, 57), (48, 57)],
    ),
    "7ed558": (
        [(58, 21), (61, 22), (63, 24), (61, 27), (65, 29), (71, 30), (72, 32), (69, 34), (65, 35), (63, 34), (62, 38), (64, 41), (66, 45), (66, 51), (64, 55), (63, 57), (67, 60), (70, 64), (70, 68), (67, 72), (61, 74), (54, 74), (47, 73), (42, 71), (40, 67), (40, 64), (44, 61), (48, 59), (47, 57), (42, 52), (40, 48), (41, 43), (46, 41), (47, 38), (48, 33), (50, 31), (49, 28), (51, 25), (54, 22)],
    ),
    "9a9b54": (
        [(61, 29), (64, 28), (67, 29), (70, 30), (69, 33), (73, 35), (78, 37), (78, 39), (75, 42), (71, 43), (67, 41), (66, 43), (67, 47), (69, 51), (69, 57), (68, 62), (71, 65), (73, 70), (72, 74), (68, 77), (61, 79), (54, 78), (49, 76), (45, 73), (43, 69), (44, 66), (49, 63), (57, 62), (56, 59), (52, 57), (48, 57), (48, 54), (51, 50), (53, 48), (55, 46), (56, 42), (57, 39), (58, 36), (57, 34), (58, 31)],
    ),
    "9f0ec4": (
        [(55, 41), (59, 41), (63, 42), (65, 45), (65, 47), (70, 49), (71, 51), (67, 54), (62, 53), (60, 52), (60, 55), (58, 57), (57, 58), (59, 61), (60, 65), (62, 71), (64, 76), (64, 78), (60, 78), (58, 77), (60, 79), (64, 82), (66, 85), (65, 89), (62, 92), (56, 95), (49, 95), (43, 94), (37, 90), (34, 85), (35, 81), (39, 78), (46, 76), (48, 73), (47, 68), (43, 71), (40, 71), (42, 68), (42, 66), (45, 62), (48, 58), (49, 55), (50, 52), (52, 50), (51, 48), (51, 45), (52, 43)],
        [(53, 71), (56, 75), (58, 78), (54, 77)],
    ),
    "9ff297": (
        [(45, 20), (48, 21), (50, 24), (50, 26), (55, 27), (58, 30), (56, 33), (54, 34), (58, 32), (59, 35), (60, 40), (58, 46), (56, 50), (60, 55), (62, 59), (64, 61), (64, 64), (60, 67), (55, 68), (48, 69), (41, 67), (37, 65), (36, 61), (39, 58), (44, 55), (43, 54), (39, 53), (34, 49), (31, 44), (29, 39), (30, 37), (32, 38), (35, 37), (39, 35), (39, 32), (41, 30), (41, 27), (39, 25), (41, 22)],
        [(50, 54), (53, 52), (56, 56), (52, 56)],
    ),
    "a41e08": (
        [(44, 21), (49, 22), (52, 25), (51, 30), (50, 33), (53, 34), (56, 36), (60, 37), (62, 37), (62, 41), (61, 46), (58, 51), (54, 55), (50, 57), (51, 59), (55, 59), (57, 63), (57, 67), (53, 70), (46, 72), (37, 73), (31, 71), (27, 68), (26, 64), (28, 61), (30, 60), (29, 58), (32, 54), (29, 51), (28, 46), (28, 41), (30, 39), (34, 37), (35, 34), (39, 32), (36, 31), (34, 29), (34, 27), (36, 26), (40, 24), (41, 22)],
        [(33, 43), (38, 40), (40, 38), (42, 41), (40, 45), (35, 46)],
        [(50, 40), (52, 37), (57, 40), (57, 43), (55, 45)],
        [(34, 56), (38, 58), (40, 58), (38, 60), (33, 60)],
    ),
    "b49a14": (
        [(32, 18), (36, 17), (39, 18), (44, 18), (47, 22), (47, 26), (46, 29), (47, 32), (46, 34), (50, 35), (54, 34), (56, 36), (55, 41), (53, 46), (52, 50), (49, 49), (47, 48), (46, 54), (45, 56), (51, 57), (55, 59), (56, 63), (54, 67), (48, 69), (42, 70), (35, 69), (30, 67), (27, 64), (26, 61), (29, 58), (36, 56), (37, 53), (38, 50), (32, 51), (27, 49), (28, 44), (29, 40), (31, 36), (30, 33), (32, 31), (30, 29), (29, 26), (31, 23), (32, 21), (30, 20)],
        [(41, 50), (43, 49), (42, 54), (40, 56), (39, 55)],
    ),
    "c19841": (
        [(43, 10), (48, 10), (53, 13), (55, 17), (54, 22), (52, 24), (55, 25), (58, 29), (57, 31), (61, 32), (64, 30), (66, 31), (66, 40), (66, 44), (64, 48), (60, 51), (54, 56), (51, 56), (50, 60), (56, 60), (61, 63), (65, 66), (66, 71), (63, 74), (57, 76), (46, 77), (37, 75), (31, 73), (28, 69), (29, 66), (32, 63), (32, 60), (36, 55), (38, 53), (35, 50), (32, 45), (30, 40), (31, 35), (32, 32), (35, 31), (38, 33), (40, 31), (39, 29), (40, 25), (34, 25), (32, 23), (34, 20), (39, 19), (39, 16), (41, 12)],
        [(36, 36), (41, 35), (46, 34), (46, 37), (44, 41), (40, 42)],
        [(57, 34), (62, 34), (61, 37), (59, 39)],
        [(39, 57), (43, 55), (47, 56), (46, 60), (39, 60), (35, 62)],
    ),
    "c9dda0": (
        [(41, 17), (46, 16), (51, 18), (53, 21), (54, 26), (52, 29), (53, 33), (55, 36), (57, 42), (60, 46), (62, 51), (59, 56), (55, 58), (49, 59), (49, 62), (55, 62), (60, 64), (62, 67), (63, 71), (61, 75), (55, 78), (47, 79), (37, 78), (30, 76), (27, 72), (27, 68), (29, 65), (35, 63), (35, 61), (42, 57), (43, 54), (41, 49), (37, 51), (34, 52), (33, 49), (37, 44), (39, 41), (38, 39), (39, 36), (38, 32), (34, 34), (32, 33), (32, 30), (29, 31), (27, 29), (28, 26), (30, 24), (36, 23), (39, 21), (39, 18)],
        [(39, 62), (44, 58), (46, 58), (45, 62)],
    ),
    "d6e479": (
        [(39, 17), (44, 16), (49, 16), (52, 18), (54, 20), (52, 24), (49, 26), (49, 29), (48, 31), (52, 32), (54, 35), (58, 38), (63, 38), (64, 41), (65, 46), (64, 49), (62, 50), (60, 48), (56, 51), (53, 55), (57, 57), (60, 60), (61, 64), (59, 68), (55, 71), (49, 72), (40, 72), (34, 71), (30, 68), (29, 64), (29, 61), (34, 58), (40, 56), (39, 55), (34, 53), (30, 49), (27, 45), (26, 38), (25, 33), (27, 32), (29, 34), (34, 35), (37, 33), (37, 31), (35, 29), (36, 26), (35, 23), (36, 20)],
        [(31, 38), (35, 37), (37, 36), (37, 39), (35, 44)],
        [(45, 37), (48, 37), (51, 39), (57, 41), (53, 45), (50, 45), (47, 42)],
    ),
    "e46ae4": (
        [(41, 14), (46, 14), (49, 13), (53, 15), (56, 17), (58, 22), (57, 26), (58, 30), (61, 35), (63, 39), (64, 44), (64, 47), (66, 52), (65, 55), (61, 53), (61, 57), (59, 59), (57, 57), (56, 54), (54, 52), (54, 56), (52, 63), (59, 64), (64, 67), (67, 71), (67, 76), (65, 80), (59, 83), (49, 84), (39, 83), (33, 81), (29, 78), (28, 74), (29, 70), (33, 67), (40, 64), (45, 60), (46, 54), (43, 55), (39, 56), (34, 55), (33, 52), (35, 50), (38, 48), (41, 42), (43, 37), (43, 33), (45, 29), (43, 25), (41, 28), (38, 31), (37, 34), (35, 32), (35, 29), (37, 25), (34, 29), (31, 29), (30, 26), (31, 23), (37, 20), (38, 18), (37, 17)],
    ),
}


def figurine_foreground(image, path):
    """Return a binary foreground mask at the same size as an RGB image."""
    contours = FIGURINE_CONTOURS.get(Path(path).stem[:6])
    if contours is None:
        raise ValueError(f"No foreground outline for {Path(path).name}")

    height, width = image.shape[:2]
    scale = np.array([width, height]) / 100
    polygons = [np.rint(np.asarray(contour) * scale).astype(np.int32) for contour in contours]
    outline = np.zeros((height, width), np.uint8)
    cv2.fillPoly(outline, polygons[:1], 1)
    for opening in polygons[1:]:
        cv2.fillPoly(outline, [opening], 0)

    band = max(3, round(height * 0.012))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    core = cv2.erode(outline, kernel)
    bounds = cv2.dilate(outline, kernel)
    labels = np.where(outline, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[bounds == 0] = cv2.GC_BGD
    labels[core == 1] = cv2.GC_FGD
    foreground_model = np.zeros((1, 65), np.float64)
    background_model = np.zeros_like(foreground_model)
    cv2.grabCut(image, labels, None, background_model, foreground_model, 4, cv2.GC_INIT_WITH_MASK)
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(25, round(mask.size * 0.00008))
    keep[0] = False
    return keep[components].astype(np.uint8)


def _silhouette_votes(points, intrinsics, poses, masks):
    height, width = masks.shape[1:]
    votes = np.zeros(len(points), dtype=np.int32)
    for K, pose, mask in zip(intrinsics, poses, masks):
        projected = points @ (K @ pose[:, :3]).T + K @ pose[:, 3]
        positive_depth = projected[:, 2] > 0
        x, y = np.rint(projected[:, :2] / np.maximum(projected[:, 2:], 1e-9)).astype(int).T
        in_frame = positive_depth & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        votes[in_frame] += mask[y[in_frame], x[in_frame]] > 0
    return votes


def _rectified_image_and_mask(image, mask, intrinsic, rotation, projection, size):
    maps = cv2.initUndistortRectifyMap(
        intrinsic, None, rotation, projection, size, cv2.CV_32FC1,
    )
    rectified = cv2.remap(image, *maps, cv2.INTER_LINEAR)
    foreground = cv2.remap(mask, *maps, cv2.INTER_NEAREST)
    return rectified, foreground


def _estimate_pair_points(images, masks, intrinsics, poses, tracks, object_center):
    """Match rectified photos, then return XYZ, RGB, and matching scores."""
    size = images[0].shape[1::-1]
    first_pose, second_pose = poses
    relative_rotation = second_pose[:, :3] @ first_pose[:, :3].T
    relative_translation = second_pose[:, 3] - relative_rotation @ first_pose[:, 3]
    r1, r2, _, p2, _, _, _ = cv2.stereoRectify(
        intrinsics[0], None, intrinsics[1], None, size,
        relative_rotation, relative_translation, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    axis = int(vertical)
    baseline = p2[axis, 3] / p2[axis, axis]
    center1 = r1 @ (first_pose[:, :3] @ object_center + first_pose[:, 3])
    center2 = r2 @ (second_pose[:, :3] @ object_center + second_pose[:, 3])
    if min(center1[2], center2[2]) <= 0:
        return None
    focal_length = intrinsics[:, 0, 0].mean()
    principal1 = np.asarray(size) / 2 - focal_length * center1[:2] / center1[2]
    principal2 = np.asarray(size) / 2 - focal_length * center2[:2] / center2[2]
    principal2[1 - axis] = principal1[1 - axis]

    # Shift the principal points along the epipolar direction to center the
    # object in both images without changing the common scanlines.
    p1 = np.array([[focal_length, 0, principal1[0], 0], [0, focal_length, principal1[1], 0], [0, 0, 1, 0]])
    p2 = p1.copy()
    p2[:2, 2] = principal2
    p2[axis, 3] = focal_length * baseline
    disparity_to_point = np.array([
        [1, 0, 0, -principal1[0]],
        [0, 1, 0, -principal1[1]],
        [0, 0, 0, focal_length],
        [0, 0, -1 / baseline, (principal1[axis] - principal2[axis]) / baseline],
    ])
    samples1 = cv2.undistortPoints(tracks[0, :, None], intrinsics[0], None, R=r1, P=p1)[:, 0]
    samples2 = cv2.undistortPoints(tracks[1, :, None], intrinsics[1], None, R=r2, P=p2)[:, 0]
    offsets = samples1 - samples2
    sample_disparities = offsets[np.abs(offsets[:, 1 - axis]) < 2, axis]
    if len(sample_disparities) < 6:
        return None
    low, high = np.percentile(sample_disparities, [5, 95])
    minimum = int(np.floor(low)) - 24
    disparity_count = int(np.ceil((high - minimum + 25) / 16)) * 16
    maximum = minimum + disparity_count
    search_width = size[axis] + min(minimum, 0) - max(maximum, 0)
    if disparity_count > size[axis] // 2 or search_width < 16:
        return None

    photo1, mask1 = _rectified_image_and_mask(images[0], masks[0], intrinsics[0], r1, p1, size)
    photo2, mask2 = _rectified_image_and_mask(images[1], masks[1], intrinsics[1], r2, p2, size)
    if vertical:
        photo1, photo2 = photo1.transpose(1, 0, 2), photo2.transpose(1, 0, 2)
        mask1, mask2 = mask1.T, mask2.T
    left = cv2.cvtColor(photo1, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(photo2, cv2.COLOR_RGB2GRAY)
    disparity, reverse = _stereo_disparities(left, right, minimum, disparity_count)
    y, x = np.nonzero(mask1)
    shift = disparity[y, x]
    projected_x = x - np.rint(shift).astype(int)
    safe_x = projected_x.clip(0, photo2.shape[1] - 1)
    inverse = reverse[y, safe_x]
    left_color = photo1[y, x].astype(np.float32)
    right_color = photo2[y, safe_x].astype(np.float32)
    round_trip = np.abs(shift + inverse)
    color_delta = np.mean(np.abs(left_color - right_color), axis=1)
    valid = (
        (projected_x == safe_x)
        & (mask2[y, safe_x] > 0)
        & (shift >= minimum)
        & (inverse >= 1 - minimum - disparity_count)
        & (round_trip < 1.0)
        & (color_delta < 22)
    )
    if not valid.any():
        return None

    rectified_x, rectified_y = (y[valid], x[valid]) if vertical else (x[valid], y[valid])
    samples = np.column_stack((rectified_x, rectified_y, shift[valid], np.ones(valid.sum())))
    homogeneous = samples @ disparity_to_point.T
    finite = np.abs(homogeneous[:, 3]) > 1e-9
    rectified = homogeneous[finite, :3] / homogeneous[finite, 3:]
    in_first_camera = rectified @ r1
    second_depth = in_first_camera @ relative_rotation[2] + relative_translation[2]
    in_front = (in_first_camera[:, 2] > 0) & (second_depth > 0)
    world = (in_first_camera[in_front] - first_pose[:, 3]) @ first_pose[:, :3]
    rgb = np.rint((left_color[valid] + right_color[valid]) / 2).astype(np.uint8)
    score = round_trip[valid] + color_delta[valid] / 22
    return world, rgb[finite][in_front], score[finite][in_front]


def densify_figurine(images, masks, intrinsics, poses, points, colors):
    """Return sparse and dense figurine points reconstructed from the photographs."""
    margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    loose_masks = np.asarray([cv2.dilate(mask, margin) for mask in masks])
    min_votes = max(2, len(masks) - 2)
    sparse_votes = _silhouette_votes(points, intrinsics, poses, loose_masks)
    inside = sparse_votes >= min_votes
    if inside.sum() < 8:
        raise ValueError("Too few sparse figurine points agree with the silhouettes")

    cloud = [points[inside]]
    rgb = [colors[inside]]
    scores = [np.full(inside.sum(), -1.0)]
    object_center = np.median(cloud[0], axis=0)
    # Use projected sparse foreground points to choose the disparity range.
    # New 3D samples are measured by stereo matching inside that range.
    cameras = intrinsics @ poses
    homogeneous = np.column_stack((cloud[0], np.ones(len(cloud[0]))))
    projected = (cameras @ homogeneous.T).transpose(0, 2, 1)
    predicted = projected[..., :2] / projected[..., 2:]
    for i, j in combinations(range(len(poses)), 2):
        rotation = poses[j, :, :3] @ poses[i, :, :3].T
        angle = np.rad2deg(np.linalg.norm(cv2.Rodrigues(rotation)[0]))
        if not 4 <= angle <= 52:
            continue
        pair = [i, j]
        result = _estimate_pair_points(
            images[pair], masks[pair], intrinsics[pair], poses[pair], predicted[pair], object_center,
        )
        if result is None:
            continue
        xyz, color, score = result
        keep = _silhouette_votes(xyz, intrinsics, poses, loose_masks) >= min_votes
        if keep.sum() >= 250:
            cloud.append(xyz[keep])
            rgb.append(color[keep])
            scores.append(score[keep])

    xyz = np.concatenate(cloud)
    colors = np.concatenate(rgb)
    quality_order = np.argsort(np.concatenate(scores), kind="stable")
    cloud_span = np.linalg.norm(np.ptp(cloud[0], axis=0))
    voxel_size = cloud_span / 220
    cells = np.floor(xyz[quality_order] / voxel_size).astype(np.int32)
    _, unique = np.unique(cells, axis=0, return_index=True)
    selected = quality_order[unique]
    return xyz[selected], colors[selected], len(cloud) - 1


def main():
    parser = ArgumentParser(description="Densify the figurine reconstruction from overlapping photos.")
    parser.add_argument("sparse", type=Path, help="Sparse reconstruction from shouban_reconstruction.py")
    parser.add_argument("output", type=Path, help="NPZ output containing measured XYZ and RGB samples")
    parser.add_argument("--masks", type=Path, help="Optional saved foreground masks matching the sparse image size")
    args = parser.parse_args()
    cv2.setNumThreads(3)

    with np.load(args.sparse) as archive:
        data = dict(archive)
    size = tuple(data["image_size"])
    paths = [Path(path) for path in data["image_paths"]]
    images = []
    for path in paths:
        with Image.open(path) as photo:
            images.append(np.asarray(photo.convert("RGB").resize(size, Image.Resampling.LANCZOS)))
    images = np.asarray(images)
    if args.masks is None:
        masks = np.asarray([figurine_foreground(image, path) for image, path in zip(images, paths)])
    else:
        with np.load(args.masks) as archive:
            if not np.array_equal(size, archive["image_size"]):
                raise ValueError("Foreground masks and camera intrinsics must have the same image size")
            masks_by_name = dict(zip(archive["image_names"], archive["masks"]))
        masks = np.asarray([masks_by_name[path.name] for path in paths])

    points, colors, stereo_pairs = densify_figurine(
        images, masks, data["K"], data["motion"], data["points"], data["colors"],
    )
    if "table_normal" in data:
        above_table = points @ data["table_normal"] >= data["table_offset"] - 0.04
        points, colors = points[above_table], colors[above_table]
    np.savez_compressed(args.output, points=points, colors=colors, stereo_pairs=stereo_pairs)
    print(f"Saved {len(points):,} photo-colored points from {stereo_pairs} accepted stereo pairs to {args.output}")


if __name__ == "__main__":
    main()
