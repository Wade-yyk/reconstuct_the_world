"""Recover a surface constrained by the figurine's silhouettes in photos."""

from argparse import ArgumentParser
from base64 import b64encode
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from p4 import linear_estimate_3d_point


def upright_basis(motion, center, front_index, up):
    rotations = motion[:, :, :3]
    camera_centers = -np.einsum("nji,nj->ni", rotations, motion[:, :, 3])
    down = -up / np.linalg.norm(up)
    forward = center - camera_centers[front_index]
    forward -= (forward @ down) * down
    forward /= np.linalg.norm(forward)
    return np.column_stack((np.cross(down, forward), down, forward))


def _project(points, camera, shape):
    height, width = shape
    projection = points @ camera[:, :3].T + camera[:, 3]
    depth = projection[:, 2]
    pixels = np.rint(projection[:, :2] / np.maximum(depth[:, None], 1e-9))
    columns, rows = pixels.T
    visible = (depth > 0) & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    indices = np.clip(pixels, 0, [width - 1, height - 1]).astype(np.intp)
    return indices[:, 1], indices[:, 0], visible, depth


def _silhouette_scores(axes, center, basis, cameras, distances, allowed_outliers):
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3) @ basis.T + center
    lowest = np.full((allowed_outliers + 1, len(points)), np.inf, dtype=np.float32)
    for camera, distance in zip(cameras, distances):
        rows, columns, visible, _ = _project(points, camera, distance.shape)
        score = np.where(visible, distance[rows, columns], -1e3)
        for layer in lowest:
            next_score = np.maximum(layer, score)
            np.minimum(layer, score, out=layer)
            score = next_score
    return lowest[-1].reshape(grid.shape[:3])


def _surface_colors(points, normals, cameras, motion, images, masks, spacing):
    colors = np.zeros((len(points), 3), dtype=np.float64)
    total = np.zeros(len(points), dtype=np.float64)
    for camera, extrinsic, image, mask in zip(cameras, motion, images, masks):
        rows, columns, visible, depth = _project(points, camera, mask.shape)
        valid = visible & mask[rows, columns]
        z_buffer = np.full(mask.shape, np.inf, dtype=np.float32)
        np.minimum.at(z_buffer, (rows[valid], columns[valid]), depth[valid])
        nearest = cv2.erode(z_buffer, np.ones((3, 3), dtype=np.uint8))
        unobscured = valid & (depth < nearest[rows, columns] + 2.5 * spacing)
        camera_center = -extrinsic[:, :3].T @ extrinsic[:, 3]
        direction = camera_center - points
        cosine = np.einsum("ij,ij->i", normals, direction) / np.linalg.norm(direction, axis=1)
        weights = np.where(unobscured, np.maximum(cosine, 0.04) ** 5, 0)
        colors += weights[:, None] * image[rows, columns]
        total += weights
    colored = total > 0
    return colored, np.rint(colors[colored] / total[colored, None]).astype(np.uint8)


def silhouette_surface(masks, K, motion, front_index, table_normal, table_offset, *, allowed_outliers=1):
    """Extract a surface from the calibrated silhouettes above the table."""
    masks = np.asarray(masks, dtype=bool)
    cameras = K @ motion
    bounds = [cv2.boundingRect(mask.astype(np.uint8)) for mask in masks]
    center_pixels = np.array([(x + width / 2, y + height / 2) for x, y, width, height in bounds])
    center = linear_estimate_3d_point(center_pixels, cameras)
    basis = upright_basis(motion, center, front_index, table_normal)
    depths = motion[:, 2, :3] @ center + motion[:, 2, 3]
    heights = np.array([bounds_for_view[3] for bounds_for_view in bounds])
    object_height = np.median(heights * depths / K[:, 1, 1])
    if not np.isfinite(object_height) or object_height <= 0:
        raise ValueError("Silhouette centers do not triangulate in front of the cameras")

    distances = [
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        - cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        for mask in masks
    ]
    coarse_axis = np.linspace(-0.8, 0.8, 49) * object_height
    coarse_axes = [coarse_axis] * 3
    coarse = _silhouette_scores(coarse_axes, center, basis, cameras, distances, allowed_outliers)
    # Exclude below-table volumes where the silhouettes cannot constrain the base.
    table_height = (center @ table_normal - table_offset) / np.linalg.norm(table_normal)
    occupied = np.argwhere((coarse >= -2) & (coarse_axis[None, :, None] <= table_height))
    if not len(occupied):
        raise ValueError("Silhouettes have no common volume")
    low = coarse_axis[np.maximum(occupied.min(axis=0) - 2, 0)]
    high = coarse_axis[np.minimum(occupied.max(axis=0) + 2, len(coarse_axis) - 1)]
    spacing = object_height / 160
    axes = [np.arange(start, end + spacing, spacing) for start, end in zip(low, high)]
    scores = _silhouette_scores(axes, center, basis, cameras, distances, allowed_outliers)
    solid = gaussian_filter(np.clip(scores, -10, 10), 0.65) >= -1.2
    solid &= axes[1][None, :, None] <= table_height + spacing / 2
    components, count = label(solid)
    if not count:
        raise ValueError("Silhouette surface could not be extracted")
    largest = np.argmax(np.bincount(components[solid], minlength=count + 1))
    solid = components == largest
    vertices, faces, normals, _ = marching_cubes(np.pad(solid, 1), level=0.5, spacing=(spacing,) * 3)
    vertices += low - spacing
    # The bottom is hidden by the table. Omit the cap created by the clipping plane.
    exposed = vertices[:, 1] < table_height - spacing / 2
    faces = faces[exposed[faces].all(axis=1)]
    faces = (np.cumsum(exposed) - 1)[faces]
    points = vertices[exposed] @ basis.T + center
    return points, faces, normals[exposed] @ basis.T, spacing


def silhouette_cloud(images, masks, K, motion, front_index, table_normal, table_offset, *, allowed_outliers=1):
    """Carve a volume with the calibrated masks and sample its colored surface."""
    points, _, normals, spacing = silhouette_surface(
        masks, K, motion, front_index, table_normal, table_offset, allowed_outliers=allowed_outliers,
    )
    colored, colors = _surface_colors(points, normals, K @ motion, motion, images, np.asarray(masks, dtype=bool), spacing)
    return points[colored], colors


def export_viewer(poses_path, stereo_path, output_path):
    from shouban_stereo import figurine_foreground

    with np.load(poses_path) as poses:
        paths = list(map(Path, poses["image_paths"]))
        size = tuple(poses["image_size"])
        K, motion = poses["K"], poses["motion"]
        table_normal, table_offset = poses["table_normal"], poses["table_offset"].item()
    images = [
        cv2.cvtColor(cv2.resize(cv2.imread(str(path)), size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for path in paths
    ]
    masks = [figurine_foreground(image, path) for image, path in zip(images, paths)]
    front_index = next((index for index, path in enumerate(paths) if path.stem.startswith("42fc70")), 0)
    hull, hull_colors = silhouette_cloud(images, masks, K, motion, front_index, table_normal, table_offset)

    with np.load(stereo_path) as stereo:
        points, colors = stereo["points"], stereo["colors"]
    center = hull.mean(axis=0)
    basis = upright_basis(motion, center, front_index, table_normal)
    hull_altitudes = hull @ table_normal - table_offset
    altitudes = points @ table_normal - table_offset
    height = np.ptp(hull_altitudes)
    hull_distance = cKDTree(hull).query(points)[0]
    plausible = (hull_distance <= 0.065 * height) & (altitudes >= -0.008 * height)
    # Keep measured pedestal points where the silhouette hull overestimates its height.
    horizontal_axes = basis[:, [0, 2]]
    footprint = hull[hull_altitudes < 0.09 * height] @ horizontal_axes
    low, high = footprint.min(axis=0), footprint.max(axis=0)
    normalized_base = (points @ horizontal_axes - (high + low) / 2) / ((high - low) / 2)
    within_base = np.linalg.norm(normalized_base, axis=1) <= 1.06
    plausible |= within_base & (altitudes >= -0.015 * height) & (altitudes < 0.045 * height)
    points, colors = points[plausible], colors[plausible]
    if not len(points):
        raise ValueError("No stereo points are consistent with the masked object")
    # Prefer stereo geometry where its local support is dense.
    nearby = cKDTree(points).query(hull, k=3)[0][:, -1]
    hue, saturation, _ = cv2.cvtColor(hull_colors[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, :].T
    # The shallow gold pedestal marks a case where silhouettes overestimate depth.
    pedestal = (hull_altitudes < 0.17 * height) & (hue >= 19) & (hue <= 35) & (saturation >= 55)
    use_hull = (nearby > 0.015 * height) & ~pedestal
    combined = np.vstack((points, hull[use_hull]))
    combined_colors = np.vstack((colors, hull_colors[use_hull]))

    visible = (combined - center) @ basis
    encoded_points = b64encode(visible.astype("<f4").tobytes()).decode("ascii")
    encoded_colors = b64encode(combined_colors.astype(np.uint8).tobytes()).decode("ascii")
    output_path.write_text(
        f'window.shoubanPointCloud = "{encoded_points}";\n'
        f'window.shoubanPointCloudColors = "{encoded_colors}";\n'
        "window.shoubanPointCloudView = { yaw: 0, pitch: 0.18, pointRadius: 1.5 };\n",
        encoding="ascii",
    )
    print(f"Exported {len(visible):,} colored points from {len(paths)} photos to {output_path}")
    print(f"Included {len(points):,} triangulated points and {use_hull.sum():,} silhouette surface samples")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path, help="Camera reconstruction NPZ from shouban_reconstruction.py")
    parser.add_argument("stereo", type=Path, help="Stereo points NPZ from shouban_stereo.py")
    parser.add_argument("output", type=Path, help="Interactive point cloud JavaScript asset")
    args = parser.parse_args()
    export_viewer(args.poses, args.stereo, args.output)
