"""Reconstruct the figurine photographs using the assignment's SfM methods."""

from argparse import ArgumentParser
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import coo_matrix

from p1 import normalized_eight_point_alg
from p4 import estimate_RT_from_E, linear_estimate_3d_point, nonlinear_estimate_3d_point


def load_images(directory, long_edge):
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    originals = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if len(originals) < 2 or any(image is None for image in originals):
        raise ValueError("Need at least two readable photos.")
    height, width = originals[0].shape[:2]
    factor = long_edge / max(height, width)
    size = round(width * factor), round(height * factor)
    if any(abs(image.shape[1] / image.shape[0] - width / height) > 0.01 for image in originals):
        raise ValueError("The photographs must have the same orientation and aspect ratio.")
    images = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in originals]
    return paths, images, size


def find_matches(images):
    sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=0.018)
    features = [sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None) for image in images]
    pixels = [np.array([keypoint.pt for keypoint in keys], dtype=np.float64) for keys, _ in features]
    descriptors = [descriptors for _, descriptors in features]
    matcher = cv2.BFMatcher()
    pairs = []
    for first, second in combinations(range(len(images)), 2):
        if descriptors[first] is None or descriptors[second] is None or len(descriptors[second]) < 2:
            continue
        forward = matcher.knnMatch(descriptors[first], descriptors[second], k=2)
        reverse = matcher.match(descriptors[second], descriptors[first])
        reverse_map = {match.queryIdx: match.trainIdx for match in reverse}
        matches = [(a.queryIdx, a.trainIdx) for a, b in forward
                   if a.distance < 0.80 * b.distance and reverse_map[a.trainIdx] == a.queryIdx]
        if len(matches) < 30:
            continue
        matches = np.array(matches)
        uv1, uv2 = pixels[first][matches[:, 0]], pixels[second][matches[:, 1]]
        _, inliers = cv2.findFundamentalMat(uv1, uv2, cv2.USAC_MAGSAC, 1.35, 0.999, 10000)
        if inliers is None:
            continue
        inliers = inliers.ravel().astype(bool)
        matches = matches[inliers]
        if len(matches) < 25:
            continue
        uv1, uv2 = uv1[inliers], uv2[inliers]
        _, planar = cv2.findHomography(uv1, uv2, cv2.USAC_MAGSAC, 2.2)
        nonplanar = len(matches) - np.count_nonzero(planar) if planar is not None else len(matches)
        fundamental = normalized_eight_point_alg(
            np.column_stack((uv1, np.ones(len(uv1)))), np.column_stack((uv2, np.ones(len(uv2)))),
        )
        pairs.append((first, second, matches, fundamental, nonplanar))
    print(f"Geometrically verified {len(pairs)} photo pairs", flush=True)
    return pixels, pairs


def intrinsics(focal, center):
    return np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1.0]])


def estimate_focal(pairs, size):
    reliable = [(fundamental, nonplanar) for _, _, _, fundamental, nonplanar in pairs if nonplanar >= 24]
    if len(reliable) < 3:
        return max(size) * 0.78
    fundamental, weights = zip(*reliable)
    fundamental = np.array(fundamental)
    weights = np.sqrt(weights)

    def objective(focal):
        K = intrinsics(focal, np.asarray(size) / 2)
        singular_values = np.linalg.svd(K.T @ fundamental @ K, compute_uv=False)
        imbalance = (singular_values[:, 0] - singular_values[:, 1]) / singular_values[:, 0]
        return np.average(np.log1p((imbalance / 0.025) ** 2), weights=weights)

    bounds = (max(size) * 0.48, max(size) * 1.5)
    return minimize_scalar(objective, bounds=bounds, method="bounded").x


def make_tracks(pairs):
    """Merge pairwise matches while keeping at most one observation per camera."""
    parent = {}
    views = {}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for first, second, matches, _, _ in sorted(pairs, key=lambda pair: -len(pair[2])):
        for feature1, feature2 in matches:
            key1, key2 = (first, feature1), (second, feature2)
            for key in (key1, key2):
                if key not in parent:
                    parent[key], views[key] = key, {key[0]}
            group1, group2 = root(key1), root(key2)
            if group1 != group2 and views[group1].isdisjoint(views[group2]):
                if len(views[group1]) < len(views[group2]):
                    group1, group2 = group2, group1
                parent[group2] = group1
                views[group1].update(views.pop(group2))
    groups = defaultdict(dict)
    for key in parent:
        groups[root(key)][key[0]] = key[1]
    return [group for group in groups.values() if len(group) >= 2]


def measure(points, cameras):
    projections = np.einsum("...i,kji->...kj", points, cameras[:, :, :3]) + cameras[:, :, 3]
    return projections[..., :2] / projections[..., 2:3], projections[..., 2]


def seed_pair(pairs, pixels, K):
    origin = np.column_stack((np.eye(3), np.zeros(3)))
    candidates = []
    for first, second, matches, fundamental, nonplanar in pairs:
        if nonplanar < 16:
            continue
        uv = np.stack((pixels[first][matches[:, 0]], pixels[second][matches[:, 1]]), axis=1)
        E = K.T @ fundamental @ K
        pose = estimate_RT_from_E(E, uv, K)
        cameras = K @ np.stack((origin, pose))
        xyz = linear_estimate_3d_point(uv, cameras)
        projected, depth = measure(xyz, cameras)
        errors = np.linalg.norm(projected - uv, axis=2).max(axis=1)
        valid = (depth > 0).all(axis=1) & (errors < 4)
        if valid.mean() < 0.65:
            continue
        baseline = -pose[:, :3].T @ pose[:, 3]
        parallax = np.linalg.norm(np.cross(xyz, xyz - baseline), axis=1) / (
            np.linalg.norm(xyz, axis=1) * np.linalg.norm(xyz - baseline, axis=1)
        )
        support = np.count_nonzero(valid & (parallax > np.sin(np.deg2rad(2.5))))
        if support < 12:
            continue
        score = support * np.sqrt(nonplanar / len(matches)) / (1 + np.median(errors[valid]))
        candidates.append((score, first, second, pose))
    if not candidates:
        raise ValueError("Could not find an overlapping camera pair with enough parallax.")
    _, first, second, pose = max(candidates, key=lambda item: item[0])
    print(f"Seed photos {first} and {second}", flush=True)
    return {first: origin, second: pose}


def triangulate(tracks, pixels, poses, K, tolerance, refine=False):
    groups = defaultdict(list)
    for identifier, track in enumerate(tracks):
        views = tuple(sorted(track.keys() & poses.keys()))
        if len(views) >= 2:
            groups[views].append(identifier)
    positions = {}
    for views, identifiers in groups.items():
        motion = np.stack([poses[view] for view in views])
        cameras = K @ motion
        uv = np.stack([pixels[view][[tracks[i][view] for i in identifiers]] for view in views], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = linear_estimate_3d_point(uv, cameras)
            projected, depth = measure(xyz, cameras)
            ray = xyz[:, None, :] + np.einsum("kji,kj->ki", motion[:, :, :3], motion[:, :, 3])
            ray /= np.linalg.norm(ray, axis=2, keepdims=True)
        min_cosine = (ray @ ray.transpose(0, 2, 1)).min(axis=(1, 2))
        errors = np.linalg.norm(projected - uv, axis=2)
        valid = ((depth > 0).all(axis=1) & (errors.max(axis=1) < tolerance)
                 & (min_cosine < np.cos(np.deg2rad(0.8))))
        if refine and valid.any():
            xyz[valid] = nonlinear_estimate_3d_point(uv[valid], cameras)
        positions.update(zip(np.asarray(identifiers)[valid], xyz[valid]))
    return positions


def register_next(pixels, tracks, poses, points, K):
    scores = []
    for view, uv in enumerate(pixels):
        if view in poses:
            continue
        usable = [i for i in points if view in tracks[i]]
        if len(usable) < 16:
            continue
        xyz = np.stack([points[i] for i in usable])
        observed = uv[[tracks[i][view] for i in usable]]
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            xyz, observed, K, None, iterationsCount=1600, reprojectionError=4.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or len(inliers) < 16:
            continue
        rotation, translation = cv2.solvePnPRefineLM(
            xyz[inliers[:, 0]], observed[inliers[:, 0]], K, None, rotation, translation,
        )
        support = len(inliers)
        scores.append((support * (support / len(usable)) ** 0.5, view,
                       np.column_stack((cv2.Rodrigues(rotation)[0], translation)), support))
    if not scores:
        return False
    _, view, pose, support = max(scores, key=lambda item: item[0])
    poses[view] = pose
    print(f"Registered photo {view} with {support} inlier correspondences", flush=True)
    return True


def observation_arrays(pixels, tracks, points, views):
    point_indices, view_indices, xy = [], [], []
    lookup = {view: index for index, view in enumerate(views)}
    for point, identifier in enumerate(points):
        for view, key in tracks[identifier].items():
            if view in lookup:
                view_indices.append(lookup[view])
                point_indices.append(point)
                xy.append(pixels[view][key])
    return np.array(view_indices), np.array(point_indices), np.array(xy)


def adjust_bundle(poses, points, tracks, pixels, K, iterations, optimize_focal=False):
    """Refine camera poses and 3D point tracks together against pixel residuals."""
    views = list(poses)
    motion = np.stack(list(poses.values()))
    parameters = np.stack([np.r_[cv2.Rodrigues(pose[:, :3])[0].ravel(), pose[:, 3]] for pose in motion])
    xyz = np.stack(list(points.values()))
    cameras, structure, observed = observation_arrays(pixels, tracks, points, views)
    center = K[:2, 2]
    adjustable = np.ones_like(parameters, dtype=bool)
    adjustable[0] = False
    adjustable[1, 3 + np.argmax(np.abs(parameters[1, 3:]))] = False
    camera_columns = np.full(parameters.shape, -1)
    camera_columns[adjustable] = np.arange(adjustable.sum())
    camera_count = adjustable.sum()
    point_end = camera_count + 3 * len(xyz)
    column_groups = [camera_columns[cameras], camera_count + 3 * structure[:, None] + np.arange(3)]
    if optimize_focal:
        column_groups.append(np.full((len(observed), 1), point_end))
    columns = np.concatenate(column_groups, axis=1)
    rows = np.broadcast_to(np.arange(2 * len(observed)).reshape(-1, 2, 1), (len(observed), 2, columns.shape[1]))
    columns = np.broadcast_to(columns[:, None, :], rows.shape)
    active = columns >= 0
    jacobian_rows, jacobian_columns = rows[active], columns[active]
    initial = np.r_[parameters[adjustable], xyz.ravel()]
    if optimize_focal:
        initial = np.r_[initial, np.log(K[0, 0])]

    def unpack(values):
        current = parameters.copy()
        current[adjustable] = values[:camera_count]
        world = values[camera_count:point_end].reshape(-1, 3)
        focal = np.exp(values[-1]) if optimize_focal else K[0, 0]
        return current, world, focal

    def project(values):
        current, world, focal = unpack(values)
        rodrigues = [cv2.Rodrigues(vector[:3]) for vector in current]
        rotation = np.stack([result[0] for result in rodrigues])
        camera_points = np.einsum("nij,nj->ni", rotation[cameras], world[structure]) + current[cameras, 3:]
        xy = focal * camera_points[:, :2] / camera_points[:, 2:]
        return xy, camera_points, focal, rotation, rodrigues, world

    def residual(values):
        return (project(values)[0] + center - observed).ravel()

    def derivative(values):
        xy, camera_points, focal, rotation, rodrigues, world = project(values)
        depth = camera_points[:, 2]
        pixel_gradient = np.zeros((len(observed), 2, 3))
        pixel_gradient[:, 0, 0] = focal / depth
        pixel_gradient[:, 1, 1] = focal / depth
        pixel_gradient[:, :, 2] = -xy / depth[:, None]
        rotations_gradient = np.stack([result[1].reshape(3, 3, 3) for result in rodrigues])
        position_gradient = np.einsum("nkij,nj->nik", rotations_gradient[cameras], world[structure])
        blocks = [pixel_gradient @ position_gradient, pixel_gradient, pixel_gradient @ rotation[cameras]]
        if optimize_focal:
            blocks.append(xy[..., None])
        entries = np.concatenate(blocks, axis=2)
        return coo_matrix((entries[active], (jacobian_rows, jacobian_columns)),
                          shape=(2 * len(observed), len(values))).tocsr()

    fit = least_squares(residual, initial, jac=derivative, loss="soft_l1", f_scale=1.5,
                        x_scale="jac", tr_options={"maxiter": 100}, max_nfev=iterations)
    current, world, focal = unpack(fit.x)
    for view, row in zip(views, current):
        poses[view] = np.column_stack((cv2.Rodrigues(row[:3])[0], row[3:]))
    K[0, 0] = K[1, 1] = focal
    points.update(zip(points, world))
    errors = np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)
    print(f"Bundle adjustment: {len(views)} cameras, median residual {np.median(errors):.2f} px, f={focal:.1f}", flush=True)


def fit_tabletop(points, observed_points, pixels, height, motion):
    """Fit the dominant plane beneath the figure from reconstructed table texture."""
    candidates = points[np.unique(observed_points[pixels[:, 1] > 0.80 * height])]
    if len(candidates) < 50:
        return {}
    centers = np.einsum("kji,kj->ki", motion[:, :, :3], -motion[:, :, 3])
    scale = np.median(np.linalg.norm(centers - np.median(centers, axis=0), axis=1))
    tolerance = 0.014 * scale
    triangles = candidates[np.random.default_rng(123).integers(len(candidates), size=(1200, 3))]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    nondegenerate = lengths > 1e-8 * scale**2
    if not nondegenerate.any() or not scale > 0:
        return {}
    normals = normals[nondegenerate] / lengths[nondegenerate, None]
    offsets = np.einsum("ij,ij->i", normals, triangles[nondegenerate, 0])
    supports = np.abs(candidates @ normals.T - offsets) < tolerance
    selected = candidates[supports[:, supports.sum(axis=0).argmax()]]
    if len(selected) < max(40, 0.4 * len(candidates)):
        return {}
    center = selected.mean(axis=0)
    _, _, basis = np.linalg.svd(selected - center, full_matrices=False)
    normal = basis[-1]
    if (centers.mean(axis=0) - center) @ normal < 0:
        normal = -normal
    offset = center @ normal
    print(f"Tabletop estimated from {len(selected):,} sparse points", flush=True)
    return {"table_normal": normal, "table_offset": offset}


def save_reconstruction(output, paths, images, size, pixels, tracks, points, poses, K):
    views = sorted(poses)
    obs_view, obs_point, obs_xy = observation_arrays(pixels, tracks, points, views)
    rounded = np.rint(obs_xy).astype(int).clip(0, np.asarray(size) - 1)
    rgb = np.empty((len(obs_view), 3), dtype=np.uint8)
    for index, view in enumerate(views):
        selected = obs_view == index
        x, y = rounded[selected].T
        rgb[selected] = images[view][y, x, ::-1]
    order = np.argsort(obs_point, kind="stable")
    _, boundaries = np.unique(obs_point[order], return_index=True)
    rgb_tracks = np.split(rgb[order], boundaries[1:])
    colors = np.array([np.median(track, axis=0) for track in rgb_tracks], dtype=np.uint8)
    xyz = np.stack(list(points.values()))
    motion = np.stack([poses[view] for view in views])
    tabletop = fit_tabletop(xyz, obs_point, obs_xy, size[1], motion)
    np.savez_compressed(output, image_paths=[str(paths[view].resolve()) for view in views],
                        K=np.broadcast_to(K, (len(views), 3, 3)), image_size=np.asarray(size),
                        motion=motion, points=xyz,
                        colors=colors, observation_view=obs_view, observation_point=obs_point,
                        observation_xy=obs_xy, **tabletop)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path)
    parser.add_argument("output", type=Path, help="NPZ containing camera poses, sparse points, and their pixel observations")
    parser.add_argument("--long-edge", type=int, default=1000)
    args = parser.parse_args()
    cv2.setNumThreads(3)
    paths, images, size = load_images(args.photos, args.long_edge)
    pixels, pairs = find_matches(images)
    K = intrinsics(estimate_focal(pairs, size), np.asarray(size) / 2)
    print(f"Estimated initial focal length: {K[0, 0]:.1f} px", flush=True)
    tracks = make_tracks(pairs)
    poses = seed_pair(pairs, pixels, K)
    while len(poses) < len(images):
        points = triangulate(tracks, pixels, poses, K, tolerance=5)
        if not register_next(pixels, tracks, poses, points, K):
            adjust_bundle(poses, points, tracks, pixels, K, iterations=22, optimize_focal=True)
            points = triangulate(tracks, pixels, poses, K, tolerance=5)
            if not register_next(pixels, tracks, poses, points, K):
                break
        if len(poses) in (5, 9):
            points = triangulate(tracks, pixels, poses, K, tolerance=5)
            adjust_bundle(poses, points, tracks, pixels, K, iterations=18, optimize_focal=True)

    points = triangulate(tracks, pixels, poses, K, tolerance=4)
    adjust_bundle(poses, points, tracks, pixels, K, iterations=50, optimize_focal=True)
    points = triangulate(tracks, pixels, poses, K, tolerance=2.8, refine=True)
    save_reconstruction(args.output, paths, images, size, pixels, tracks, points, poses, K)
    print(f"Saved {len(points):,} sparse points and {len(poses)} of {len(images)} camera poses to {args.output}")


if __name__ == "__main__":
    main()
