"""Create a photo-textured mesh from calibrated figurine views and stereo points."""

from argparse import ArgumentParser
from base64 import b64encode
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from shouban_silhouette import silhouette_surface, upright_basis
from shouban_stereo import figurine_foreground


def mesh_neighborhood(faces, vertex_count):
    """Return uniform vertex smoothing weights and pairs of adjacent triangles."""
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    directed = np.vstack((edges, edges[:, ::-1]))
    graph = coo_matrix((np.ones(len(directed)), directed.T), shape=(vertex_count, vertex_count)).tocsr()
    graph.data.fill(1)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    graph = graph.multiply((1 / np.maximum(degree, 1))[:, None]).tocsr()

    edges.sort(axis=1)
    edge_order = np.lexsort(edges.T)
    ordered = edges[edge_order]
    shared = np.all(ordered[1:] == ordered[:-1], axis=1)
    pair_indices = np.column_stack((edge_order[:-1][shared], edge_order[1:][shared]))
    return graph, pair_indices // 3


def refine_surface(vertices, normals, stereo_points, graph, spacing):
    """Shift the hull toward nearby stereo measurements that agree on depth."""
    if len(stereo_points) < 8:
        raise ValueError("At least eight stereo measurements are required")
    distances, indices = cKDTree(stereo_points).query(vertices, k=min(16, len(stereo_points)))
    difference = stereo_points[indices] - vertices[:, None, :]
    offsets = np.einsum("vki,vi->vk", difference, normals)
    median = np.median(offsets[:, :8], axis=1)
    deviation = np.median(abs(offsets[:, :8] - median[:, None]), axis=1)
    support = (distances[:, 7] < 3.6 * spacing) & (deviation < 0.65 * spacing)
    displacements = np.where(support, np.clip(median, -2.3 * spacing, spacing), 0.0)
    displacements = 0.45 * displacements + 0.55 * (graph @ displacements)
    points = vertices + displacements[:, None] * normals
    # Taubin smoothing removes voxel steps with little net shrinkage.
    for _ in range(4):
        points += 0.5 * (graph @ points - points)
        points -= 0.52 * (graph @ points - points)
    return points


def project_vertices(points, normals, camera, extrinsic, mask, spacing):
    height, width = mask.shape
    projection = points @ camera[:, :3].T + camera[:, 3]
    depth = projection[:, 2]
    uv = projection[:, :2] / np.maximum(depth[:, None], 1e-9)
    columns, rows = np.rint(uv).astype(np.intp).T
    inside = (depth > 0) & (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    rows = rows.clip(0, height - 1)
    columns = columns.clip(0, width - 1)
    signed_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    photo_distance = signed_distance[rows, columns]

    depth_map = np.full(mask.shape, np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (rows[inside], columns[inside]), depth[inside])
    nearest = cv2.erode(depth_map, np.ones((5, 5), dtype=np.uint8))[rows, columns]
    proximity = np.exp(-np.maximum(depth - nearest - 2 * spacing, 0) / (3 * spacing))
    source = -extrinsic[:, :3].T @ extrinsic[:, 3]
    rays = source - points
    facing = np.einsum("ni,ni->n", rays, normals) / np.linalg.norm(rays, axis=1)
    is_foreground = np.clip((photo_distance + 5) / 9, 0.015, 1)
    scores = np.where(inside, proximity * is_foreground * np.clip(facing + 0.09, 0.01, 1) ** 2, 0)
    return uv, scores


def pick_texture_views(faces, adjacency, projections, front_index):
    scores = np.array([np.mean(weights[faces], axis=1) for _, weights in projections]).T
    scores[:, front_index] *= 1.34
    labels = np.argmax(scores, axis=1)
    # Regularize camera choices over shared edges to keep source patches coherent.
    base = scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-10)
    first, second = adjacency.T
    for _ in range(6):
        regularized = base.copy()
        np.add.at(regularized, (first, labels[second]), 0.095)
        np.add.at(regularized, (second, labels[first]), 0.095)
        labels = np.argmax(regularized, axis=1)
    return labels


def texture_atlas(photos, masks):
    tiles, bounds = [], []
    for photo, mask in zip(photos, masks):
        x, y, width, height = cv2.boundingRect(mask)
        x, y = max(x - 5, 0), max(y - 5, 0)
        crop = photo[y:y + height + 10, x:x + width + 10].copy()
        crop_mask = mask[y:y + height + 10, x:x + width + 10]
        # Extend the foreground colors across the crop edge for bilinear texture sampling.
        _, nearest = cv2.distanceTransformWithLabels(
            1 - crop_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL,
        )
        palette = np.zeros((nearest.max() + 1, 3), dtype=np.uint8)
        palette[nearest[crop_mask > 0]] = crop[crop_mask > 0]
        tiles.append(palette[nearest])
        bounds.append((x, y))

    columns = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / columns))
    tile_width = max(tile.shape[1] for tile in tiles)
    tile_height = max(tile.shape[0] for tile in tiles)
    atlas = np.zeros((tile_height * rows, tile_width * columns, 3), dtype=np.uint8)
    transforms = []
    for index, (tile, (x, y)) in enumerate(zip(tiles, bounds)):
        left, top = (index % columns) * tile_width, (index // columns) * tile_height
        height, width = tile.shape[:2]
        atlas[top:top + height, left:left + width] = tile
        transforms.append((left - x, top - y, x, y, width, height))
    return atlas, transforms


def bake_uv(faces, position, labels, projections, atlas, transforms):
    # Give each source camera an independent texture coordinate at a shared vertex.
    count = len(position)
    key, inverse = np.unique(faces.ravel() + np.repeat(labels * count, 3), return_inverse=True)
    cameras, vertices = np.divmod(key, count)
    coordinates = np.empty((len(key), 2), dtype=np.float32)
    for camera, (left, top, x, y, width, height) in enumerate(transforms):
        mask = cameras == camera
        uv = projections[camera][0][vertices[mask]].copy()
        uv = np.clip(uv, [x, y], [x + width - 1, y + height - 1])
        coordinates[mask] = (uv + [left + 0.5, top + 0.5]) / [atlas.shape[1], atlas.shape[0]]
    return position[vertices], coordinates, inverse.reshape(-1, 3)


def export_mesh(poses_path, stereo_path, output):
    cv2.setNumThreads(3)
    with np.load(poses_path) as poses:
        paths = [Path(path) for path in poses["image_paths"]]
        width, height = poses["image_size"]
        K, motion = poses["K"], poses["motion"]
        up, table_offset = poses["table_normal"], poses["table_offset"].item()
    photos = []
    for path in paths:
        with Image.open(path) as image:
            photos.append(np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)))
    masks = [figurine_foreground(photo, path) for photo, path in zip(photos, paths)]
    front = next((index for index, path in enumerate(paths) if path.stem.startswith("42fc70")), 0)
    points, faces, normals, spacing = silhouette_surface(masks, K, motion, front, up, table_offset)
    active, faces = np.unique(faces.ravel(), return_inverse=True)
    vertices, normals, faces = points[active], normals[active], faces.reshape(-1, 3)
    graph, adjacency = mesh_neighborhood(faces, len(vertices))
    with np.load(stereo_path) as stereo:
        vertices = refine_surface(vertices, normals, stereo["points"], graph, spacing)
    projections = [
        project_vertices(vertices, normals, camera, extrinsic, mask, spacing)
        for camera, extrinsic, mask in zip(K @ motion, motion, masks)
    ]
    labels = pick_texture_views(faces, adjacency, projections, front)
    atlas, transforms = texture_atlas(photos, masks)
    center = vertices.mean(axis=0)
    basis = upright_basis(motion, center, front, up)
    visible = (vertices - center) @ basis
    positions, coords, indices = bake_uv(faces, visible, labels, projections, atlas, transforms)
    success, jpeg = cv2.imencode(".jpg", cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Texture atlas could not be encoded")
    data = {
        "positions": b64encode(positions.astype("<f4").tobytes()).decode("ascii"),
        "indices": b64encode(indices.astype("<u4").tobytes()).decode("ascii"),
        "uvs": b64encode(coords.astype("<f4").tobytes()).decode("ascii"),
        "texture": b64encode(jpeg.tobytes()).decode("ascii"),
        "view": {"yaw": 0, "pitch": 0.18},
    }
    output.write_text("window.shoubanMesh = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="ascii")
    print(f"Exported {len(indices):,} faces with a {atlas.shape[1]}x{atlas.shape[0]} photo texture")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("stereo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_mesh(args.poses, args.stereo, args.output)
