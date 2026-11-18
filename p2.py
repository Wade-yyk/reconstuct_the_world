from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from epipolar_utils import compute_rectified_image, get_data_from_txt_file
from p1 import normalized_eight_point_alg

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogeneous coordinates of the epipole in the image,
              with a last coordinate of 1 when the epipole is finite
'''
def compute_epipole(F):
    _, _, vh = np.linalg.svd(F.T)
    epipole = vh[-1]
    if abs(epipole[2]) > np.finfo(float).eps:
        epipole = epipole / epipole[2]
    return epipole

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[:2]
    center = np.array([width / 2, height / 2])
    translate = np.eye(3)
    translate[:2, 2] = -center
    x, y, w = translate @ e
    if w < 0:
        x, y, w = -x, -y, -w

    radius = np.hypot(x, y)
    if radius == 0:
        raise ValueError("Cannot center the rectification at the epipole.")

    # Choose the rotation that moves the epipole to the horizontal axis.
    sign = -1 if x < 0 else 1
    cosine, sine = sign * np.array([x, y]) / radius
    rotate = np.array([
        [cosine, sine, 0],
        [-sine, cosine, 0],
        [0, 0, 1],
    ])

    project = np.eye(3)
    project[2, 0] = -w / (sign * radius)
    untranslate = np.eye(3)
    untranslate[:2, 2] = center
    return untranslate @ project @ rotate @ translate


def _transform_points(points, homography):
    transformed = points @ homography.T
    return transformed / transformed[:, 2:3]

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    x, y, w = e2
    cross_e2 = np.array([
        [0, -w, y],
        [w, 0, -x],
        [-y, x, 0],
    ])
    M = cross_e2 @ F + np.outer(e2, np.ones(3))
    H1_initial = H2 @ M

    # Fit an affine change of x while keeping the rectified y coordinates.
    warped1 = _transform_points(points1, H1_initial)
    warped2 = _transform_points(points2, H2)
    affine = np.eye(3)
    affine[0] = np.linalg.lstsq(warped1, warped2[:, 0], rcond=None)[0]
    return affine @ H1_initial, H2


def plot_rectified_pair(images, offsets, point_sets):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 5), layout="constrained")
    widths = [image.shape[1] for image in images]
    common_width = max(widths)
    top = min(offset[1] for offset in offsets) - 0.5
    bottom = max(offset[1] + image.shape[0] for image, offset in zip(images, offsets)) - 0.5

    mean_rows = (point_sets[0][:, 1] + point_sets[1][:, 1]) / 2
    row_order = np.argsort(mean_rows)
    guide_indices = row_order[np.array([1, 2, 3]) * len(row_order) // 4]

    for view, (axis, image, offset, points) in enumerate(zip(axes, images, offsets, point_sets), 1):
        height, width = image.shape[:2]
        left, image_top = np.array(offset) - 0.5
        axis.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=(left, left + width, image_top + height, image_top),
        )
        for match in guide_indices:
            axis.axhline(mean_rows[match], color="#fbbf24", linewidth=0.9, alpha=0.8)
        axis.scatter(
            points[guide_indices, 0], points[guide_indices, 1],
            s=20, color="#fbbf24", edgecolors="#171717", linewidths=0.7, zorder=3,
        )
        margin = (common_width - width) / 2
        axis.set(xlim=(left - margin, left + width + margin), ylim=(bottom, top))
        axis.set_title(f"Rectified view {view:02d}", loc="left")
        axis.set_axis_off()
    return fig

if __name__ == '__main__':
    parser = ArgumentParser(description="Rectify the Set 01 stereo image pair.")
    parser.add_argument("--output", type=Path, help="Save the rectified pair as an image.")
    args = parser.parse_args()

    im_set = Path(__file__).resolve().parent / "data" / "set1"
    im1 = imread(im_set / "image1.jpg")
    im2 = imread(im_set / "image2.jpg")
    points1 = get_data_from_txt_file(im_set / "pt_2D_1.txt")
    points2 = get_data_from_txt_file(im_set / "pt_2D_2.txt")
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F maps points from the first image to epipolar lines in the second.
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    new_points1 = _transform_points(points1, H1)
    new_points2 = _transform_points(points2, H2)
    before = np.abs(points1[:, 1] - points2[:, 1]).mean()
    after = np.abs(new_points1[:, 1] - new_points2[:, 1]).mean()
    print(f"Mean absolute vertical difference ({len(points1)} matches): {before:.4f} px -> {after:.4f} px")

    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    fig = plot_rectified_pair(
        (rectified_im1, rectified_im2), (offset1, offset2), (new_points1, new_points2),
    )
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi=160)
