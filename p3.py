from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from epipolar_utils import get_data_from_txt_file


def factorization_method(points_im1, points_im2, *, return_singular_values=False):
    """Recover affine structure (3 × N) and camera motion (4 × 3).

    The input rows contain matched image coordinates in the same order.
    Optionally return the measurement matrix's singular values before truncation.
    """
    observations = np.vstack((points_im1[:, :2].T, points_im2[:, :2].T))
    centered = observations - observations.mean(axis=1, keepdims=True)
    u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    # Split the three retained singular values between motion and structure.
    root_scales = np.sqrt(singular_values[:3])
    motion = u[:, :3] * root_scales
    structure = root_scales[:, None] * vh[:3]
    if return_singular_values:
        return structure, motion, singular_values
    return structure, motion


def _align_for_comparison(estimate, reference):
    """Fit a translation, uniform scale, and orthogonal transform for display."""
    estimate_centered = estimate - estimate.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    u, singular_values, vh = np.linalg.svd(estimate_centered.T @ reference_centered)
    orientation = u @ vh
    scale = singular_values.sum() / np.sum(estimate_centered**2)
    return scale * (estimate_centered @ orientation) + reference.mean(axis=0)


def _plot_comparison(estimate, reference):
    fig = plt.figure(figsize=(9, 8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.14, hspace=0.07)
    layout = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    points = np.vstack((estimate, reference))
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2
    radius = 0.53 * np.max(upper - lower)
    limits = [(value - radius, value + radius) for value in center]
    marker = dict(s=29, color="#d49017", edgecolors="#583c12", linewidths=0.5)

    for col, (cloud, title) in enumerate((
        (estimate, "Aligned reconstruction"),
        (reference, "Ground truth"),
    )):
        oblique = fig.add_subplot(layout[0, col], projection="3d")
        oblique.scatter(*cloud.T, depthshade=False, **marker)
        oblique.set(
            xlim=limits[0], ylim=limits[1], zlim=limits[2],
            xlabel="X", ylabel="Y", zlabel="Z", title=f"{title} · Oblique view",
        )
        oblique.set_box_aspect((1, 1, 1))
        oblique.set_proj_type("ortho")
        oblique.view_init(elev=24, azim=-58)
        oblique.tick_params(labelsize=8)

        side = fig.add_subplot(layout[1, col])
        side.scatter(cloud[:, 1], cloud[:, 2], **marker)
        side.set(
            xlim=limits[1], ylim=limits[2],
            xlabel="Y", ylabel="Z", title=f"{title} · Side view",
        )
        side.set_aspect("equal")
        side.set_axisbelow(True)
        side.grid(linewidth=0.6, alpha=0.5)
        side.tick_params(labelsize=8)
    return fig


if __name__ == "__main__":
    parser = ArgumentParser(description="Recover 3D structure from two views with rank-3 SVD.")
    parser.add_argument("--output-dir", type=Path, help="Save both 3D comparisons as PNGs.")
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(__file__).resolve().parent / "data"
    for dataset, filename in (("set1", "q3-full.png"), ("set1_subset", "q3-subset.png")):
        directory = data_dir / dataset
        points_im1 = get_data_from_txt_file(directory / "pt_2D_1.txt")
        points_im2 = get_data_from_txt_file(directory / "pt_2D_2.txt")
        reference = get_data_from_txt_file(directory / "pt_3D.txt")
        structure, _, singular_values = factorization_method(
            points_im1, points_im2, return_singular_values=True,
        )

        aligned = _align_for_comparison(structure.T, reference)
        relative_error = np.linalg.norm(aligned - reference) / np.linalg.norm(
            reference - reference.mean(axis=0),
        )
        print(f"{dataset}: {len(reference)} matched points")
        print(f"Singular values: {np.array2string(singular_values, precision=8)}")
        print(f"Relative 3D RMS error after similarity alignment: {relative_error:.2%}")
        fig = _plot_comparison(aligned, reference)
        if args.output_dir is not None:
            fig.savefig(args.output_dir / filename, dpi=160)
            plt.close(fig)

    if args.output_dir is None:
        plt.show()
