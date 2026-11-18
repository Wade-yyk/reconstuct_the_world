from argparse import ArgumentParser
from base64 import b64encode
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    u, _, vh = np.linalg.svd(E)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = u[:, 2]
    candidates = []
    for middle in (w, w.T):
        orthogonal = u @ middle @ vh
        rotation = np.linalg.det(orthogonal) * orthogonal
        candidates.append(np.column_stack((rotation, translation)))
        candidates.append(np.column_stack((rotation, -translation)))
    return np.array(candidates)

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # Each view contributes u P[2] - P[0] and v P[2] - P[1].
    rows = image_points[..., None] * camera_matrices[:, 2:3, :] - camera_matrices[:, :2, :]
    design = rows.reshape(image_points.shape[:-2] + (2 * camera_matrices.shape[0], 4))
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    point = vh[..., -1, :]
    return point[..., :3] / point[..., 3:4]


def _project_homogeneous(point_3d, camera_matrices):
    return np.einsum("...j,kij->...ki", point_3d, camera_matrices[:, :, :3]) + camera_matrices[:, :, 3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    projected = _project_homogeneous(point_3d, camera_matrices)
    residuals = projected[..., :2] / projected[..., 2:3] - image_points
    return residuals.reshape(residuals.shape[:-2] + (2 * camera_matrices.shape[0],))

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    projected = _project_homogeneous(point_3d, camera_matrices)
    depth = projected[..., 2:3]
    pixels = projected[..., :2] / depth
    numerator = camera_matrices[:, :2, :3] - pixels[..., None] * camera_matrices[:, 2:3, :3]
    derivatives = numerator / depth[..., None]
    return derivatives.reshape(derivatives.shape[:-3] + (2 * camera_matrices.shape[0], 3))

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point = linear_estimate_3d_point(image_points, camera_matrices)
    for _ in range(10):
        error = reprojection_error(point, image_points, camera_matrices)
        derivatives = jacobian(point, camera_matrices)
        # The pseudoinverse solves the Gauss-Newton least-squares step.
        step = np.linalg.pinv(derivatives) @ error[..., None]
        point = point - step[..., 0]
    return point

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    candidates = estimate_initial_RT(E)
    first_camera = K @ np.column_stack((np.eye(3), np.zeros(3)))
    counts = []
    for pose in candidates:
        projection_matrices = np.stack((first_camera, K @ pose))
        points = linear_estimate_3d_point(image_points, projection_matrices)
        depth1 = points[:, 2]
        depth2 = points @ pose[2, :3] + pose[2, 3]
        counts.append(np.count_nonzero((depth1 > 0) & (depth2 > 0)))
    return candidates[np.argmax(counts)]

if __name__ == '__main__':
    parser = ArgumentParser(description="Triangulate and reconstruct the statue from multiple views.")
    parser.add_argument("--checks-only", action="store_true", help="Run the supplied numerical examples.")
    parser.add_argument("--output", type=Path, help="Save the reconstructed statue as a figure.")
    parser.add_argument("--point-cloud-data", type=Path, help="Export the plotted XYZ points as data for the website.")
    parser.add_argument("--refine-cloud", action="store_true", help="Add consistency-checked stereo points and colors from the photos.")
    args = parser.parse_args()

    # Load the data
    data_dir = Path(__file__).resolve().parent / "data"
    image_data_dir = data_dir / "statue"
    unit_test_camera_matrix = np.load(data_dir / "unit_test_camera_matrix.npy")
    unit_test_image_matches = np.load(data_dir / "unit_test_image_matches.npy")
    image_paths = sorted((image_data_dir / "images").glob("*.jpg"))
    focal_length = 719.5459
    matches_subset = np.load(image_data_dir / "matches_subset.npy", allow_pickle=True, encoding="latin1")[0]
    fundamental_matrices = np.load(
        image_data_dir / "fundamental_matrices.npy", allow_pickle=True, encoding="latin1",
    )[0]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if args.checks_only:
        raise SystemExit(0)
    from sfm_utils import Frame, bundle_adjustment, merge_all_frames

    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
        print(f"Initialized and refined views {i + 1} and {i + 2}", flush=True)
    merged_frame = merge_all_frames(frames)
    print(f"Merged {len(merged_frame.motion)} camera poses", flush=True)

    # Construct the dense matching
    dense_matches = np.load(image_data_dir / "dense_matches.npy", allow_pickle=True, encoding="latin1")
    point_clouds = []
    point_colors = []
    # Keep the three dense view pairs used in the supplied visualization.
    for i in range(len(frames)-1):
        matches = dense_matches[i].T.reshape(-1, 2, 2)
        camera_matrices = merged_frame.K @ merged_frame.motion[i:i + 2]
        point_clouds.append(nonlinear_estimate_3d_point(matches, camera_matrices))
        if args.refine_cloud:
            image = imread(image_paths[i])
            height, width = image.shape[:2]
            # The supplied matches use a centered origin and reverse both image axes.
            pixels = np.rint([width / 2, height / 2] - matches[:, 0]).astype(int)
            pixels = np.clip(pixels, 0, [width - 1, height - 1])
            point_colors.append(image[pixels[:, 1], pixels[:, 0]])
        print(f"Triangulated {len(matches):,} dense matches for views {i + 1} and {i + 2}", flush=True)

    dense_structure = np.vstack(point_clouds)
    lower = np.array([-5, -5, 0])
    upper = np.array([5, 5, 10])
    in_view = np.all((dense_structure >= lower) & (dense_structure <= upper), axis=1)
    visible = dense_structure[in_view]
    print(f"Retained {len(visible):,} of {len(dense_structure):,} triangulated points in the supplied plot limits")

    colors = None
    if args.refine_cloud:
        from statue_stereo import refine_statue_cloud

        visible, colors = refine_statue_cloud(
            visible, np.vstack(point_colors)[in_view], image_paths,
            merged_frame.K, merged_frame.motion, dense_matches,
        )
        print(f"Refined point cloud contains {len(visible):,} points")

    if args.point_cloud_data is not None:
        coordinates = visible.astype("<f4").tobytes(order="C")
        encoded = b64encode(coordinates).decode("ascii")
        script_data = f'window.q4PointCloud = "{encoded}";\n'
        if colors is not None:
            encoded_colors = b64encode(colors.astype(np.uint8, copy=False).tobytes(order="C")).decode("ascii")
            script_data += f'window.q4PointCloudColors = "{encoded_colors}";\n'
        args.point_cloud_data.write_text(script_data, encoding="ascii")
        print(f"Exported point cloud data to {args.point_cloud_data}")
        if args.output is None:
            raise SystemExit(0)

    fig = plt.figure(figsize=(9, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    if colors is None:
        ax.scatter(*visible.T, c="#23272e", depthshade=False, s=0.65, alpha=0.65, linewidths=0)
    else:
        ax.scatter(*visible.T, c=colors / 255.0, depthshade=False, s=1.15, linewidths=0)
    ax.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(0, 10))
    ax.set_box_aspect((1, 1, 1), zoom=1.44)
    ax.view_init(-100, 90)
    ax.set_axis_off()
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi=160)
