from .metrics import cal_3dbb_vertices
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2mat, mat2euler


def draw_bbox(bbox, pcd, name='bbox'):
    # coordinates of the bounding box vertices
    v1 = cal_3dbb_vertices(bbox.center, bbox.wlh, bbox.rotation_matrix)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # order: 0-1-2-3 (top square), 4-5-6-7 (bottom square)
    edges1 = [
        [v1[0], v1[1]], [v1[1], v1[5]], [v1[5], v1[4]], [v1[4], v1[0]],  # Top square
        [v1[3], v1[2]], [v1[2], v1[6]], [v1[6], v1[7]], [v1[7], v1[3]],  # Bottom square
        [v1[0], v1[3]], [v1[1], v1[2]], [v1[5], v1[6]], [v1[4], v1[7]]  # Vertical edges
    ]

    for edge1 in edges1:
        xs, ys, zs = zip(*edge1)
        ax.plot(xs, ys, zs, color='r', linewidth=1)

    points = np.asarray(pcd.points)

    def is_point_in_box(point, box_min, box_max):
        return np.all(point >= box_min) and np.all(point <= box_max)

    box_min = v1.min(axis=0)
    box_max = v1.max(axis=0)
    inside_points = np.array([pt for pt in points.T if is_point_in_box(pt, box_min, box_max)]).T
    x = inside_points[0]
    y = inside_points[1]
    z = inside_points[2]

    # x = points[0]
    # y = points[1]
    # z = points[2]

    ax.scatter(x, y, z, c='b', marker='.', s=0.8)

    ax.view_init(elev=90, azim=0)  # camera view angle

    # save image
    plt.savefig(f"/gsot3d/{name}.png")


def are_points_in_box(points, center, dimensions, rotation, order='rxyz'):
    """
    Determine if points are inside a 3D bounding box.

    Args:
        points (np.ndarray): An Nx3 array representing the coordinates of N points.
        center (array-like): A 3-element array representing the center of the box.
        dimensions (array-like): A 3-element array representing the dimensions (length, width, height) of the box.
        rotation (array-like): A 3-element array representing the rotation angles (yaw, pitch, roll) in radians.
        order (str): The order of the rotation angles. Default

    Returns:
        np.ndarray: A boolean array of length N, where True indicates the corresponding point is inside the box.
    """
    # Compute the rotation matrix
    # r = compute_rotation_matrix(rotation)
    r = euler2mat(rotation[0], rotation[1], rotation[2], order)

    # Translate points to the box's local coordinate system
    local_points = (points - center) @ r

    # Define the half-dimensions of the box
    half_dim = np.array(dimensions) / 2.0

    # Check if points are within the box's bounds
    inside_mask = np.all(np.abs(local_points) <= half_dim, axis=1)

    return inside_mask
