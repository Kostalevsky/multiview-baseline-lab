import json
from pathlib import Path

import numpy as np


def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def backproject_depth(
    depth,
    intrinsics,
    camera_to_world,
):
    """
    Back-project a depth map into a world-space point cloud.

    Parameters
    ----------
    depth : np.ndarray
        Depth map of shape (H, W).
        Invalid pixels should contain NaN.

    intrinsics : dict
        Camera intrinsics with:
        - fx
        - fy
        - cx
        - cy

    camera_to_world : array-like
        4x4 camera-to-world transformation matrix.

    Returns
    -------
    np.ndarray
        Point cloud of shape (N, 3) in world coordinates.
    """

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    if depth.ndim != 2:
        raise ValueError("Depth must have shape (H, W).")

    fx = float(intrinsics["fx"])

    fy = float(intrinsics["fy"])

    cx = float(intrinsics["cx"])

    cy = float(intrinsics["cy"])

    camera_to_world = np.asarray(
        camera_to_world,
        dtype=np.float64,
    )

    if camera_to_world.shape != (
        4,
        4,
    ):
        raise ValueError("camera_to_world must be a 4x4 matrix.")

    valid_mask = np.isfinite(depth) & (depth > 0.0)

    v, u = np.nonzero(valid_mask)

    if len(u) == 0:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

    z = depth[
        v,
        u,
    ].astype(np.float64)

    x = (u.astype(np.float64) - cx) * z / fx

    y = -(v.astype(np.float64) - cy) * z / fy

    camera_points = np.stack(
        [
            x,
            y,
            -z,
        ],
        axis=1,
    )

    homogeneous = np.concatenate(
        [
            camera_points,
            np.ones(
                (
                    camera_points.shape[0],
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    world_homogeneous = (camera_to_world @ homogeneous.T).T

    world_points = world_homogeneous[
        :,
        :3,
    ]

    return world_points.astype(np.float32)


def load_visible_points(
    metadata_path,
    view_index,
):
    """
    Load one candidate view and reconstruct its visible
    point cloud from the saved depth map.

    Parameters
    ----------
    metadata_path : str or Path
        Path to candidates/metadata.json.

    view_index : int
        Candidate view index.

    Returns
    -------
    np.ndarray
        Visible point cloud of shape (N, 3).
    """

    metadata = load_metadata(metadata_path)

    intrinsics = metadata["camera"]["intrinsics"]

    views = metadata["views"]

    view_lookup = {int(item["index"]): item for item in views}

    if view_index not in view_lookup:
        raise KeyError(f"View index {view_index} " "not found in metadata.")

    view = view_lookup[view_index]

    depth_path = Path(view["depth"])

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: " f"{depth_path}")

    depth = np.load(depth_path)

    camera_to_world = view["camera_to_world"]

    return backproject_depth(
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=(camera_to_world),
    )


def load_all_visible_points(
    metadata_path,
):
    """
    Reconstruct visible point clouds for all candidate views.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping:
        view_index -> visible world-space points
    """

    metadata = load_metadata(metadata_path)

    visible_points = {}

    for view in metadata["views"]:
        view_index = int(view["index"])

        visible_points[view_index] = load_visible_points(
            metadata_path=metadata_path,
            view_index=view_index,
        )

    return visible_points
