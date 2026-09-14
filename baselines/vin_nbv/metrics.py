import numpy as np
from scipy.spatial import cKDTree


def coverage_distance(
    ground_truth_points,
    visible_points,
):
    """
    Compute one-sided distance from the full GT surface
    to the currently observed surface.

    D(P_gt, P_visible) =
        mean_{p in P_gt} min_{q in P_visible} ||p - q||

    Lower is better.
    """

    ground_truth_points = np.asarray(
        ground_truth_points,
        dtype=np.float32,
    )

    visible_points = np.asarray(
        visible_points,
        dtype=np.float32,
    )

    if ground_truth_points.ndim != 2:
        raise ValueError("ground_truth_points must have shape (N, 3).")

    if visible_points.ndim != 2:
        raise ValueError("visible_points must have shape (M, 3).")

    if ground_truth_points.shape[1] != 3:
        raise ValueError("ground_truth_points must have shape (N, 3).")

    if visible_points.shape[1] != 3:
        raise ValueError("visible_points must have shape (M, 3).")

    if len(visible_points) == 0:
        return float("inf")

    tree = cKDTree(visible_points)

    distances, _ = tree.query(
        ground_truth_points,
        k=1,
        workers=-1,
    )

    return float(np.mean(distances))


def surface_coverage(
    ground_truth_points,
    visible_points,
    threshold=0.01,
):
    """
    Fraction of GT surface points that lie within a given
    distance threshold from the observed geometry.

    Returns value in [0, 1].

    Example:
        0.72 = 72% of the surface is covered.
    """

    ground_truth_points = np.asarray(
        ground_truth_points,
        dtype=np.float32,
    )

    visible_points = np.asarray(
        visible_points,
        dtype=np.float32,
    )

    if len(visible_points) == 0:
        return 0.0

    tree = cKDTree(visible_points)

    distances, _ = tree.query(
        ground_truth_points,
        k=1,
        workers=-1,
    )

    covered = distances <= threshold

    return float(np.mean(covered))


def relative_reconstruction_improvement(
    distance_before,
    distance_after,
):
    """
    Adapted RRI.

    RRI =
        (D_before - D_after)
        / D_before

    Higher is better.
    """

    distance_before = float(distance_before)

    distance_after = float(distance_after)

    if not np.isfinite(distance_before):
        raise ValueError("distance_before must be finite.")

    if distance_before <= 0.0:
        return 0.0

    return float((distance_before - distance_after) / distance_before)


def merge_visible_points(
    point_clouds,
):
    """
    Concatenate visible point clouds from multiple views.
    """

    valid_clouds = [
        np.asarray(
            points,
            dtype=np.float32,
        )
        for points in point_clouds
        if len(points) > 0
    ]

    if not valid_clouds:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

    return np.concatenate(
        valid_clouds,
        axis=0,
    )
