import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_NUM_VIEWS = 6

DEFAULT_SEED = 42

DEFAULT_COVERAGE_THRESHOLD = 0.01


def load_json(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_geometry(
    geometry_dir,
):
    geometry_dir = Path(geometry_dir)

    metadata_path = geometry_dir / "geometry_metadata.json"

    metadata = load_json(metadata_path)

    ground_truth_path = Path(metadata["ground_truth"])

    ground_truth = np.load(ground_truth_path).astype(np.float32)

    visible_points = {}

    for view in metadata["views"]:
        index = int(view["index"])

        path = Path(view["path"])

        visible_points[index] = np.load(path).astype(np.float32)

    return (
        ground_truth,
        visible_points,
        metadata,
    )


def load_candidate_metadata(
    metadata_path,
):
    metadata = load_json(metadata_path)

    return {int(view["index"]): view for view in metadata["views"]}


def compute_candidate_distance_profile(
    ground_truth,
    visible_points,
):
    """
    For every GT point, compute the distance
    to its nearest visible point from one view.

    Returns
    -------
    np.ndarray
        Shape (N_gt,).
    """

    if len(visible_points) == 0:
        return np.full(
            len(ground_truth),
            np.inf,
            dtype=np.float32,
        )

    tree = cKDTree(visible_points)

    distances, _ = tree.query(
        ground_truth,
        k=1,
        workers=-1,
    )

    return distances.astype(np.float32)


def precompute_distance_profiles(
    ground_truth,
    visible_points,
):
    """
    Precompute GT-to-view distance arrays
    for all candidate views.
    """

    profiles = {}

    total = len(visible_points)

    print("[VIN-NBV] " "Precomputing candidate " "distance profiles")

    for position, (
        index,
        points,
    ) in enumerate(
        sorted(visible_points.items()),
        start=1,
    ):
        print(
            "[VIN-NBV] "
            f"Profile "
            f"{position:03d}/"
            f"{total:03d}: "
            f"view {index:03d}"
        )

        profiles[index] = compute_candidate_distance_profile(
            ground_truth=(ground_truth),
            visible_points=(points),
        )

    return profiles


def coverage_distance(
    distances,
):
    """
    Mean one-sided GT -> observed surface distance.
    """

    return float(np.mean(distances))


def surface_coverage(
    distances,
    threshold=DEFAULT_COVERAGE_THRESHOLD,
):
    """
    Fraction of GT points whose nearest observed
    surface point is within threshold.
    """

    return float(np.mean(distances <= threshold))


def relative_reconstruction_improvement(
    distance_before,
    distance_after,
):
    if distance_before <= 0.0:
        return 0.0

    return float((distance_before - distance_after) / distance_before)


def camera_position(
    candidate,
):
    return np.asarray(
        candidate["camera_position"],
        dtype=np.float64,
    )


def find_nearest_camera(
    first_index,
    candidate_map,
):
    """
    Reproduce VIN-NBV-style initialization:
    the second camera is the spatially closest
    candidate to the first camera.
    """

    first_position = camera_position(candidate_map[first_index])

    nearest_index = None

    nearest_distance = float("inf")

    for index, candidate in candidate_map.items():
        if index == first_index:
            continue

        position = camera_position(candidate)

        distance = float(np.linalg.norm(position - first_position))

        if distance < nearest_distance:
            nearest_distance = distance

            nearest_index = index

    if nearest_index is None:
        raise RuntimeError("Could not find " "nearest camera.")

    return (
        nearest_index,
        nearest_distance,
    )


def combine_profiles(
    profiles,
    selected_indices,
):
    """
    Compute the GT -> currently observed geometry
    distance for the union of selected views.
    """

    if not selected_indices:
        raise ValueError("At least one selected " "view is required.")

    combined = profiles[selected_indices[0]].copy()

    for index in selected_indices[1:]:
        combined = np.minimum(
            combined,
            profiles[index],
        )

    return combined


def evaluate_candidate(
    current_distances,
    candidate_distances,
):
    distance_before = coverage_distance(current_distances)

    proposed_distances = np.minimum(
        current_distances,
        candidate_distances,
    )

    distance_after = coverage_distance(proposed_distances)

    rri = relative_reconstruction_improvement(
        distance_before=(distance_before),
        distance_after=(distance_after),
    )

    return (
        rri,
        distance_after,
        proposed_distances,
    )


def select_views(
    candidate_metadata_path,
    geometry_dir,
    num_views=DEFAULT_NUM_VIEWS,
    seed=DEFAULT_SEED,
    coverage_threshold=(DEFAULT_COVERAGE_THRESHOLD),
):
    """
    VIN-NBV Adapted / Oracle Geometry selector.

    Initialization:
        1. random first view
        2. nearest camera to first view

    Greedy phase:
        repeatedly select the remaining candidate
        with the largest true geometric RRI.
    """

    if num_views < 2:
        raise ValueError("VIN-NBV initialization " "requires at least 2 views.")

    (
        ground_truth,
        visible_points,
        _geometry_metadata,
    ) = load_geometry(geometry_dir)

    candidate_map = load_candidate_metadata(candidate_metadata_path)

    candidate_indices = sorted(candidate_map.keys())

    if num_views > len(candidate_indices):
        raise ValueError("num_views is larger than " "candidate pool.")

    missing_geometry = set(candidate_indices) - set(visible_points.keys())

    if missing_geometry:
        raise RuntimeError(
            "Missing geometry for " f"candidate views: " f"{sorted(missing_geometry)}"
        )

    profiles = precompute_distance_profiles(
        ground_truth=(ground_truth),
        visible_points=(visible_points),
    )

    rng = np.random.default_rng(seed)

    first_index = int(rng.choice(candidate_indices))

    (
        second_index,
        nearest_distance,
    ) = find_nearest_camera(
        first_index=(first_index),
        candidate_map=(candidate_map),
    )

    selected_indices = [
        first_index,
        second_index,
    ]

    current_distances = combine_profiles(
        profiles=profiles,
        selected_indices=(selected_indices),
    )

    initial_distance = coverage_distance(current_distances)

    initial_coverage = surface_coverage(
        current_distances,
        threshold=(coverage_threshold),
    )

    history = [
        {
            "step": 1,
            "selection_type": ("random_initial"),
            "selected_index": (first_index),
            "camera_position": (candidate_map[first_index]["camera_position"]),
        },
        {
            "step": 2,
            "selection_type": ("nearest_initial"),
            "selected_index": (second_index),
            "camera_position": (candidate_map[second_index]["camera_position"]),
            "distance_to_first_camera": (nearest_distance),
            "coverage_distance": (initial_distance),
            "surface_coverage": (initial_coverage),
        },
    ]

    print()

    print("[VIN-NBV] " f"Initial random view: " f"{first_index:03d}")

    print("[VIN-NBV] " f"Nearest second view: " f"{second_index:03d}")

    print("[VIN-NBV] " f"Initial distance: " f"{initial_distance:.6f}")

    print(
        "[VIN-NBV] "
        f"Initial coverage "
        f"@{coverage_threshold}: "
        f"{initial_coverage:.4f}"
    )

    print()

    while len(selected_indices) < num_views:
        step = len(selected_indices) + 1

        remaining_indices = [
            index for index in candidate_indices if index not in selected_indices
        ]

        distance_before = coverage_distance(current_distances)

        coverage_before = surface_coverage(
            current_distances,
            threshold=(coverage_threshold),
        )

        best_index = None

        best_rri = -float("inf")

        best_distance_after = None

        best_distances = None

        candidate_scores = []

        for index in remaining_indices:
            (
                rri,
                distance_after,
                proposed_distances,
            ) = evaluate_candidate(
                current_distances=(current_distances),
                candidate_distances=(profiles[index]),
            )

            coverage_after = surface_coverage(
                proposed_distances,
                threshold=(coverage_threshold),
            )

            candidate_scores.append(
                {
                    "index": (index),
                    "rri": (rri),
                    "distance_after": (distance_after),
                    "coverage_after": (coverage_after),
                }
            )

            if rri > best_rri:
                best_index = index

                best_rri = rri

                best_distance_after = distance_after

                best_distances = proposed_distances

        if best_index is None:
            raise RuntimeError("Failed to select " "next best view.")

        selected_indices.append(best_index)

        current_distances = best_distances

        coverage_after = surface_coverage(
            current_distances,
            threshold=(coverage_threshold),
        )

        coverage_gain = coverage_after - coverage_before

        candidate_scores.sort(
            key=lambda item: (item["rri"]),
            reverse=True,
        )

        history.append(
            {
                "step": (step),
                "selection_type": ("oracle_rri"),
                "selected_index": (best_index),
                "camera_position": (candidate_map[best_index]["camera_position"]),
                "distance_before": (distance_before),
                "distance_after": (best_distance_after),
                "rri": (best_rri),
                "coverage_before": (coverage_before),
                "coverage_after": (coverage_after),
                "coverage_gain": (coverage_gain),
                "top_candidates": (candidate_scores[:10]),
            }
        )

        print("[VIN-NBV] " f"Step {step}/{num_views}: " f"view {best_index:03d}")

        print("[VIN-NBV] " f"RRI: " f"{best_rri:.6f}")

        print(
            "[VIN-NBV] "
            "Coverage distance: "
            f"{distance_before:.6f} "
            "-> "
            f"{best_distance_after:.6f}"
        )

        print(
            "[VIN-NBV] "
            f"Coverage "
            f"@{coverage_threshold}: "
            f"{coverage_before:.4f} "
            "-> "
            f"{coverage_after:.4f}"
        )

        print()

    final_distance = coverage_distance(current_distances)

    final_coverage = surface_coverage(
        current_distances,
        threshold=(coverage_threshold),
    )

    result = {
        "baseline": ("vin_nbv_oracle"),
        "selection_mode": ("oracle_geometry_rri"),
        "seed": int(seed),
        "num_candidates": len(candidate_indices),
        "num_selected": len(selected_indices),
        "coverage_threshold": float(coverage_threshold),
        "selected_indices": (selected_indices),
        "final_coverage_distance": (final_distance),
        "final_surface_coverage": (final_coverage),
        "iterations": (history),
    }

    return result
