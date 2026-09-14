from pathlib import Path

import numpy as np
import trimesh

DEFAULT_NUM_GT_POINTS = 100_000


def load_mesh(mesh_path):
    mesh_path = Path(mesh_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(
        mesh_path,
        force="mesh",
        process=False,
    )

    if mesh.is_empty:
        raise RuntimeError(f"Loaded mesh is empty: {mesh_path}")

    return mesh


def normalize_mesh(
    mesh,
    target_max_extent=1.0,
):
    """
    Apply the same normalization convention as the Blender renderer:
    center the bounding box at the origin and scale the largest
    extent to target_max_extent.
    """

    mesh = mesh.copy()

    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]

    center = (bbox_min + bbox_max) * 0.5

    extents = bbox_max - bbox_min

    max_extent = float(np.max(extents))

    if max_extent <= 0.0:
        raise RuntimeError("Mesh has invalid bounding-box extent.")

    scale = target_max_extent / max_extent

    mesh.vertices = (mesh.vertices - center) * scale

    return mesh, {
        "bbox_min": (bbox_min.tolist()),
        "bbox_max": (bbox_max.tolist()),
        "center": (center.tolist()),
        "extents": (extents.tolist()),
        "scale": float(scale),
    }


def sample_ground_truth_surface(
    mesh_path,
    num_points=DEFAULT_NUM_GT_POINTS,
    seed=42,
):
    """
    Sample points uniformly from the normalized mesh surface.

    Returns
    -------
    np.ndarray
        Ground-truth surface points with shape (N, 3).

    dict
        Sampling / normalization metadata.
    """

    mesh = load_mesh(mesh_path)

    mesh, normalization = normalize_mesh(mesh)

    np.random.seed(seed)

    points, face_indices = trimesh.sample.sample_surface(
        mesh,
        num_points,
    )

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    return points, {
        "num_points": len(points),
        "seed": int(seed),
        "normalization": (normalization),
        "sampled_faces": len(np.unique(face_indices)),
    }
