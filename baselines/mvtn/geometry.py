from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class PreparedGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    points: np.ndarray
    mesh_transform: np.ndarray
    points_transform: np.ndarray


def input_to_mvtn_rotation(input_up_axis):
    if input_up_axis == "y":
        return np.eye(3)
    if input_up_axis == "z":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    raise ValueError("input_up_axis must be 'y' or 'z'")


def normalize_points(points, rotation):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("Expected at least two 3D points")
    if not np.isfinite(points).all():
        raise ValueError("Geometry contains non-finite coordinates")
    rotated = points @ rotation.T
    center = rotated.mean(axis=0)
    centered = rotated - center
    radius = np.linalg.norm(centered, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Geometry has zero or invalid radius")
    transform = np.eye(4)
    transform[:3, :3] = rotation / radius
    transform[:3, 3] = -center / radius
    return (centered / radius).astype(np.float32), transform


def prepare_geometry(input_path, *, input_up_axis, num_points=2048, seed=42):
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, int)
        or num_points < 2
    ):
        raise ValueError("num_points must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rotation = input_to_mvtn_rotation(input_up_axis)
    input_path = Path(input_path).resolve(strict=True)
    if input_path.suffix.lower() != ".obj":
        raise ValueError("MVTN preparation currently accepts only OBJ files")
    scene = trimesh.load_scene(input_path, process=False, skip_materials=True)
    mesh = scene.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Input has no triangular mesh surface")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite coordinates")
    if not np.isfinite(mesh.area_faces).all() or mesh.area <= 0:
        raise ValueError("Mesh has zero or invalid surface area")
    vertices, mesh_transform = normalize_points(mesh.vertices, rotation)
    sampled, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    points, points_transform = normalize_points(sampled, rotation)
    return PreparedGeometry(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        points=points,
        mesh_transform=mesh_transform,
        points_transform=points_transform,
    )
