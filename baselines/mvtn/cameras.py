import numpy as np


def camera_positions(azimuth, elevation, distance):
    azimuth, elevation, distance = np.broadcast_arrays(
        np.asarray(azimuth, dtype=np.float64),
        np.asarray(elevation, dtype=np.float64),
        np.asarray(distance, dtype=np.float64),
    )
    if azimuth.ndim > 1 or azimuth.size == 0:
        raise ValueError("Camera parameters must be scalars or non-empty 1D arrays")
    if not all(np.isfinite(x).all() for x in (azimuth, elevation, distance)):
        raise ValueError("Camera parameters must be finite")
    if (distance <= 0).any():
        raise ValueError("Camera distances must be positive")
    azimuth, elevation = np.deg2rad(azimuth), np.deg2rad(elevation)
    return np.stack(
        [
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
            distance * np.cos(elevation) * np.cos(azimuth),
        ],
        axis=-1,
    ).reshape(-1, 3)


def cameras_in_input_frame(positions, mesh_transform):
    positions = np.asarray(positions, dtype=np.float64)
    transform = np.asarray(mesh_transform, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions must have shape (M, 3), M >= 1")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("mesh_transform must be a finite 4x4 matrix")
    if not np.isfinite(positions).all():
        raise ValueError("Camera positions must be finite")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("mesh_transform must be affine")
    inverse = np.linalg.inv(transform)
    cameras = positions @ inverse[:3, :3].T + inverse[:3, 3]
    target = inverse[:3, 3].copy()
    return cameras, target


def blender_camera_matrix(position, target=(0, 0, 0), up=(0, 1, 0)):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64, copy=True)
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (position, target, up)
    ):
        raise ValueError("Camera position, target and up must be finite 3D vectors")
    forward = target - position
    length = np.linalg.norm(forward)
    if length <= 1e-12 or np.linalg.norm(up) <= 1e-12:
        raise ValueError("Camera direction and up must be non-zero")
    forward /= length
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    fallback = np.linalg.norm(right) <= 1e-8
    if fallback:
        up = np.eye(3)[np.argmin(np.abs(forward))]
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, camera_up, -forward])
    matrix[:3, 3] = position
    return matrix, bool(fallback)
