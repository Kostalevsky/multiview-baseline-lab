import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from baselines.mvtn.cameras import camera_positions, cameras_in_input_frame
from baselines.mvtn.geometry import (
    input_to_mvtn_rotation,
    normalize_points,
    prepare_geometry,
)
from baselines.mvtn.prepare import prepare_inputs


class MVTNGeometryTests(unittest.TestCase):
    def test_mean_center_and_radius_with_recoverable_transform(self):
        original = np.array([[2, 3, 5], [4, 3, 5], [2, 7, 5], [2, 3, 11]])
        normalized, transform = normalize_points(original, input_to_mvtn_rotation("z"))
        np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(normalized, axis=1).max()), 1)
        homogeneous = np.column_stack([original, np.ones(len(original))])
        np.testing.assert_allclose(
            (homogeneous @ transform.T)[:, :3], normalized, atol=1e-7
        )
        recovered, center = cameras_in_input_frame(normalized, transform)
        np.testing.assert_allclose(recovered, original, atol=1e-6)
        np.testing.assert_allclose(center, original.mean(0))

    def test_pytorch3d_cardinal_camera_directions(self):
        positions = camera_positions([0, 90, 180, -90, 0], [0, 0, 0, 0, 90], 2)
        np.testing.assert_allclose(
            positions,
            [[0, 0, 2], [2, 0, 0], [0, 0, -2], [-2, 0, 0], [0, 2, 0]],
            atol=1e-12,
        )
        original = np.array([[-1, -1, -1], [1, 1, 1]])
        _, transform = normalize_points(original, input_to_mvtn_rotation("z"))
        cameras, target = cameras_in_input_frame(positions[[4]], transform)
        np.testing.assert_allclose(cameras, [[0, 0, 2 * np.sqrt(3)]], atol=1e-12)
        np.testing.assert_allclose(target, [0, 0, 0])

    def test_offsets_beyond_poles_keep_distance(self):
        positions = camera_positions([0, 45], [110, -120], [2.2, 3.1])
        np.testing.assert_allclose(np.linalg.norm(positions, axis=1), [2.2, 3.1])
        self.assertEqual(camera_positions(0, 0, 2.2).shape, (1, 3))

    def test_reject_invalid_geometry_and_cameras(self):
        for points in ([[1, 1, 1]] * 2, [[0, 0, 0], [np.nan, 1, 2]], []):
            with self.assertRaises(ValueError):
                normalize_points(points, np.eye(3))
        with self.assertRaises(ValueError):
            input_to_mvtn_rotation("x")
        for azimuth, elevation, distance in (
            (0, 0, 0),
            (0, 0, -1),
            (np.nan, 0, 1),
            ([], [], []),
            ([[0]], 0, 1),
        ):
            with self.assertRaises(ValueError):
                camera_positions(azimuth, elevation, distance)

    def test_surface_sampling_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "asymmetric.obj"
            mesh = trimesh.creation.box(extents=[1, 2, 4])
            mesh.apply_translation([10, 20, 30])
            mesh.export(model)
            input_bytes = model.read_bytes()
            np.random.seed(7)
            state = np.random.get_state()
            first = prepare_geometry(model, input_up_axis="z", num_points=2048)
            second = prepare_geometry(model, input_up_axis="z", num_points=2048)
            state_after = np.random.get_state()
            np.testing.assert_array_equal(state[1], state_after[1])
            self.assertEqual(state[2:], state_after[2:])
            np.testing.assert_array_equal(first.points, second.points)
            different = prepare_geometry(
                model, input_up_axis="z", num_points=2048, seed=43
            )
            self.assertFalse(np.array_equal(first.points, different.points))
            sampled, _ = cameras_in_input_frame(first.points, first.points_transform)
            relative = (sampled - [10, 20, 30]) / [0.5, 1, 2]
            np.testing.assert_allclose(np.abs(relative).max(1), 1, atol=1e-6)
            output = root / "prepared"
            prepare_inputs(model, output, input_up_axis="z")
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(
                metadata["input_sha256"], hashlib.sha256(input_bytes).hexdigest()
            )
            self.assertFalse(metadata["learned_views_generated"])
            self.assertEqual(metadata["stage"], "input_preparation_only")
            with np.load(output / "geometry.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["points"], first.points)
                np.testing.assert_array_equal(saved["faces"], first.faces)
            saved_bytes = (output / "geometry.npz").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_inputs(model, output, input_up_axis="z")
            self.assertEqual(saved_bytes, (output / "geometry.npz").read_bytes())
            self.assertEqual(input_bytes, model.read_bytes())


if __name__ == "__main__":
    unittest.main()
