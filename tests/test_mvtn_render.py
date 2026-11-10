import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from baselines.mvtn.cameras import blender_camera_matrix
from baselines.mvtn.render import load_render_input, render_views


class MVTNRenderTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MVTN_BLENDER_TEST") == "1",
        "Set MVTN_BLENDER_TEST=1 to run Blender",
    )
    def test_real_blender_render_and_projection(self):
        import torch
        import trimesh

        from baselines.mvtn.model import OffsetMVTN

        with tempfile.TemporaryDirectory(prefix="mvtn-blender-test-") as tmp:
            root = Path(tmp)
            parts = []
            for extents, position in [
                ([1.2, 0.18, 0.18], [0, -0.35, 0]),
                ([0.18, 0.95, 0.18], [-0.51, 0.05, 0]),
                ([0.3, 0.32, 0.5], [0.35, 0.15, 0.2]),
            ]:
                part = trimesh.creation.box(extents=extents)
                part.apply_translation(position)
                parts.append(part)
            source = root / "asymmetric.obj"
            trimesh.util.concatenate(parts).export(source)
            with torch.random.fork_rng(devices=[]):
                model = OffsetMVTN(3)
            selector = model.view_selector.chosen_view_selector
            with torch.no_grad():
                selector.views_azim.copy_(torch.tensor([0.0, 90.0, 0.0]))
                selector.views_elev.copy_(torch.tensor([0.0, 0.0, 90.0]))
                selector.views_dist.fill_(3.0)
                for parameter in selector.view_transformer.parameters():
                    parameter.zero_()
            checkpoint = root / "synthetic_only.pth"
            torch.save({"mvtn": model.state_dict()}, checkpoint)
            project_root = Path(__file__).resolve().parents[1]
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(project_root / "run.py"),
                    "--baseline",
                    "mvtn",
                    "--input",
                    str(source),
                    "--checkpoint",
                    str(checkpoint),
                    "--input-up-axis",
                    "y",
                    "--output-dir",
                    str(root / "run"),
                    "--mvtn-image-size",
                    "128",
                    "--mvtn-samples",
                    "4",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((root / "run/render/render_metadata.json").read_text())
            metadata = result["renderer"]
            self.assertFalse(result["paper_renderer_reproduced"])
            self.assertFalse(result["view_quality_evaluated"])
            self.assertEqual(result["checkpoint"]["training_dataset"], "unknown")
            np.testing.assert_array_equal(metadata["mesh_matrix_world"], np.eye(4))
            with np.load(root / "run/prediction/geometry.npz") as geometry:
                np.testing.assert_allclose(
                    metadata["mesh_bounds"]["min"], geometry["vertices"].min(0)
                )
                np.testing.assert_allclose(
                    metadata["mesh_bounds"]["max"], geometry["vertices"].max(0)
                )
            self.assertEqual(
                [pose["up_fallback"] for pose in metadata["views"]],
                [False, False, True],
            )
            for pose in metadata["views"]:
                np.testing.assert_allclose(
                    pose["actual_matrix_world"], pose["matrix_world"], atol=1e-6
                )
                inverse = np.linalg.inv(np.array(pose["actual_matrix_world"]))
                for probe in pose["projection_probes"]:
                    local = inverse @ [*probe["point"], 1]
                    depth = -local[2]
                    expected = [
                        0.5 + 0.5 * local[0] / (depth * np.tan(np.pi / 6)),
                        0.5 + 0.5 * local[1] / (depth * np.tan(np.pi / 6)),
                        depth,
                    ]
                    np.testing.assert_allclose(probe["ndc"], expected, atol=1e-6)
                with Image.open(pose["image"]) as image:
                    self.assertEqual(image.size, (128, 128))
                    array = np.array(image)
                    self.assertGreater(int(array.max()) - int(array.min()), 50)
            self.assertTrue(Path(result["collage"]).is_file())

    def test_camera_frame_preserves_target_up_and_right(self):
        for position in ([0, 0, 3], [3, 1, 0], [-2, -1, 2]):
            up = np.array([0.0, 1, 0])
            matrix, fallback = blender_camera_matrix(position, up=up)
            self.assertFalse(fallback)
            np.testing.assert_array_equal(up, [0, 1, 0])
            rotation = matrix[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            local_target = np.linalg.inv(matrix) @ [0, 0, 0, 1]
            np.testing.assert_allclose(local_target[:2], [0, 0], atol=1e-12)
            self.assertLess(local_target[2], 0)
            self.assertGreater(np.dot(rotation[:, 1], up), 0)
        front, _ = blender_camera_matrix([0, 0, 3])
        np.testing.assert_allclose(np.linalg.inv(front) @ [1, 1, 0, 1], [1, 1, -3, 1])

    def test_poles_use_a_finite_deterministic_fallback(self):
        for position in ([0, 3, 0], [0, -3, 0]):
            matrix, fallback = blender_camera_matrix(position)
            self.assertTrue(fallback)
            self.assertTrue(np.isfinite(matrix).all())
            np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
            repeated, _ = blender_camera_matrix(position)
            np.testing.assert_array_equal(matrix, repeated)
        with self.assertRaises(ValueError):
            blender_camera_matrix([0, 0, 0])

    def write_fixture(self, root):
        geometry = root / "geometry.npz"
        np.savez(
            geometry,
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            faces=np.array([[0, 1, 2]]),
        )
        manifest = {
            "schema_version": 1,
            "stage": "camera_prediction_only",
            "geometry": str(geometry),
            "geometry_sha256": hashlib.sha256(geometry.read_bytes()).hexdigest(),
            "target_mvtn": [0, 0, 0],
            "up_mvtn": [0, 1, 0],
            "num_views": 1,
            "views": [
                {
                    "index": 0,
                    "position_mvtn": [0, 0, 3],
                    "azimuth_degrees": 0,
                    "elevation_degrees": 0,
                    "distance": 3,
                }
            ],
        }
        path = root / "cameras.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_manifest_hash_and_camera_consistency_before_blender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, manifest = self.write_fixture(root)
            _, request = load_render_input(path)
            self.assertEqual(len(request["poses"]), 1)
            manifest["views"][0]["position_mvtn"] = [1, 0, 3]
            path.write_text(json.dumps(manifest))
            with patch("baselines.mvtn.render.find_blender") as blender:
                with self.assertRaisesRegex(ValueError, "spherical"):
                    render_views(path, root / "output")
                blender.assert_not_called()
                self.assertFalse((root / "output").exists())
            manifest["views"][0]["position_mvtn"] = [0, 0, 3]
            path.write_text(json.dumps(manifest))
            with (root / "geometry.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_render_input(path)

    def test_existing_output_and_invalid_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.write_fixture(root)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                render_views(path, output)
            self.assertEqual(sentinel.read_text(), "keep")
            for options in ({"image_size": 0}, {"samples": 0}, {"image_size": True}):
                with self.assertRaises(ValueError):
                    render_views(path, root / "new", **options)
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
