import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from baselines.mvtn.predict import predict_views


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class MVTNPipelineTests(unittest.TestCase):
    def test_obj_to_camera_manifest_preserves_frames_and_provenance(self):
        import torch

        from baselines.mvtn.model import OffsetMVTN

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "asymmetric.obj"
            mesh = trimesh.creation.box(extents=[1, 2, 4])
            mesh.apply_translation([10, 20, 30])
            mesh.export(obj)
            obj_bytes = obj.read_bytes()
            with torch.random.fork_rng(devices=[]):
                model = OffsetMVTN(6)
            selector = model.view_selector.chosen_view_selector
            with torch.no_grad():
                selector.views_azim.copy_(torch.arange(6) * 60)
                selector.views_elev.fill_(30)
                selector.views_dist.fill_(2.2)
                for parameter in selector.view_transformer.parameters():
                    parameter.zero_()
            checkpoint = root / "synthetic.pth"
            torch.save({"mvtn": model.state_dict()}, checkpoint)
            output = root / "predicted"
            predict_views(obj, checkpoint, output, input_up_axis="z")
            result = json.loads((output / "cameras.json").read_text())
            self.assertEqual(result["num_views"], 6)
            self.assertEqual(result["stage"], "camera_prediction_only")
            self.assertFalse(result["renders_generated"])
            self.assertFalse(result["view_quality_evaluated"])
            self.assertEqual(result["checkpoint"]["training_dataset"], "unknown")
            self.assertEqual(
                result["input_sha256"], hashlib.sha256(obj_bytes).hexdigest()
            )
            self.assertEqual(
                result["checkpoint"]["sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["geometry_sha256"],
                hashlib.sha256((output / "geometry.npz").read_bytes()).hexdigest(),
            )
            transform = np.array(result["mesh_transform"])
            np.testing.assert_allclose(result["target_input"], [10, 20, 30])
            for view in result["views"]:
                original = np.array([*view["position_input"], 1])
                np.testing.assert_allclose(
                    (transform @ original)[:3], view["position_mvtn"], atol=1e-7
                )
            saved = (output / "cameras.json").read_bytes()
            with self.assertRaises(FileExistsError):
                predict_views(obj, checkpoint, output, input_up_axis="z")
            self.assertEqual((output / "cameras.json").read_bytes(), saved)
            self.assertEqual(obj.read_bytes(), obj_bytes)

    def test_invalid_checkpoint_creates_no_outputs(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "classifier.pth"
            torch.save({"net.conv1.weight": torch.zeros(1)}, checkpoint)
            output = root / "prediction"
            with self.assertRaisesRegex(ValueError, "mvtn"):
                predict_views(
                    root / "unused.obj", checkpoint, output, input_up_axis="z"
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
