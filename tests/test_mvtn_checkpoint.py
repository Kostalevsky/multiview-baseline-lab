import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.mvtn.checkpoint import (
    FEATURE_PREFIX,
    VIEW_PREFIX,
    extract_mvtn_state,
)
from baselines.mvtn.predictor import MVTNPredictor


class CheckpointStructureTests(unittest.TestCase):
    def test_reject_classifier_and_pointnet_only(self):
        for checkpoint in (
            {},
            {"net.conv1.weight": np.zeros((64, 3, 7, 7))},
            {"state_dict": {"conv1.0.weight": np.zeros((64, 3, 1))}},
            {"mvtn": {}},
            {"mvtn": {FEATURE_PREFIX + "conv1.0.weight": np.zeros((64, 3, 1))}},
        ):
            with self.subTest(keys=list(checkpoint)), self.assertRaises(ValueError):
                extract_mvtn_state(checkpoint)

    def test_camera_count_comes_from_matching_buffers(self):
        state = {
            FEATURE_PREFIX + "conv1.0.weight": np.zeros((64, 3, 1)),
            VIEW_PREFIX + "view_transformer.0.0.weight": np.zeros((40, 52)),
            **{
                VIEW_PREFIX + name: np.zeros(6)
                for name in ("views_azim", "views_elev", "views_dist")
            },
        }
        _, count = extract_mvtn_state({"mvtn": state})
        self.assertEqual(count, 6)
        state[VIEW_PREFIX + "views_dist"] = np.zeros(12)
        with self.assertRaisesRegex(ValueError, "matching"):
            extract_mvtn_state({"mvtn": state})

    def test_predictor_requires_checkpoint_construction(self):
        with self.assertRaisesRegex(RuntimeError, "from_checkpoint"):
            MVTNPredictor().predict(np.zeros((2048, 3)))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class CheckpointRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch

        from baselines.mvtn.model import OffsetMVTN

        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.pth"
        self.model = OffsetMVTN(6)
        selector = self.model.view_selector.chosen_view_selector
        with torch.no_grad():
            selector.views_azim.copy_(torch.tensor([-180, -120, -60, 0, 60, 120]))
            selector.views_elev.fill_(30)
            selector.views_dist.fill_(2.2)
            for parameter in selector.view_transformer.parameters():
                parameter.zero_()
            selector.view_transformer[1][0].bias[:6].fill_(0.5)
            selector.view_transformer[1][0].bias[6:].fill_(-0.25)
        torch.save({"mvtn": self.model.state_dict()}, self.path)
        rng = np.random.default_rng(42)
        self.points = rng.normal(size=(128, 3)).astype(np.float32)
        self.points -= self.points.mean(0)
        self.points /= np.linalg.norm(self.points, axis=1).max()

    def test_load_freeze_and_known_camera_offsets(self):
        rng_state = self.torch.get_rng_state().clone()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        predictor = MVTNPredictor.from_checkpoint(self.path, expected_sha256=digest)
        self.assertTrue(self.torch.equal(rng_state, self.torch.get_rng_state()))
        self.assertFalse(predictor._model.training)
        self.assertTrue(all(not p.requires_grad for p in predictor._model.parameters()))
        result = predictor.predict(self.points)
        repeated = predictor.predict(self.points)
        np.testing.assert_allclose(
            result["azimuth"],
            np.array([-180, -120, -60, 0, 60, 120]) + 30 * np.tanh(0.5),
            atol=2e-5,
        )
        np.testing.assert_allclose(
            result["elevation"], 30 - 90 * np.tanh(0.25), atol=2e-5
        )
        np.testing.assert_allclose(np.linalg.norm(result["positions"], axis=1), 2.2)
        np.testing.assert_array_equal(result["azimuth"], repeated["azimuth"])

    def test_reject_wrong_hash_missing_layer_and_nonfinite_weights(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MVTNPredictor.from_checkpoint(self.path, expected_sha256="0" * 64)
        state = self.model.state_dict()
        removed = state.pop(FEATURE_PREFIX + "conv1.0.weight")
        self.torch.save({"mvtn": state}, self.path)
        with self.assertRaisesRegex(ValueError, "does not match"):
            MVTNPredictor.from_checkpoint(self.path)
        state[FEATURE_PREFIX + "conv1.0.weight"] = removed
        state[VIEW_PREFIX + "views_dist"][0] = float("nan")
        self.torch.save({"mvtn": state}, self.path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MVTNPredictor.from_checkpoint(self.path)

    def test_reject_wrong_input_normalization(self):
        predictor = MVTNPredictor.from_checkpoint(self.path)
        for points in (self.points + 2, self.points * 0.5, np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                predictor.predict(points)


if __name__ == "__main__":
    unittest.main()
