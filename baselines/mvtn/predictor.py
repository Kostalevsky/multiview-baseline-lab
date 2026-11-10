import numpy as np

from .cameras import camera_positions
from .checkpoint import load_checkpoint, require_torch


class MVTNPredictor:
    @classmethod
    def from_checkpoint(cls, path, *, expected_sha256=None):
        state, info = load_checkpoint(path, expected_sha256=expected_sha256)
        from .model import OffsetMVTN

        torch = require_torch()
        with torch.random.fork_rng(devices=[]):
            model = OffsetMVTN(info["num_views"])
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint does not match PointNet-40 MVTN with learned angular "
                "offsets and fixed distances. Direct, DGCNN and learned-distance "
                "variants are not supported. " + str(exc)
            ) from exc
        model.eval()
        model.requires_grad_(False)
        predictor = cls()
        predictor._model = model
        predictor.checkpoint_info = {
            **info,
            "architecture": "pointnet40_learned_angular_offsets_fixed_distance",
            "strict_state_loading": True,
            "device": "cpu",
        }
        return predictor

    def predict(self, points):
        if not hasattr(self, "_model"):
            raise RuntimeError("Construct MVTNPredictor using from_checkpoint()")
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Expected points with shape (P, 3), P >= 2")
        if not np.isfinite(points).all():
            raise ValueError("Input points must be finite")
        if not np.allclose(points.mean(axis=0), 0, atol=1e-4):
            raise ValueError("Input points must be mean-centered")
        if not np.isclose(np.linalg.norm(points, axis=1).max(), 1, atol=1e-4):
            raise ValueError("Input points must have maximum L2 radius 1")
        torch = require_torch()
        with torch.inference_mode():
            outputs = self._model(torch.from_numpy(points.copy()).unsqueeze(0))
        arrays = [output[0].cpu().numpy().copy() for output in outputs]
        azimuth, elevation, distance = arrays
        expected = (self.checkpoint_info["num_views"],)
        if any(value.shape != expected for value in arrays):
            raise RuntimeError(
                "MVTN returned an unexpected number of camera parameters"
            )
        return {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "positions": camera_positions(azimuth, elevation, distance),
        }
