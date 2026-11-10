import argparse
import hashlib
import importlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

VIEW_PREFIX = "view_selector.chosen_view_selector."
FEATURE_PREFIX = "feature_extractor.fe_model."


def require_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError(
            "MVTN checkpoint loading requires PyTorch in the project .venv. "
            "Geometry preparation does not require it."
        ) from exc


def extract_mvtn_state(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a checkpoint dictionary with an 'mvtn' state")
    state = checkpoint.get("mvtn")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            "No non-empty 'mvtn' state found. ResNet/ViewGCN or standalone "
            "PointNet weights cannot predict camera viewpoints."
        )
    if not all(isinstance(key, str) for key in state):
        raise ValueError("MVTN state keys must be strings")
    if not any(key.startswith(FEATURE_PREFIX) for key in state):
        raise ValueError("MVTN checkpoint is missing the point encoder")
    if not any(key.startswith(VIEW_PREFIX + "view_transformer.") for key in state):
        raise ValueError("MVTN checkpoint is missing the learned camera regressor")
    buffers = [
        state.get(VIEW_PREFIX + name)
        for name in ("views_azim", "views_elev", "views_dist")
    ]
    shape = getattr(buffers[0], "shape", ())
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError("Invalid MVTN camera buffers")
    if any(tuple(getattr(value, "shape", ())) != tuple(shape) for value in buffers):
        raise ValueError("MVTN camera buffers must have matching 1D shapes")
    return state, int(shape[0])


def load_checkpoint(path, *, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    torch = require_torch()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise ValueError("Checkpoint SHA-256 does not match the expected value")
        stream.seek(0)
        try:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
            raise ValueError(
                f"Cannot read checkpoint with weights_only=True: {exc}"
            ) from exc
    state, num_views = extract_mvtn_state(checkpoint)
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("MVTN state must contain only tensors")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("MVTN checkpoint contains non-finite values")
    if (state[VIEW_PREFIX + "views_dist"] <= 0).any().item():
        raise ValueError("MVTN camera distances must be positive")
    return state, {
        "path": str(path),
        "sha256": digest,
        "num_views": num_views,
        "num_state_entries": len(state),
        "training_dataset": "unknown",
        "training_configuration": "not_verified",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a full MVTN checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    try:
        from .predictor import MVTNPredictor

        predictor = MVTNPredictor.from_checkpoint(
            args.checkpoint, expected_sha256=args.sha256
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"MVTN checkpoint validation failed: {exc}\n")
    print(json.dumps(predictor.checkpoint_info, indent=2))


if __name__ == "__main__":
    main()
