import ast
import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE_HASHES = {
    "models/blocks.py": "f7023312caa498ac8d287a6f23f5f180314253be",
    "models/pointnet.py": "5944164e9e30438946d82b081aaeb371872d5fdb",
    "models/mvtn.py": "44a2f8a95ab47724c6ea1625b647054f99806630",
    "util.py": "4d433bc5157b16437168a441942e7e01d421d89c",
}


def load_reference(root):
    for relative, expected in SOURCE_HASHES.items():
        content = (root / relative).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(content)).encode() + b"\0" + content
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source checksum mismatch: {relative}")
    source = ast.parse((root / "util.py").read_text())
    names = {"unit_spherical_grid", "batch_tensor"}
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("Missing upstream utility functions")
    helper_source = "import numpy as np\nimport torch\n\n" + ast.unparse(
        ast.Module(body=functions, type_ignores=[])
    )
    with tempfile.TemporaryDirectory(prefix="mvtn-reference-helpers-") as tmp:
        helper_path = Path(tmp) / "util.py"
        helper_path.write_text(helper_source)
        helper_spec = importlib.util.spec_from_file_location("util", helper_path)
        helpers = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helpers)
    package = types.ModuleType("mvtn_reference")
    package.__path__ = [str(root / "models")]
    with patch.dict(
        sys.modules, {"util": helpers, "mvtn_reference": package}
    ), patch.object(sys, "path", sys.path.copy()):
        for name in ("blocks", "pointnet", "mvtn"):
            full_name = f"mvtn_reference.{name}"
            spec = importlib.util.spec_from_file_location(
                full_name, root / "models" / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        return module.MVTN


@unittest.skipUnless(
    os.environ.get("MVTN_UPSTREAM_DIR") and importlib.util.find_spec("torch"),
    "Set MVTN_UPSTREAM_DIR to pinned upstream sources and install PyTorch",
)
class UpstreamParityTests(unittest.TestCase):
    def test_pointnet_and_cameras_match_upstream(self):
        import torch

        from baselines.mvtn.model import OffsetMVTN

        reference_type = load_reference(Path(os.environ["MVTN_UPSTREAM_DIR"]))
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for views in (4, 6, 12):
                for variant in ("learned_circular", "learned_spherical"):
                    with self.subTest(views=views, variant=variant):
                        torch.manual_seed(20261101)
                        reference = reference_type(
                            nb_views=views,
                            views_config=variant,
                            canonical_elevation=30,
                            canonical_distance=2.2,
                            shape_extractor="PointNet",
                            shape_features_size=40,
                            screatch_feature_extractor=True,
                        ).eval()
                        actual = OffsetMVTN(views).eval()
                        actual.load_state_dict(reference.state_dict(), strict=True)
                        points = torch.randn(2, 2048, 3)
                        points -= points.mean(dim=1, keepdim=True)
                        points /= points.norm(dim=2).amax(dim=1).reshape(-1, 1, 1)
                        with torch.inference_mode():
                            expected_features = reference.feature_extractor(points, 2)
                            actual_features = actual.feature_extractor(points)
                            expected = reference(points, c_batch_size=2)
                            result = actual(points)
                        torch.testing.assert_close(
                            actual_features, expected_features, atol=1e-6, rtol=1e-5
                        )
                        errors = []
                        for value, target in zip(result, expected):
                            torch.testing.assert_close(
                                value, target, atol=1e-6, rtol=1e-5
                            )
                            errors.append(float((value - target).abs().max()))
                        print(
                            f"MVTN parity: {variant}, M={views}, max_abs_errors={errors}"
                        )


if __name__ == "__main__":
    unittest.main()
