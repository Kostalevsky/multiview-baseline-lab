import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from baselines.cad2tech.output import create_collage
from utils.blender import find_blender

from .cameras import blender_camera_matrix, camera_positions


def load_render_input(cameras_path):
    cameras_path = Path(cameras_path).resolve(strict=True)
    content = cameras_path.read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict):
        raise TypeError("Camera manifest must be a JSON object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "camera_prediction_only"
    ):
        raise ValueError("Expected a version 1 MVTN camera prediction manifest")
    views = manifest.get("views")
    if (
        not isinstance(views, list)
        or not views
        or type(manifest.get("num_views")) is not int
        or manifest["num_views"] != len(views)
    ):
        raise ValueError("Invalid number of camera views")
    target = np.asarray(manifest.get("target_mvtn"), dtype=float)
    up = np.asarray(manifest.get("up_mvtn"), dtype=float)
    if (
        target.shape != (3,)
        or up.shape != (3,)
        or not np.array_equal(target, [0, 0, 0])
        or not np.array_equal(up, [0, 1, 0])
    ):
        raise ValueError("Only centered Y-up MVTN geometry is supported")
    geometry_path = Path(manifest["geometry"])
    if not geometry_path.is_absolute():
        geometry_path = cameras_path.parent / geometry_path
    geometry_path = geometry_path.resolve(strict=True)
    with geometry_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest.get("geometry_sha256"):
        raise ValueError("Prepared geometry SHA-256 does not match cameras.json")
    with np.load(geometry_path, allow_pickle=False) as archive:
        vertices, faces = archive["vertices"], archive["faces"]
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Invalid prepared vertices")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces < 0).any()
            or (faces >= len(vertices)).any()
        ):
            raise ValueError("Invalid prepared triangular faces")
    poses = []
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise TypeError("Each camera view must be a JSON object")
        if type(view.get("index")) is not int or view["index"] != index:
            raise ValueError("Camera indices must be consecutive from zero")
        expected = camera_positions(
            view["azimuth_degrees"], view["elevation_degrees"], view["distance"]
        )
        position = np.asarray(view["position_mvtn"], dtype=float)
        if position.shape != (3,) or not np.allclose(
            position, expected[0], atol=1e-6, rtol=1e-6
        ):
            raise ValueError("Camera position does not match spherical parameters")
        matrix, fallback = blender_camera_matrix(position, target, up)
        poses.append(
            {"index": index, "matrix_world": matrix.tolist(), "up_fallback": fallback}
        )
    return manifest, {
        "cameras": str(cameras_path),
        "cameras_sha256": hashlib.sha256(content).hexdigest(),
        "geometry": str(geometry_path),
        "geometry_sha256": digest,
        "poses": poses,
    }


def render_views(cameras_path, output_dir, *, image_size=512, samples=32):
    if type(image_size) is not int or not 32 <= image_size <= 4096:
        raise ValueError("image_size must be an integer between 32 and 4096")
    if type(samples) is not int or not 1 <= samples <= 4096:
        raise ValueError("samples must be an integer between 1 and 4096")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    manifest, request = load_render_input(cameras_path)
    blender = find_blender()
    request.update(
        {
            "image_size": image_size,
            "samples": samples,
            "output_dir": str(output_dir),
            "fov_degrees": 60.0,
            "clip_start": 1.0,
            "clip_end": 100.0,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    request_path = output_dir / "render_request.json"
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
    script = Path(__file__).resolve().parents[2] / "rendering" / "render_mvtn_views.py"
    log_path = output_dir / "blender.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(script),
                "--",
                str(request_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(f"MVTN Blender rendering failed; see {log_path}")
    paths = [
        output_dir / "views" / f"view_{index:03d}.png"
        for index in range(len(request["poses"]))
    ]
    for path in paths:
        with Image.open(path) as image:
            image.load()
            if image.size != (image_size, image_size):
                raise RuntimeError(f"Unexpected render dimensions: {path}")
    collage = output_dir / "collage.png"
    create_collage(
        paths,
        collage,
        columns=min(3, len(paths)),
        labels=[f"MVTN view {index + 1}" for index in range(len(paths))],
    )
    worker_metadata = json.loads((output_dir / "blender_metadata.json").read_text())
    result = {
        "baseline": "mvtn",
        "stage": "blender_render_from_saved_cameras",
        "source_cameras": request["cameras"],
        "source_cameras_sha256": request["cameras_sha256"],
        "input": manifest.get("input"),
        "input_sha256": manifest.get("input_sha256"),
        "checkpoint": manifest.get("checkpoint"),
        "geometry_sha256": request["geometry_sha256"],
        "num_views": len(paths),
        "views": [str(path) for path in paths],
        "collage": str(collage),
        "renderer": worker_metadata,
        "renders_generated": True,
        "view_quality_evaluated": False,
        "paper_renderer_reproduced": False,
    }
    (output_dir / "render_metadata.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Render saved MVTN cameras in Blender")
    parser.add_argument("--cameras", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    try:
        result = render_views(
            args.cameras,
            args.output_dir,
            image_size=args.image_size,
            samples=args.samples,
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"MVTN render failed: {exc}\n")
    print(f"Rendered {result['num_views']} views. Collage: {result['collage']}")


if __name__ == "__main__":
    main()
