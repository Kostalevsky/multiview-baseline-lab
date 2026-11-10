import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from .geometry import prepare_geometry


def prepare_inputs(input_path, output_dir, *, input_up_axis, num_points=2048, seed=42):
    input_path = Path(input_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    geometry = prepare_geometry(
        input_path, input_up_axis=input_up_axis, num_points=num_points, seed=seed
    )
    after = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Input changed during geometry preparation")
    metadata = {
        "schema_version": 1,
        "stage": "input_preparation_only",
        "learned_views_generated": False,
        "checkpoint": None,
        "input": str(input_path),
        "input_sha256": before,
        "input_up_axis": input_up_axis,
        "coordinate_frame": "mvtn_y_up",
        "normalization": "independent_mean_center_max_l2_radius",
        "num_points": num_points,
        "seed": seed,
        "num_vertices": len(geometry.vertices),
        "num_faces": len(geometry.faces),
        "mesh_transform": geometry.mesh_transform.tolist(),
        "points_transform": geometry.points_transform.tolist(),
        "versions": {"numpy": np.__version__, "trimesh": trimesh.__version__},
        "source": "https://github.com/ajhamdi/MVTN/blob/master/custom_dataset.py",
        "checkpoint_compatibility": "not_verified",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "geometry.npz",
        vertices=geometry.vertices,
        faces=geometry.faces,
        points=geometry.points,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare MVTN inputs only")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-up-axis", required=True, choices=["y", "z"])
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        metadata = prepare_inputs(
            args.input,
            args.output_dir,
            input_up_axis=args.input_up_axis,
            num_points=args.num_points,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"MVTN preparation failed: {exc}\n")
    print(
        f"Prepared {metadata['num_points']} points in {args.output_dir}. "
        "No learned cameras or renders generated."
    )


if __name__ == "__main__":
    main()
