import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cameras import cameras_in_input_frame
from .predictor import MVTNPredictor
from .prepare import prepare_inputs


def predict_views(
    input_path,
    checkpoint_path,
    output_dir,
    *,
    input_up_axis,
    num_points=2048,
    seed=42,
    expected_sha256=None,
):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    predictor = MVTNPredictor.from_checkpoint(
        checkpoint_path, expected_sha256=expected_sha256
    )
    preparation = prepare_inputs(
        input_path,
        output_dir,
        input_up_axis=input_up_axis,
        num_points=num_points,
        seed=seed,
    )
    geometry_path = output_dir / "geometry.npz"
    with np.load(geometry_path, allow_pickle=False) as geometry:
        prediction = predictor.predict(geometry["points"])
    input_positions, input_target = cameras_in_input_frame(
        prediction["positions"], preparation["mesh_transform"]
    )
    views = []
    for index in range(predictor.checkpoint_info["num_views"]):
        views.append(
            {
                "index": index,
                "azimuth_degrees": float(prediction["azimuth"][index]),
                "elevation_degrees": float(prediction["elevation"][index]),
                "distance": float(prediction["distance"][index]),
                "position_mvtn": prediction["positions"][index].tolist(),
                "position_input": input_positions[index].tolist(),
            }
        )
    result = {
        "schema_version": 1,
        "stage": "camera_prediction_only",
        "input": preparation["input"],
        "input_sha256": preparation["input_sha256"],
        "input_up_axis": input_up_axis,
        "num_points": num_points,
        "seed": seed,
        "checkpoint": predictor.checkpoint_info,
        "geometry": str(geometry_path),
        "geometry_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        "preparation_metadata": str(output_dir / "metadata.json"),
        "normalization": preparation["normalization"],
        "mesh_transform": preparation["mesh_transform"],
        "points_transform": preparation["points_transform"],
        "target_mvtn": [0.0, 0.0, 0.0],
        "up_mvtn": [0.0, 1.0, 0.0],
        "target_input": input_target.tolist(),
        "num_views": len(views),
        "views": views,
        "renders_generated": False,
        "view_quality_evaluated": False,
    }
    with (output_dir / "cameras.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict MVTN cameras from an OBJ")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-up-axis", required=True, choices=["y", "z"])
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    try:
        result = predict_views(
            args.input,
            args.checkpoint,
            args.output_dir,
            input_up_axis=args.input_up_axis,
            num_points=args.num_points,
            seed=args.seed,
            expected_sha256=args.sha256,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"MVTN prediction failed: {exc}\n")
    print(
        f"Predicted {result['num_views']} cameras: "
        f"{args.output_dir.resolve() / 'cameras.json'}. "
        "Training provenance is not verified; no renders generated."
    )


if __name__ == "__main__":
    main()
