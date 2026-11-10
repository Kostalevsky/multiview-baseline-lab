import json
import subprocess
from pathlib import Path

from baselines.cad2tech.output import create_collage
from utils.blender import find_blender


def render_fixed_views(name, views, input_path, output_dir=None, details=None):
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input OBJ not found: {input_path}")

    project_root = Path(__file__).resolve().parents[1]
    if output_dir is None:
        output_dir = project_root / "outputs" / input_path.stem / name
    output_dir = Path(output_dir).resolve()
    blender = find_blender()
    views_dir = output_dir / "selected"
    output_dir.mkdir(parents=True, exist_ok=False)
    views_dir.mkdir()

    subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(project_root / "rendering" / "render_mvcnn_views.py"),
            "--",
            str(input_path),
            str(output_dir),
            json.dumps(views),
        ],
        check=True,
    )

    collage_path = create_collage(
        [output_dir / view["image"] for view in views],
        output_dir / "collage.png",
        columns=3,
        labels=[
            f"Az {view['azimuth_deg']:.1f}°  El {view['elevation_deg']:.1f}°"
            for view in views
        ],
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "baseline": name,
                "input": str(input_path),
                "num_views": len(views),
                "projection": "perspective",
                "views": views,
                **(details or {}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "baseline": name,
        "num_views": len(views),
        "output_dir": str(output_dir),
        "collage": str(collage_path),
        "metadata": str(metadata_path),
    }
