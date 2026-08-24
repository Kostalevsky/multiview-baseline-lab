import subprocess
from pathlib import Path

from baselines.cad2tech.output import create_collage
from utils.blender import find_blender


class ISOIsometricBaseline:
    name = "iso_isometric"

    def run(self, input_path):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input model not found: {input_path}"
            )

        project_root = Path(__file__).resolve().parents[2]

        blender_path = find_blender()

        render_script = (
            project_root
            / "rendering"
            / "render_iso_isometric.py"
        )

        model_name = input_path.stem

        output_dir = (
            project_root
            / "outputs"
            / model_name
            / "iso_isometric"
        )

        selected_dir = output_dir / "selected"
        collage_path = output_dir / "collage.png"

        selected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("[ISO Isometric] Starting baseline")
        print(f"[ISO Isometric] Blender: {blender_path}")

        subprocess.run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(render_script),
                "--",
                str(input_path),
                str(selected_dir),
            ],
            check=True,
        )

        view_names = [
            "front_right_top",
            "front_left_top",
            "back_right_top",
            "back_left_top",
        ]

        view_paths = [
            selected_dir / f"{view_name}.png"
            for view_name in view_names
        ]

        missing_views = [
            path
            for path in view_paths
            if not path.exists()
        ]

        if missing_views:
            raise RuntimeError(
                f"Missing isometric views: {missing_views}"
            )

        create_collage(
            view_paths,
            collage_path,
            columns=2,
        )

        print(
            "[ISO Isometric] Baseline completed."
        )

        return {
            "baseline": self.name,
            "input": str(input_path),
            "num_views": 4,
            "views": view_names,
            "projection": "isometric",
            "output_dir": str(output_dir),
            "collage": str(collage_path),
        }