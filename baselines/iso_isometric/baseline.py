from pathlib import Path
import subprocess

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

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            output_dir
            / "isometric.png"
        )

        print("[ISO Isometric] Starting baseline")
        print(
            f"[ISO Isometric] Blender: {blender_path}"
        )

        subprocess.run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(render_script),
                "--",
                str(input_path),
                str(image_path),
            ],
            check=True,
        )

        if not image_path.exists():
            raise RuntimeError(
                "Isometric rendering was not created."
            )

        print(
            "[ISO Isometric] Baseline completed."
        )

        return {
            "baseline": self.name,
            "input": str(input_path),
            "num_views": 1,
            "view": "front-right-top",
            "projection": "isometric",
            "output_dir": str(output_dir),
            "image": str(image_path),
        }