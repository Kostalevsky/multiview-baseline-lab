import subprocess
from pathlib import Path

from baselines.cad2tech.output import create_collage
from utils.blender import find_blender


class ISOBaseline:
    name = "iso"

    def run(self, input_path: str):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")

        project_root = Path(__file__).resolve().parents[2]

        blender_path = find_blender()

        render_script = project_root / "rendering" / "render_iso_views.py"

        model_name = input_path.stem

        output_dir = project_root / "outputs" / model_name / "iso"

        selected_dir = output_dir / "selected"
        collage_path = output_dir / "collage.png"

        selected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 50)
        print("ISO baseline")
        print("=" * 50)

        print(f"[ISO] Input model: {input_path}")
        print(f"[ISO] Output dir: {output_dir}")

        print()
        print("[ISO] Step 1/2: rendering 6 orthographic views")

        result = subprocess.run(
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

        if result.returncode != 0:
            raise RuntimeError("ISO rendering failed.")

        view_names = [
            "front",
            "back",
            "left",
            "right",
            "top",
            "bottom",
        ]

        view_paths = [selected_dir / f"{name}.png" for name in view_names]

        missing = [path for path in view_paths if not path.exists()]

        if missing:
            raise RuntimeError(f"Missing ISO views: {missing}")

        print("[ISO] Generated 6 orthographic views")

        print()
        print("[ISO] Step 2/2: creating collage")

        create_collage(
            view_paths,
            collage_path,
            columns=3,
            labels=[
                f"#{position} | {name}"
                for position, name in enumerate(view_names, start=1)
            ],
        )

        print(f"[ISO] Collage saved to: " f"{collage_path}")

        print()
        print("[ISO] Done.")

        return {
            "baseline": self.name,
            "input": str(input_path),
            "num_views": 6,
            "views": view_names,
            "output_dir": str(output_dir),
            "collage": str(collage_path),
        }
