import json
import subprocess
from pathlib import Path

from baselines.canonical.output import (
    VIEW_ORDER,
    create_canonical_collage,
)
from utils.blender import find_blender


class CanonicalBaseline:
    def run(
        self,
        input_path,
        output_dir=None,
    ):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: " f"{input_path}")

        if output_dir is None:
            output_dir = Path("outputs") / input_path.stem / "canonical"
        else:
            output_dir = Path(output_dir)

        output_dir = output_dir.resolve()

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        views_dir = output_dir / "views"

        metadata_path = output_dir / "metadata.json"

        collage_path = output_dir / "collage.png"

        print()
        print("[Canonical] " "Starting baseline")

        print("[Canonical] " f"Input: {input_path}")

        print("[Canonical] " f"Output: {output_dir}")

        print()

        if self._renders_exist(
            views_dir=views_dir,
            metadata_path=(metadata_path),
        ):
            print("[Canonical] " "Rendered views already " "exist — skipping Blender")
        else:
            self._render_views(
                input_path=(input_path),
                output_dir=(output_dir),
            )

        print("[Canonical] " "Creating collage")

        create_canonical_collage(
            views_dir=(views_dir),
            output_path=(collage_path),
        )

        metadata = self._load_metadata(metadata_path)

        result = {
            "baseline": ("canonical"),
            "input": str(input_path),
            "num_views": 8,
            "view_names": (VIEW_ORDER),
            "views": (metadata["views"]),
            "output_dir": str(output_dir),
            "collage": str(collage_path),
            "metadata": str(metadata_path),
        }

        print()
        print("[Canonical] " "Views:")

        for view_name in VIEW_ORDER:
            print("[Canonical] " f"  {view_name}")

        print("[Canonical] " f"Collage: " f"{collage_path}")

        print()
        print("[Canonical] Done.")

        return result

    @staticmethod
    def _renders_exist(
        views_dir,
        metadata_path,
    ):
        if not (metadata_path.exists()):
            return False

        if not views_dir.exists():
            return False

        for view_name in VIEW_ORDER:
            image_path = views_dir / f"{view_name}.png"

            if not (image_path.exists()):
                return False

        return True

    @staticmethod
    def _render_views(
        input_path,
        output_dir,
    ):
        blender_path = find_blender()

        project_root = Path(__file__).resolve().parents[2]

        render_script = project_root / "rendering" / "render_canonical_views.py"

        if not (render_script.exists()):
            raise FileNotFoundError(
                "Canonical rendering " "script not found: " f"{render_script}"
            )

        print("[Canonical] " "Rendering 8 canonical views")

        result = subprocess.run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(render_script),
                "--",
                str(input_path),
                str(output_dir),
            ],
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError("Canonical rendering " "failed.")

    @staticmethod
    def _load_metadata(
        metadata_path,
    ):
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
