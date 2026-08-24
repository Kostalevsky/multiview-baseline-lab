import subprocess
from pathlib import Path
from baselines.cad2tech.output import (
    create_collage,
    save_selected_views,
)
from baselines.cad2tech.selector import select_views
from utils.blender import find_blender


class CAD2TechBaseline:
    name = "cad2tech"

    def run(self, input_path: str):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")

        project_root = Path(__file__).resolve().parents[2]

        blender_path = find_blender()

        render_script = project_root / "rendering" / "render_candidates.py"

        model_name = input_path.stem

        output_dir = project_root / "outputs" / model_name / "cad2tech"

        candidates_dir = output_dir / "candidates"
        collage_path = output_dir / "collage.png"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidates_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 50)
        print("CAD2Tech baseline")
        print("=" * 50)

        print(f"[CAD2Tech] Input model: {input_path}")
        print(f"[CAD2Tech] Output dir: {output_dir}")

        print()
        print("[CAD2Tech] Step 1/3: rendering candidates")

        result = subprocess.run(
            [
                "--background",
                "--python",
                str(render_script),
                "--",
                str(input_path),
                str(candidates_dir),
                str(blender_path),
            ],
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError("CAD2Tech candidate rendering failed.")

        candidate_paths = sorted(candidates_dir.glob("view_*.png"))

        if len(candidate_paths) != 28:
            raise RuntimeError(
                f"Expected 28 candidates, " f"got {len(candidate_paths)}."
            )

        print(f"[CAD2Tech] Generated " f"{len(candidate_paths)} candidates")

        print()
        print("[CAD2Tech] Step 2/3: selecting views")

        (
            selected_indices,
            selected_paths,
            embedding,
        ) = select_views(
            candidates_dir,
            num_views=6,
        )

        print(
            "[CAD2Tech] Selected candidate indices:",
            selected_indices,
        )

        print()
        print("[CAD2Tech] Step 3/3: saving output")

        saved_paths = save_selected_views(
            selected_paths,
            output_dir,
        )

        create_collage(
            saved_paths,
            collage_path,
        )

        print(f"[CAD2Tech] Selected views saved to: " f"{output_dir / 'selected'}")

        print(f"[CAD2Tech] Collage saved to: " f"{collage_path}")

        print()
        print("[CAD2Tech] Done.")

        return {
            "baseline": self.name,
            "input": str(input_path),
            "num_candidates": len(candidate_paths),
            "num_selected": len(saved_paths),
            "selected_indices": selected_indices,
            "output_dir": str(output_dir),
            "collage": str(collage_path),
        }
