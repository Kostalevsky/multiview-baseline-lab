import json
import shutil
import subprocess
from pathlib import Path

from baselines.cad2tech.output import (
    create_collage,
)
from baselines.vlm_iterative.client import (
    QwenVLMClient,
)
from baselines.vlm_iterative.selector import (
    select_views_iteratively,
)
from utils.blender import find_blender


class VLMIterativeBaseline:
    name = "vlm_iterative"

    def __init__(
        self,
        client=None,
    ):
        if client is None:
            client = QwenVLMClient()

        self.client = client

    def run(
        self,
        input_path: str,
    ):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: " f"{input_path}")

        project_root = Path(__file__).resolve().parents[2]

        blender_path = find_blender()

        render_script = project_root / "rendering" / "render_vlm_candidates.py"

        model_name = input_path.stem

        output_dir = project_root / "outputs" / model_name / "vlm_iterative"

        candidates_dir = output_dir / "candidates"

        working_dir = output_dir / "iterations"

        selected_dir = output_dir / "selected"

        collage_path = output_dir / "collage.png"

        history_path = output_dir / "selection_history.json"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidates_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        working_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 50)
        print("VLM Iterative baseline")
        print("=" * 50)
        print(f"[VLM Iterative] Input: " f"{input_path}")
        print()

        print("[VLM Iterative] " "Step 1/3: " "rendering candidate views")

        result = subprocess.run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(render_script),
                "--",
                str(input_path),
                str(candidates_dir),
            ],
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError("VLM candidate " "rendering failed.")

        candidate_paths = sorted(candidates_dir.glob("view_*.png"))

        if len(candidate_paths) != 32:
            raise RuntimeError(
                "Expected 32 VLM "
                "candidate views, "
                f"got "
                f"{len(candidate_paths)}."
            )

        print()
        print("[VLM Iterative] " "Step 2/3: " "iterative selection")

        (
            selected_indices,
            history,
            candidate_map,
            refinement_history,
        ) = select_views_iteratively(
            client=self.client,
            candidates_dir=candidates_dir,
            working_dir=working_dir,
            num_views=6,
        )

        print("[VLM Iterative] " "Step 3/3: " "saving selected views")

        selected_paths = []

        for output_index, (candidate_index) in enumerate(selected_indices):
            source_path = Path(candidate_map[candidate_index]["image"])

            destination_path = selected_dir / (f"view_" f"{output_index:02d}.png")

            shutil.copy2(
                source_path,
                destination_path,
            )

            selected_paths.append(destination_path)

        create_collage(
            selected_paths,
            collage_path,
            columns=3,
        )

        history_data = {
            "baseline": self.name,
            "input": str(input_path),
            "selection_mode": "iterative_vlm",
            "num_candidates": 32,
            "num_selected": 6,
            "refinement": refinement_history,
            "candidate_space": {
                "azimuths": [
                    0,
                    45,
                    90,
                    135,
                    180,
                    225,
                    270,
                    315,
                ],
                "elevations": [
                    -35,
                    15,
                    45,
                    70,
                ],
            },
            "selected_indices": (selected_indices),
            "selected_views": [
                {
                    "index": index,
                    "azimuth": candidate_map[index]["azimuth"],
                    "elevation": candidate_map[index]["elevation"],
                }
                for index in selected_indices
            ],
            "iterations": history,
        }

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history_data,
                file,
                indent=2,
            )

        print()
        print("[VLM Iterative] " f"Selected indices: " f"{selected_indices}")

        print("[VLM Iterative] " f"Collage: " f"{collage_path}")

        print("[VLM Iterative] " f"History: " f"{history_path}")

        print()
        print("[VLM Iterative] Done.")

        return {
            "baseline": (self.name),
            "input": str(input_path),
            "num_candidates": 32,
            "num_selected": 6,
            "selected_indices": (selected_indices),
            "output_dir": str(output_dir),
            "collage": str(collage_path),
            "history": str(history_path),
        }
