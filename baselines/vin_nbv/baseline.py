import json
import subprocess
from pathlib import Path

from baselines.cad2tech.output import (
    create_collage,
    save_selected_views,
)
from baselines.vin_nbv.selector import select_views
from utils.blender import find_blender


class VINNBVBaseline:
    def __init__(
        self,
        num_views=6,
        num_gt_points=100_000,
        visibility_resolution=192,
        seed=42,
        coverage_threshold=0.01,
    ):
        self.num_views = num_views
        self.num_gt_points = num_gt_points
        self.visibility_resolution = visibility_resolution
        self.seed = seed
        self.coverage_threshold = coverage_threshold

    def run(
        self,
        input_path,
        output_dir=None,
    ):
        input_path = Path(input_path).resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: " f"{input_path}")

        if output_dir is None:
            output_dir = Path("outputs") / input_path.stem / "vin_nbv"
        else:
            output_dir = Path(output_dir)

        output_dir = output_dir.resolve()

        candidates_dir = output_dir / "candidates"

        geometry_dir = output_dir / "geometry"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("[VIN-NBV] Starting baseline")

        print("[VIN-NBV] " f"Input: {input_path}")

        print("[VIN-NBV] " f"Output: {output_dir}")

        print()

        self._render_candidates(
            input_path=(input_path),
            candidates_dir=(candidates_dir),
        )

        self._precompute_geometry(
            input_path=(input_path),
            candidates_dir=(candidates_dir),
            geometry_dir=(geometry_dir),
        )

        print("[VIN-NBV] " "Step 3/4: " "selecting next-best views")

        selection_result = select_views(
            candidate_metadata_path=(candidates_dir / "metadata.json"),
            geometry_dir=(geometry_dir),
            num_views=(self.num_views),
            seed=(self.seed),
            coverage_threshold=(self.coverage_threshold),
        )

        selected_indices = selection_result["selected_indices"]

        print("[VIN-NBV] " "Step 4/4: " "saving selected views")

        selected_source_paths = [
            candidates_dir / "rgb" / f"view_{index:03d}.png"
            for index in selected_indices
        ]

        for source_path in selected_source_paths:
            if not source_path.exists():
                raise FileNotFoundError(
                    "Selected RGB view " "not found: " f"{source_path}"
                )

        selected_paths = save_selected_views(
            selected_paths=(selected_source_paths),
            output_dir=(output_dir),
        )

        collage_path = output_dir / "collage.png"

        create_collage(
            image_paths=(selected_paths),
            output_path=(collage_path),
            columns=3,
        )

        history_path = output_dir / "selection_history.json"

        selected_metadata = []

        candidate_metadata = self._load_candidate_metadata(
            candidates_dir / "metadata.json"
        )

        candidate_map = {
            int(view["index"]): view for view in candidate_metadata["views"]
        }

        for position, (
            index,
            saved_path,
        ) in enumerate(
            zip(
                selected_indices,
                selected_paths,
            ),
            start=1,
        ):
            candidate = candidate_map[index]

            selected_metadata.append(
                {
                    "position": (position),
                    "index": (index),
                    "saved_rgb": str(saved_path),
                    "candidate_rgb": str(
                        candidates_dir / "rgb" / (f"view_" f"{index:03d}.png")
                    ),
                    "candidate_depth": str(
                        candidates_dir / "depth" / (f"view_" f"{index:03d}.npy")
                    ),
                    "shell": (candidate.get("shell")),
                    "radius": (candidate.get("radius")),
                    "azimuth": (candidate.get("azimuth")),
                    "elevation": (candidate.get("elevation")),
                    "camera_position": (candidate.get("camera_position")),
                }
            )

        history_payload = {
            **selection_result,
            "input": str(input_path),
            "output_dir": str(output_dir),
            "candidate_metadata": str(candidates_dir / "metadata.json"),
            "geometry_metadata": str(geometry_dir / "geometry_metadata.json"),
            "collage": str(collage_path),
            "selected": (selected_metadata),
        }

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history_payload,
                file,
                indent=2,
            )

        print()
        print("[VIN-NBV] " f"Selected indices: " f"{selected_indices}")

        print(
            "[VIN-NBV] "
            "Final coverage: "
            f"{selection_result['final_surface_coverage']:.4f}"
        )

        print(
            "[VIN-NBV] "
            "Final coverage distance: "
            f"{selection_result['final_coverage_distance']:.6f}"
        )

        print("[VIN-NBV] " f"Collage: " f"{collage_path}")

        print("[VIN-NBV] " f"History: " f"{history_path}")

        print()
        print("[VIN-NBV] Done.")

        return {
            "baseline": ("vin_nbv"),
            "input": str(input_path),
            "num_candidates": (selection_result["num_candidates"]),
            "num_selected": (selection_result["num_selected"]),
            "selected_indices": (selected_indices),
            "final_surface_coverage": (selection_result["final_surface_coverage"]),
            "final_coverage_distance": (selection_result["final_coverage_distance"]),
            "output_dir": str(output_dir),
            "collage": str(collage_path),
            "history": str(history_path),
        }

    def _render_candidates(
        self,
        input_path,
        candidates_dir,
    ):
        metadata_path = candidates_dir / "metadata.json"

        rgb_dir = candidates_dir / "rgb"

        depth_dir = candidates_dir / "depth"

        rgb_files = list(rgb_dir.glob("view_*.png")) if rgb_dir.exists() else []

        depth_files = list(depth_dir.glob("view_*.npy")) if depth_dir.exists() else []

        if metadata_path.exists() and len(rgb_files) == 120 and len(depth_files) == 120:
            print(
                "[VIN-NBV] "
                "Step 1/4: "
                "candidate renders "
                "already exist — skipping"
            )

            return

        print("[VIN-NBV] " "Step 1/4: " "rendering candidate " "RGB-D views")

        blender_path = find_blender()

        project_root = Path(__file__).resolve().parents[2]

        render_script = project_root / "rendering" / ("render_" "vin_nbv_candidates.py")

        if not render_script.exists():
            raise FileNotFoundError(
                "VIN-NBV rendering script " "not found: " f"{render_script}"
            )

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
            raise RuntimeError("VIN-NBV candidate " "rendering failed.")

        if not (candidates_dir / "metadata.json").exists():
            raise RuntimeError(
                "VIN-NBV rendering "
                "completed but "
                "metadata.json "
                "was not created."
            )

    def _precompute_geometry(
        self,
        input_path,
        candidates_dir,
        geometry_dir,
    ):
        geometry_metadata = geometry_dir / "geometry_metadata.json"

        ground_truth_path = geometry_dir / "ground_truth.npy"

        visible_dir = geometry_dir / "visible"

        visible_files = (
            list(visible_dir.glob("view_*.npy")) if visible_dir.exists() else []
        )

        if (
            geometry_metadata.exists()
            and ground_truth_path.exists()
            and len(visible_files) == 120
        ):
            print(
                "[VIN-NBV] " "Step 2/4: " "geometry cache " "already exists — skipping"
            )

            return

        print("[VIN-NBV] " "Step 2/4: " "precomputing geometry")

        blender_path = find_blender()

        project_root = Path(__file__).resolve().parents[2]

        geometry_script = (
            project_root / "rendering" / ("precompute_" "vin_nbv_geometry.py")
        )

        if not geometry_script.exists():
            raise FileNotFoundError(
                "VIN-NBV geometry " "script not found: " f"{geometry_script}"
            )

        candidate_metadata_path = candidates_dir / "metadata.json"

        if not (candidate_metadata_path.exists()):
            raise FileNotFoundError(
                "VIN-NBV candidate " "metadata not found: " f"{candidate_metadata_path}"
            )

        result = subprocess.run(
            [
                str(blender_path),
                "--background",
                "--python",
                str(geometry_script),
                "--",
                str(input_path),
                str(candidate_metadata_path),
                str(geometry_dir),
                str(self.num_gt_points),
                str(self.visibility_resolution),
            ],
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError("VIN-NBV geometry " "precomputation failed.")

        if not (ground_truth_path.exists()):
            raise RuntimeError(
                "VIN-NBV geometry "
                "precomputation completed "
                "but ground_truth.npy "
                "was not created."
            )

        if not (geometry_metadata.exists()):
            raise RuntimeError(
                "VIN-NBV geometry "
                "precomputation completed "
                "but geometry_metadata.json "
                "was not created."
            )

    @staticmethod
    def _load_candidate_metadata(
        metadata_path,
    ):
        metadata_path = Path(metadata_path)

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
