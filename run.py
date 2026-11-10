import argparse
from pathlib import Path
from subprocess import CalledProcessError

from baselines.cad2tech.baseline import CAD2TechBaseline
from baselines.canonical.baseline import CanonicalBaseline
from baselines.gift.baseline import GIFTFibonacciBaseline
from baselines.iso.baseline import ISOBaseline
from baselines.mvcnn.baseline import MVCNNViewsBaseline
from baselines.mvtn.baseline import MVTNBaseline
from baselines.vin_nbv.baseline import VINNBVBaseline
from baselines.vlm_iterative.baseline import VLMIterativeBaseline
from comparison import create_comparison


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        choices=[
            "cad2tech",
            "iso",
            "canonical",
            "vlm_iterative",
            "vin_nbv",
            "mvcnn12",
            "mvcnn6",
            "gift",
            "mvtn",
            "all",
        ],
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument("--checkpoint", help="MVTN checkpoint path")
    parser.add_argument("--checkpoint-sha256", help="Expected MVTN checkpoint hash")
    parser.add_argument(
        "--input-up-axis", choices=["y", "z"], help="MVTN input vertical axis"
    )
    parser.add_argument(
        "--output-dir",
        help="New output directory for MVTN, MVCNN views, or GIFT-inspired views",
    )
    parser.add_argument("--mvtn-image-size", type=int, default=512)
    parser.add_argument("--mvtn-samples", type=int, default=32)

    args = parser.parse_args()

    if args.baseline != "mvtn" and (
        any(
            value is not None
            for value in (
                args.checkpoint,
                args.checkpoint_sha256,
                args.input_up_axis,
            )
        )
        or args.mvtn_image_size != 512
        or args.mvtn_samples != 32
    ):
        parser.error("MVTN options require --baseline mvtn")

    if args.output_dir is not None and args.baseline not in (
        "mvtn", "mvcnn12", "mvcnn6", "gift"
    ):
        parser.error("--output-dir requires mvtn, mvcnn12, mvcnn6, or gift")

    if args.baseline in ("mvcnn12", "mvcnn6", "gift"):
        baseline = (
            GIFTFibonacciBaseline()
            if args.baseline == "gift"
            else MVCNNViewsBaseline(args.baseline)
        )
        try:
            print(baseline.run(args.input, output_dir=args.output_dir))
        except (OSError, CalledProcessError) as exc:
            parser.exit(1, f"{args.baseline} views failed: {exc}\n")
        return

    if args.baseline == "mvtn":
        if args.checkpoint is None or args.input_up_axis is None:
            parser.error("mvtn requires --checkpoint and --input-up-axis")
        baseline = MVTNBaseline(
            args.checkpoint,
            input_up_axis=args.input_up_axis,
            expected_sha256=args.checkpoint_sha256,
            image_size=args.mvtn_image_size,
            samples=args.mvtn_samples,
        )
        try:
            print(baseline.run(args.input, output_dir=args.output_dir))
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"MVTN failed: {exc}\n")
        return

    if args.baseline == "cad2tech":
        baseline = CAD2TechBaseline()

        result = baseline.run(
            args.input,
        )

        print(result)

    elif args.baseline == "iso":
        baseline = ISOBaseline()

        result = baseline.run(
            args.input,
        )

        print(result)
    
    elif args.baseline == "canonical":
        baseline = (
            CanonicalBaseline()
        )

        result = baseline.run(
            args.input,
        )

        print(result)

    elif args.baseline == "vlm_iterative":
        baseline = VLMIterativeBaseline()

        result = baseline.run(
            args.input,
        )

        print(result)

    elif args.baseline == "vin_nbv":
        baseline = VINNBVBaseline()

        result = baseline.run(
            args.input,
        )

        print(result)

    elif args.baseline == "all":
        input_path = Path(
            args.input
        ).resolve()

        project_root = (
            Path(__file__)
            .resolve()
            .parent
        )

        print()
        print("=" * 60)
        print("Running all baselines")
        print("=" * 60)
        print()

        cad2tech = CAD2TechBaseline()
        iso = ISOBaseline()
        vin_nbv = VINNBVBaseline()
        canonical = CanonicalBaseline()

        cad2tech_result = (
            cad2tech.run(
                input_path
            )
        )

        iso_result = (
            iso.run(
                input_path
            )
        )

        vin_nbv_result = (
            vin_nbv.run(
                input_path
            )
        )

        canonical_result = canonical.run(input_path)

        model_name = (
            input_path.stem
        )

        comparison_path = (
            project_root
            / "outputs"
            / model_name
            / "compare.png"
        )

        extra_collages = {}
        for name in ("mvcnn12", "mvcnn6"):
            path = project_root / "outputs" / model_name / name / "collage.png"
            if path.is_file():
                extra_collages[f"{name}_collage"] = path
        gift_collage = (
            project_root / "outputs" / model_name / "gift_fibonacci6" / "collage.png"
        )
        if gift_collage.is_file():
            extra_collages["gift_collage"] = gift_collage

        create_comparison(
            cad2tech_result[
                "collage"
            ],
            iso_result[
                "collage"
            ],
            vin_nbv_result[
                "collage"
            ],
            comparison_path,
            canonical_collage=canonical_result["collage"],
            **extra_collages,
        )

        result = {
            "canonical": canonical_result,
            "cad2tech": (
                cad2tech_result
            ),
            "iso": (
                iso_result
            ),
            "vin_nbv": (
                vin_nbv_result
            ),
            "comparison": str(
                comparison_path
            ),
        }

        print()
        print("=" * 60)
        print(
            "All baselines completed"
        )
        print("=" * 60)
        print()

        print(result)


if __name__ == "__main__":
    main()
