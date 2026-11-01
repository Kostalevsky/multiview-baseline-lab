import argparse
from pathlib import Path

from baselines.cad2tech.baseline import CAD2TechBaseline
from baselines.iso.baseline import ISOBaseline
from baselines.vin_nbv.baseline import VINNBVBaseline
from baselines.vlm_iterative.baseline import VLMIterativeBaseline
from baselines.canonical.baseline import CanonicalBaseline
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
            "all",
        ],
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    args = parser.parse_args()

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

        model_name = (
            input_path.stem
        )

        comparison_path = (
            project_root
            / "outputs"
            / model_name
            / "compare.png"
        )

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
        )

        result = {
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