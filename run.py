import argparse
from pathlib import Path

from baselines.cad2tech.baseline import CAD2TechBaseline
from baselines.iso.baseline import ISOBaseline
from baselines.iso_isometric.baseline import ISOIsometricBaseline
from comparison import create_comparison


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        required=True,
        choices=[
            "cad2tech",
            "iso",
            "iso_isometric",
            "all",
        ],
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.baseline == "cad2tech":
        baseline = CAD2TechBaseline()

        result = baseline.run(
            args.input
        )

    elif args.baseline == "iso":
        baseline = ISOBaseline()

        result = baseline.run(
            args.input
        )

    elif args.baseline == "iso_isometric":
        baseline = ISOIsometricBaseline()

        result = baseline.run(
            args.input
        )

    elif args.baseline == "all":
        cad2tech = CAD2TechBaseline()
        iso = ISOBaseline()
        iso_isometric = ISOIsometricBaseline()

        print(
            "\n=== CAD2Tech ===\n"
        )

        cad2tech_result = cad2tech.run(
            args.input
        )

        print(
            "\n=== ISO Orthographic ===\n"
        )

        iso_result = iso.run(
            args.input
        )

        print(
            "\n=== ISO Isometric ===\n"
        )

        iso_isometric_result = (
            iso_isometric.run(
                args.input
            )
        )

        input_path = Path(
            args.input
        ).resolve()

        project_root = Path(
            __file__
        ).resolve().parent

        model_name = input_path.stem

        comparison_path = (
            project_root
            / "outputs"
            / model_name
            / "compare.png"
        )

        create_comparison(
            cad2tech_result["collage"],
            iso_result["collage"],
            iso_isometric_result["collage"],
            comparison_path,
        )

        result = {
            "cad2tech": cad2tech_result,
            "iso": iso_result,
            "iso_isometric": iso_isometric_result,
            "comparison": str(
                comparison_path
            ),
        }

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()