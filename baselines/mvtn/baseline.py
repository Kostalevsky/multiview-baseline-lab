from pathlib import Path

from .predict import predict_views
from .render import render_views


class MVTNBaseline:
    name = "mvtn"

    def __init__(
        self,
        checkpoint_path,
        *,
        input_up_axis,
        num_points=2048,
        seed=42,
        image_size=512,
        samples=32,
        expected_sha256=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.input_up_axis = input_up_axis
        self.num_points = num_points
        self.seed = seed
        self.image_size = image_size
        self.samples = samples
        self.expected_sha256 = expected_sha256

    def run(self, input_path, output_dir=None):
        input_path = Path(input_path).resolve(strict=True)
        if output_dir is None:
            output_dir = (
                Path(__file__).resolve().parents[2]
                / "outputs"
                / input_path.stem
                / "mvtn"
            )
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(f"Choose a new output directory: {output_dir}")
        cameras_path = output_dir / "prediction" / "cameras.json"
        predict_views(
            input_path,
            self.checkpoint_path,
            cameras_path.parent,
            input_up_axis=self.input_up_axis,
            num_points=self.num_points,
            seed=self.seed,
            expected_sha256=self.expected_sha256,
        )
        result = render_views(
            cameras_path,
            output_dir / "render",
            image_size=self.image_size,
            samples=self.samples,
        )
        return {
            **result,
            "output_dir": str(output_dir),
            "camera_metadata": str(cameras_path),
            "metadata": str(output_dir / "render" / "render_metadata.json"),
        }
