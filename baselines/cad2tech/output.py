import shutil
from pathlib import Path

from PIL import Image


def save_selected_views(
    selected_paths,
    output_dir,
):
    output_dir = Path(output_dir)
    selected_dir = output_dir / "selected"

    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths = []

    for i, source_path in enumerate(selected_paths):
        destination = selected_dir / f"view_{i:02d}.png"

        shutil.copy2(
            source_path,
            destination,
        )

        saved_paths.append(destination)

    return saved_paths


def create_collage(
    image_paths,
    output_path,
    columns=3,
):
    images = [Image.open(path).convert("RGB") for path in image_paths]

    if not images:
        raise ValueError("No images for collage.")

    width, height = images[0].size

    rows = (len(images) + columns - 1) // columns

    collage = Image.new(
        "RGB",
        (
            columns * width,
            rows * height,
        ),
        "white",
    )

    for i, image in enumerate(images):
        x = (i % columns) * width
        y = (i // columns) * height

        collage.paste(
            image,
            (x, y),
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collage.save(output_path)

    return output_path
