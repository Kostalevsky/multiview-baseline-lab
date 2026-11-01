import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


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
    labels=None,
):
    images = [Image.open(path).convert("RGB") for path in image_paths]

    if not images:
        raise ValueError("No images for collage.")

    if labels is not None and len(labels) != len(images):
        raise ValueError("Each image must have exactly one label.")

    width, height = images[0].size

    rows = (len(images) + columns - 1) // columns
    label_height = 64 if labels is not None else 0

    collage = Image.new(
        "RGB",
        (
            columns * width,
            rows * (height + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(collage)
    font = load_font(24)

    for i, image in enumerate(images):
        x = (i % columns) * width
        y = (i // columns) * (height + label_height)

        collage.paste(
            image,
            (x, y),
        )

        if labels is not None:
            draw.text(
                (x + width // 2, y + height + label_height // 2),
                labels[i], fill="black", font=font, anchor="mm",
            )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collage.save(output_path)

    return output_path
