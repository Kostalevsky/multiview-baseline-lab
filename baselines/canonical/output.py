from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

VIEW_LABELS = {
    "pos_x": "+X",
    "neg_x": "-X",
    "pos_y": "+Y",
    "neg_y": "-Y",
    "pos_z": "+Z",
    "neg_z": "-Z",
    "iso_1": "Iso 1",
    "iso_2": "Iso 2",
}


VIEW_ORDER = [
    "pos_x",
    "neg_x",
    "pos_y",
    "neg_y",
    "pos_z",
    "neg_z",
    "iso_1",
    "iso_2",
]


def load_font(size):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default(size=size)


def create_canonical_collage(
    views_dir,
    output_path,
):
    views_dir = Path(views_dir)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = 4
    rows = 2

    tile_size = 512
    label_height = 50

    canvas = Image.new(
        "RGB",
        (
            columns * tile_size,
            rows * (tile_size + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    font = load_font(28)

    for position, view_name in enumerate(VIEW_ORDER):
        image_path = views_dir / f"{view_name}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Canonical view not found: " f"{image_path}")

        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize(
                (
                    tile_size,
                    tile_size,
                )
            )
        )

        row = position // columns

        column = position % columns

        x = column * tile_size

        y = row * (tile_size + label_height)

        canvas.paste(
            image,
            (
                x,
                y + label_height,
            ),
        )

        draw.text(
            (
                x + tile_size // 2,
                y + label_height // 2,
            ),
            f"#{position + 1} | {VIEW_LABELS[view_name]}",
            fill="black",
            font=font,
            anchor="mm",
        )

    canvas.save(output_path)

    return output_path
