from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)


def resize_to_height(
    image,
    target_height,
):
    ratio = (
        target_height
        / image.height
    )

    target_width = int(
        image.width
        * ratio
    )

    return image.resize(
        (
            target_width,
            target_height,
        )
    )


def load_font(size):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size,
        )
    except OSError:
        return (
            ImageFont.load_default(size=size)
        )


def create_comparison(
    cad2tech_collage,
    iso_collage,
    vin_nbv_collage,
    output_path,
    canonical_collage=None,
    mvcnn12_collage=None,
    mvcnn6_collage=None,
    vlm_iterative_collage=None,
    gift_collage=None,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entries = [
        (
            "CAD2Tech",
            Path(
                cad2tech_collage
            ),
        ),
        (
            "ISO Orthographic",
            Path(
                iso_collage
            ),
        ),
    ]

    if vin_nbv_collage is not None:
        entries.append(("VIN-NBV Oracle", Path(vin_nbv_collage)))

    if canonical_collage is not None:
        entries.append(("Canonical (8 views)", Path(canonical_collage)))

    for label, path in (
        ("VLM Iterative", vlm_iterative_collage),
        ("MVCNN Views (12 views)", mvcnn12_collage),
        ("MVCNN Views (6 views)", mvcnn6_collage),
        ("GIFT-inspired Fibonacci (6 views)", gift_collage),
    ):
        if path is not None:
            entries.append((label, Path(path)))

    images = [
        Image.open(
            path
        ).convert(
            "RGB"
        )
        for _, path in entries
    ]

    target_height = max(
        image.height
        for image in images
    )

    images = [
        resize_to_height(
            image,
            target_height,
        )
        for image in images
    ]

    gap = 24
    margin = 24
    header_height = 100

    canvas_width = (
        sum(
            image.width
            for image in images
        )
        + gap
        * (
            len(images)
            - 1
        )
        + margin
        * 2
    )

    canvas_height = (
        target_height
        + header_height
        + margin
        * 2
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = load_font(
        36
    )

    x = margin

    for (
        label,
        _,
    ), image in zip(
        entries,
        images,
    ):
        center_x = (
            x
            + image.width
            // 2
        )

        draw.text(
            (
                center_x,
                margin
                + header_height
                // 2,
            ),
            label,
            fill="black",
            font=font,
            anchor="mm",
        )

        canvas.paste(
            image,
            (
                x,
                margin
                + header_height,
            ),
        )

        x += (
            image.width
            + gap
        )

    canvas.save(
        output_path
    )

    print(
        "[Comparison] Saved: "
        f"{output_path}"
    )

    return output_path
