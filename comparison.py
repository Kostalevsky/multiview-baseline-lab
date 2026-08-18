from pathlib import Path

from PIL import Image, ImageDraw


def create_comparison(
    cad2tech_collage,
    iso_collage,
    output_path,
):
    cad2tech_collage = Path(cad2tech_collage)
    iso_collage = Path(iso_collage)
    output_path = Path(output_path)

    cad_image = Image.open(cad2tech_collage).convert("RGB")
    iso_image = Image.open(iso_collage).convert("RGB")

    target_height = max(
        cad_image.height,
        iso_image.height,
    )

    def resize_to_height(image, height):
        ratio = height / image.height

        new_width = int(image.width * ratio)

        return image.resize((new_width, height))

    cad_image = resize_to_height(
        cad_image,
        target_height,
    )

    iso_image = resize_to_height(
        iso_image,
        target_height,
    )

    header_height = 80
    gap = 20

    total_width = cad_image.width + iso_image.width + gap

    total_height = target_height + header_height

    comparison = Image.new(
        "RGB",
        (
            total_width,
            total_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(comparison)

    draw.text(
        (
            cad_image.width // 2,
            30,
        ),
        "CAD2Tech",
        fill="black",
        anchor="mm",
    )

    draw.text(
        (
            cad_image.width + gap + iso_image.width // 2,
            30,
        ),
        "ISO",
        fill="black",
        anchor="mm",
    )

    comparison.paste(
        cad_image,
        (0, header_height),
    )

    comparison.paste(
        iso_image,
        (
            cad_image.width + gap,
            header_height,
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.save(output_path)

    return output_path
