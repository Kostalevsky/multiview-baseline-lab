import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from baselines.vlm_iterative.prompts import (
    build_refinement_prompt,
    build_selection_prompt,
)


def load_font(size):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default()


def create_labeled_collage(
    entries,
    output_path,
    columns=6,
    tile_size=256,
):
    if not entries:
        return None

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_height = 32

    rows = (
        len(entries)
        + columns
        - 1
    ) // columns

    canvas = Image.new(
        "RGB",
        (
            columns * tile_size,
            rows * (
                tile_size
                + label_height
            ),
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = load_font(
        20
    )

    for position, (
        index,
        path,
    ) in enumerate(entries):
        image = (
            Image.open(path)
            .convert("RGB")
            .resize(
                (
                    tile_size,
                    tile_size,
                )
            )
        )

        row = (
            position
            // columns
        )

        column = (
            position
            % columns
        )

        x = (
            column
            * tile_size
        )

        y = (
            row
            * (
                tile_size
                + label_height
            )
        )

        canvas.paste(
            image,
            (
                x,
                y,
            ),
        )

        draw.rectangle(
            (
                x,
                y + tile_size,
                x + tile_size,
                y + tile_size + label_height,
            ),
            fill="white",
        )

        draw.text(
            (
                x + tile_size // 2,
                y
                + tile_size
                + label_height
                // 2,
            ),
            f"View {index:02d}",
            fill="black",
            font=font,
            anchor="mm",
        )

    canvas.save(
        output_path
    )

    return output_path


def load_metadata(
    metadata_path,
):
    metadata_path = Path(
        metadata_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def select_views_iteratively(
    client,
    candidates_dir,
    working_dir,
    num_views=6,
):
    candidates_dir = Path(
        candidates_dir
    )

    working_dir = Path(
        working_dir
    )

    working_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = load_metadata(
        candidates_dir
        / "metadata.json"
    )

    candidate_map = {
        item["index"]: item
        for item in metadata
    }

    selected_indices = []

    history = []

    for step in range(
        num_views
    ):
        remaining_indices = [
            index
            for index in candidate_map
            if index
            not in selected_indices
        ]

        remaining_entries = [
            (
                index,
                candidate_map[
                    index
                ][
                    "image"
                ],
            )
            for index
            in remaining_indices
        ]

        candidate_collage = (
            working_dir
            / (
                f"candidates_step_"
                f"{step + 1:02d}.png"
            )
        )

        create_labeled_collage(
            remaining_entries,
            candidate_collage,
            columns=6,
        )

        selected_collage = None

        if selected_indices:
            selected_entries = [
                (
                    index,
                    candidate_map[
                        index
                    ][
                        "image"
                    ],
                )
                for index
                in selected_indices
            ]

            selected_collage = (
                working_dir
                / (
                    f"selected_step_"
                    f"{step + 1:02d}.png"
                )
            )

            create_labeled_collage(
                selected_entries,
                selected_collage,
                columns=3,
            )

        prompt = (
            build_selection_prompt(
                step=step,
                num_views=num_views,
                remaining_indices=(
                    remaining_indices
                ),
                selected_indices=(
                    selected_indices
                ),
                candidate_map=(
                    candidate_map
                ),
            )
        )

        response = (
            client.select_next_view(
                candidate_collage=(
                    candidate_collage
                ),
                selected_collage=(
                    selected_collage
                ),
                prompt=prompt,
                remaining_indices=(
                    remaining_indices
                ),
            )
        )

        selected_index = (
            response.get(
                "selected_index"
            )
        )

        if selected_index not in (
            remaining_indices
        ):
            raise RuntimeError(
                "VLM returned invalid "
                f"candidate index: "
                f"{selected_index}"
            )

        selected_before = list(
            selected_indices
        )

        selected_indices.append(
            selected_index
        )

        candidate = (
            candidate_map[
                selected_index
            ]
        )

        history.append(
            {
                "step": (
                    step
                    + 1
                ),
                "selected_index": (
                    selected_index
                ),
                "azimuth": (
                    candidate[
                        "azimuth"
                    ]
                ),
                "elevation": (
                    candidate[
                        "elevation"
                    ]
                ),
                "camera_position": (
                    candidate.get(
                        "camera_position"
                    )
                ),
                "selected_before": (
                    selected_before
                ),
                "reason": (
                    response.get(
                        "reason",
                        "",
                    )
                ),
                "covered_geometry": (
                    response.get(
                        "covered_geometry",
                        [],
                    )
                ),
                "missing_geometry": (
                    response.get(
                        "missing_geometry",
                        [],
                    )
                ),
            }
        )

        print(
            "[VLM Iterative] "
            f"Step {step + 1}/{num_views}: "
            f"selected view "
            f"{selected_index:02d}"
        )

        print(
            "[VLM Iterative] "
            f"Camera: "
            f"azimuth="
            f"{candidate['azimuth']}°, "
            f"elevation="
            f"{candidate['elevation']}°"
        )

        print(
            "[VLM Iterative] "
            f"Reason: "
            f"{response.get('reason', '')}"
        )

        print()

    remaining_indices = [
        index
        for index in candidate_map
        if index
        not in selected_indices
    ]

    selected_entries = [
        (
            index,
            candidate_map[
                index
            ][
                "image"
            ],
        )
        for index
        in selected_indices
    ]

    remaining_entries = [
        (
            index,
            candidate_map[
                index
            ][
                "image"
            ],
        )
        for index
        in remaining_indices
    ]

    final_selected_collage = (
        working_dir
        / "final_selected_before_refinement.png"
    )

    final_remaining_collage = (
        working_dir
        / "final_remaining_candidates.png"
    )

    create_labeled_collage(
        selected_entries,
        final_selected_collage,
        columns=3,
    )

    create_labeled_collage(
        remaining_entries,
        final_remaining_collage,
        columns=6,
    )

    refinement_prompt = (
        build_refinement_prompt(
            selected_indices=(
                selected_indices
            ),
            remaining_indices=(
                remaining_indices
            ),
            candidate_map=(
                candidate_map
            ),
        )
    )

    print(
        "[VLM Iterative] "
        "Reviewing final selected set"
    )

    refinement = (
        client.review_final_set(
            selected_collage=(
                final_selected_collage
            ),
            candidate_collage=(
                final_remaining_collage
            ),
            prompt=(
                refinement_prompt
            ),
            selected_indices=(
                selected_indices
            ),
            remaining_indices=(
                remaining_indices
            ),
        )
    )

    refinement_history = {
        "replace": (
            refinement.get(
                "replace",
                False,
            )
        ),
        "reason": (
            refinement.get(
                "reason",
                "",
            )
        ),
    }

    if refinement.get(
        "replace",
        False,
    ):
        remove_index = (
            refinement[
                "remove_index"
            ]
        )

        add_index = (
            refinement[
                "add_index"
            ]
        )

        if remove_index not in (
            selected_indices
        ):
            raise RuntimeError(
                "VLM refinement returned "
                "invalid remove_index: "
                f"{remove_index}"
            )

        if add_index not in (
            remaining_indices
        ):
            raise RuntimeError(
                "VLM refinement returned "
                "invalid add_index: "
                f"{add_index}"
            )

        position = (
            selected_indices.index(
                remove_index
            )
        )

        selected_indices[
            position
        ] = add_index

        refinement_history[
            "remove_index"
        ] = remove_index

        refinement_history[
            "add_index"
        ] = add_index

        refinement_history[
            "removed_view"
        ] = {
            "index": (
                remove_index
            ),
            "azimuth": (
                candidate_map[
                    remove_index
                ][
                    "azimuth"
                ]
            ),
            "elevation": (
                candidate_map[
                    remove_index
                ][
                    "elevation"
                ]
            ),
        }

        refinement_history[
            "added_view"
        ] = {
            "index": (
                add_index
            ),
            "azimuth": (
                candidate_map[
                    add_index
                ][
                    "azimuth"
                ]
            ),
            "elevation": (
                candidate_map[
                    add_index
                ][
                    "elevation"
                ]
            ),
        }

        print(
            "[VLM Iterative] "
            "Final refinement:"
        )

        print(
            "[VLM Iterative] "
            f"Removed view "
            f"{remove_index:02d}"
        )

        print(
            "[VLM Iterative] "
            f"Added view "
            f"{add_index:02d}"
        )

        print(
            "[VLM Iterative] "
            f"Reason: "
            f"{refinement.get('reason', '')}"
        )

    else:
        print(
            "[VLM Iterative] "
            "Final refinement: "
            "no replacement"
        )

        print(
            "[VLM Iterative] "
            f"Reason: "
            f"{refinement.get('reason', '')}"
        )

    print()

    return (
        selected_indices,
        history,
        candidate_map,
        refinement_history,
    )