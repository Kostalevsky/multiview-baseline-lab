def format_view_metadata(
    indices,
    candidate_map,
):
    lines = []

    for index in indices:
        candidate = candidate_map[index]

        lines.append(
            (
                f"View {index:02d}: "
                f"azimuth={candidate['azimuth']}°, "
                f"elevation={candidate['elevation']}°"
            )
        )

    return "\n".join(lines)


def build_selection_prompt(
    step,
    num_views,
    remaining_indices,
    selected_indices,
    candidate_map,
):
    remaining_metadata = format_view_metadata(
        remaining_indices,
        candidate_map,
    )

    if selected_indices:
        selected_metadata = format_view_metadata(
            selected_indices,
            candidate_map,
        )

        context = f"""
Already selected viewpoints:

{selected_metadata}
"""
    else:
        context = """
No viewpoints have been selected yet.
"""

    return f"""
You are selecting camera viewpoints for a known 3D object.

The goal is to build a compact set of exactly {num_views} views that together
represent the object's 3D shape as clearly and completely as possible.

You are selecting view {step + 1} of {num_views}.

{context}

AVAILABLE CANDIDATE VIEWPOINTS

{remaining_metadata}

Each candidate image is labeled with its View index.

Use both:
- the rendered appearance;
- the provided azimuth and elevation metadata.

The metadata is authoritative.

Before selecting the next viewpoint, reason about the CURRENT STATE of the
multi-view representation.

Identify:

1. COVERED GEOMETRY
   Which important geometric regions, sides, structures, or features are
   already well represented by the selected views?

2. MISSING GEOMETRY
   Which important aspects of the object's 3D shape are still missing,
   ambiguous, poorly represented, or only weakly visible?

Then choose the single candidate that best resolves the most important missing
geometric information.

Prefer views that:
- reveal previously unseen surfaces or structures;
- clarify holes, cavities, protrusions, thickness, or depth;
- expose important underside, upper, side, front, or rear geometry when useful;
- complement the selected set;
- remain visually informative.

Avoid views that:
- mostly duplicate geometry already selected;
- differ only slightly in camera direction;
- show mostly smooth or featureless surfaces without adding useful information;
- collapse the object into an uninformative thin silhouette.

Do not maximize angular diversity by itself.

Think about the FINAL SET of {num_views} images.

Return valid JSON only:

{{
    "covered_geometry": [
        "brief item"
    ],
    "missing_geometry": [
        "brief item"
    ],
    "selected_index": 12,
    "reason": "Briefly explain why this candidate best fills the most important missing geometric information."
}}

The selected_index MUST belong to:

{remaining_indices}
"""


def build_refinement_prompt(
    selected_indices,
    remaining_indices,
    candidate_map,
):
    selected_metadata = format_view_metadata(
        selected_indices,
        candidate_map,
    )

    remaining_metadata = format_view_metadata(
        remaining_indices,
        candidate_map,
    )

    return f"""
You are reviewing a final set of camera viewpoints for a known 3D object.

The goal is to represent the object's 3D geometry as completely, clearly, and
non-redundantly as possible using exactly {len(selected_indices)} views.

CURRENT SELECTED SET

{selected_metadata}

REMAINING CANDIDATES

{remaining_metadata}

You will see:
1. a collage of the current selected views;
2. a collage of the remaining candidate views.

Evaluate the selected set as a whole.

Look for:
- duplicated or weakly useful selected views;
- important geometry that is still missing;
- over-representation of one side or region;
- under-representation of another important side or feature;
- views dominated by smooth or featureless geometry;
- missed underside, upper, side, front, or rear structure when it is meaningful.

Do NOT replace a view just to increase angular diversity.

Only propose a replacement if it clearly improves the overall geometric
representation.

Return valid JSON only.

If the current set is already good:

{{
    "replace": false,
    "reason": "The selected set already provides a balanced geometric representation."
}}

If one replacement would improve the set:

{{
    "replace": true,
    "remove_index": 12,
    "add_index": 7,
    "reason": "Explain what is redundant or missing and why this replacement improves the final set."
}}

Rules:
- remove_index MUST belong to the selected set;
- add_index MUST belong to the remaining candidate set;
- propose at most one replacement.
"""