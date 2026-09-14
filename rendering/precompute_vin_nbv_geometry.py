import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


DEFAULT_GT_POINTS = 100_000

# Для visibility не обязательно пускать 512×512 лучей.
# 192×192 уже дает плотное surface observation,
# а вычисляется заметно быстрее.
DEFAULT_VISIBILITY_RESOLUTION = 192

TARGET_MAX_EXTENT = 1.0

RANDOM_SEED = 42


def get_script_arguments():
    if "--" not in sys.argv:
        raise RuntimeError(
            "Expected arguments after '--': "
            "<input_obj> <metadata_json> <output_dir>"
        )

    args = sys.argv[
        sys.argv.index("--") + 1 :
    ]

    if len(args) < 3:
        raise RuntimeError(
            "Usage: blender --background "
            "--python precompute_vin_nbv_geometry.py "
            "-- <input_obj> <metadata_json> <output_dir> "
            "[num_gt_points] [visibility_resolution]"
        )

    input_path = Path(
        args[0]
    ).resolve()

    metadata_path = Path(
        args[1]
    ).resolve()

    output_dir = Path(
        args[2]
    ).resolve()

    num_gt_points = (
        int(args[3])
        if len(args) >= 4
        else DEFAULT_GT_POINTS
    )

    visibility_resolution = (
        int(args[4])
        if len(args) >= 5
        else DEFAULT_VISIBILITY_RESOLUTION
    )

    return (
        input_path,
        metadata_path,
        output_dir,
        num_gt_points,
        visibility_resolution,
    )


def clear_scene():
    bpy.ops.object.select_all(
        action="SELECT"
    )

    bpy.ops.object.delete(
        use_global=False
    )


def import_obj(
    input_path,
):
    try:
        bpy.ops.wm.obj_import(
            filepath=str(
                input_path
            )
        )
    except AttributeError:
        bpy.ops.import_scene.obj(
            filepath=str(
                input_path
            )
        )

    mesh_objects = [
        obj
        for obj
        in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    if not mesh_objects:
        raise RuntimeError(
            f"No mesh objects found in "
            f"{input_path}"
        )

    return mesh_objects


def get_world_bbox(
    objects,
):
    coordinates = []

    for obj in objects:
        for corner in obj.bound_box:
            world_corner = (
                obj.matrix_world
                @ Vector(corner)
            )

            coordinates.append(
                world_corner
            )

    minimum = Vector(
        (
            min(
                point.x
                for point
                in coordinates
            ),
            min(
                point.y
                for point
                in coordinates
            ),
            min(
                point.z
                for point
                in coordinates
            ),
        )
    )

    maximum = Vector(
        (
            max(
                point.x
                for point
                in coordinates
            ),
            max(
                point.y
                for point
                in coordinates
            ),
            max(
                point.z
                for point
                in coordinates
            ),
        )
    )

    return (
        minimum,
        maximum,
    )


def normalize_objects(
    objects,
):
    bbox_min, bbox_max = (
        get_world_bbox(
            objects
        )
    )

    center = (
        bbox_min
        + bbox_max
    ) * 0.5

    extents = (
        bbox_max
        - bbox_min
    )

    max_extent = max(
        extents
    )

    if max_extent <= 0:
        raise RuntimeError(
            "Invalid mesh extent."
        )

    scale = (
        TARGET_MAX_EXTENT
        / max_extent
    )

    for obj in objects:
        obj.location = (
            obj.location
            - center
        ) * scale

        obj.scale = (
            obj.scale
            * scale
        )

    bpy.context.view_layer.update()

    normalized_min, normalized_max = (
        get_world_bbox(
            objects
        )
    )

    return {
        "original_bbox_min": [
            float(value)
            for value
            in bbox_min
        ],
        "original_bbox_max": [
            float(value)
            for value
            in bbox_max
        ],
        "center": [
            float(value)
            for value
            in center
        ],
        "scale": float(
            scale
        ),
        "normalized_bbox_min": [
            float(value)
            for value
            in normalized_min
        ],
        "normalized_bbox_max": [
            float(value)
            for value
            in normalized_max
        ],
    }


def extract_world_triangles(
    objects,
):
    """
    Extract every mesh triangle in Blender world coordinates.

    Returns
    -------
    np.ndarray
        Shape (T, 3, 3).
    """

    triangles = []

    depsgraph = (
        bpy.context.evaluated_depsgraph_get()
    )

    for obj in objects:
        evaluated = (
            obj.evaluated_get(
                depsgraph
            )
        )

        mesh = (
            evaluated.to_mesh()
        )

        mesh.calc_loop_triangles()

        matrix_world = (
            evaluated.matrix_world
        )

        for triangle in mesh.loop_triangles:
            points = []

            for vertex_index in (
                triangle.vertices
            ):
                coordinate = (
                    matrix_world
                    @ mesh.vertices[
                        vertex_index
                    ].co
                )

                points.append(
                    (
                        float(
                            coordinate.x
                        ),
                        float(
                            coordinate.y
                        ),
                        float(
                            coordinate.z
                        ),
                    )
                )

            triangles.append(
                points
            )

        evaluated.to_mesh_clear()

    if not triangles:
        raise RuntimeError(
            "No mesh triangles found."
        )

    return np.asarray(
        triangles,
        dtype=np.float64,
    )


def triangle_areas(
    triangles,
):
    edge_1 = (
        triangles[:, 1]
        - triangles[:, 0]
    )

    edge_2 = (
        triangles[:, 2]
        - triangles[:, 0]
    )

    cross = np.cross(
        edge_1,
        edge_2,
    )

    return (
        0.5
        * np.linalg.norm(
            cross,
            axis=1,
        )
    )


def sample_surface_points(
    triangles,
    num_points,
    seed=RANDOM_SEED,
):
    """
    Uniform area-weighted sampling over the complete mesh surface.
    """

    areas = triangle_areas(
        triangles
    )

    valid = (
        areas > 0.0
    )

    triangles = (
        triangles[
            valid
        ]
    )

    areas = (
        areas[
            valid
        ]
    )

    total_area = float(
        areas.sum()
    )

    if total_area <= 0.0:
        raise RuntimeError(
            "Mesh has zero surface area."
        )

    probabilities = (
        areas
        / total_area
    )

    rng = np.random.default_rng(
        seed
    )

    triangle_indices = (
        rng.choice(
            len(triangles),
            size=num_points,
            replace=True,
            p=probabilities,
        )
    )

    selected = (
        triangles[
            triangle_indices
        ]
    )

    u = rng.random(
        num_points
    )

    v = rng.random(
        num_points
    )

    # Uniform barycentric sampling.
    sqrt_u = np.sqrt(
        u
    )

    weight_a = (
        1.0
        - sqrt_u
    )

    weight_b = (
        sqrt_u
        * (
            1.0
            - v
        )
    )

    weight_c = (
        sqrt_u
        * v
    )

    points = (
        weight_a[:, None]
        * selected[:, 0]
        + weight_b[:, None]
        * selected[:, 1]
        + weight_c[:, None]
        * selected[:, 2]
    )

    return points.astype(
        np.float32
    )


def build_bvh(
    triangles,
):
    """
    Build one BVH containing the complete normalized mesh.
    """

    vertices = []

    polygons = []

    for triangle in triangles:
        start = len(
            vertices
        )

        vertices.extend(
            [
                Vector(
                    triangle[0]
                ),
                Vector(
                    triangle[1]
                ),
                Vector(
                    triangle[2]
                ),
            ]
        )

        polygons.append(
            (
                start,
                start + 1,
                start + 2,
            )
        )

    return BVHTree.FromPolygons(
        vertices,
        polygons,
        all_triangles=True,
    )


def load_metadata(
    metadata_path,
):
    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def scaled_intrinsics(
    original_intrinsics,
    resolution,
):
    original_width = float(
        original_intrinsics[
            "width"
        ]
    )

    original_height = float(
        original_intrinsics[
            "height"
        ]
    )

    scale_x = (
        resolution
        / original_width
    )

    scale_y = (
        resolution
        / original_height
    )

    return {
        "fx": (
            float(
                original_intrinsics[
                    "fx"
                ]
            )
            * scale_x
        ),
        "fy": (
            float(
                original_intrinsics[
                    "fy"
                ]
            )
            * scale_y
        ),
        "cx": (
            float(
                original_intrinsics[
                    "cx"
                ]
            )
            * scale_x
        ),
        "cy": (
            float(
                original_intrinsics[
                    "cy"
                ]
            )
            * scale_y
        ),
    }


def raycast_visible_points(
    bvh,
    view,
    intrinsics,
    resolution,
):
    """
    Cast one ray per image pixel.

    For every ray, keep only the first intersection with the mesh.
    This is the exact visible surface from this camera pose.
    """

    camera_to_world = np.asarray(
        view[
            "camera_to_world"
        ],
        dtype=np.float64,
    )

    rotation = (
        camera_to_world[
            :3,
            :3
        ]
    )

    origin_np = (
        camera_to_world[
            :3,
            3
        ]
    )

    origin = Vector(
        origin_np
    )

    fx = float(
        intrinsics[
            "fx"
        ]
    )

    fy = float(
        intrinsics[
            "fy"
        ]
    )

    cx = float(
        intrinsics[
            "cx"
        ]
    )

    cy = float(
        intrinsics[
            "cy"
        ]
    )

    points = []

    for pixel_y in range(
        resolution
    ):
        for pixel_x in range(
            resolution
        ):
            camera_x = (
                (
                    pixel_x
                    + 0.5
                    - cx
                )
                / fx
            )

            camera_y = (
                -(
                    pixel_y
                    + 0.5
                    - cy
                )
                / fy
            )

            # Blender camera looks down local -Z.
            camera_direction = np.array(
                [
                    camera_x,
                    camera_y,
                    -1.0,
                ],
                dtype=np.float64,
            )

            camera_direction /= (
                np.linalg.norm(
                    camera_direction
                )
            )

            world_direction = (
                rotation
                @ camera_direction
            )

            world_direction /= (
                np.linalg.norm(
                    world_direction
                )
            )

            direction = Vector(
                world_direction
            )

            (
                location,
                _normal,
                _face_index,
                _distance,
            ) = bvh.ray_cast(
                origin,
                direction,
            )

            if location is not None:
                points.append(
                    (
                        location.x,
                        location.y,
                        location.z,
                    )
                )

    if not points:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

    return np.asarray(
        points,
        dtype=np.float32,
    )


def main():
    (
        input_path,
        metadata_path,
        output_dir,
        num_gt_points,
        visibility_resolution,
    ) = get_script_arguments()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input OBJ not found: "
            f"{input_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: "
            f"{metadata_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visible_dir = (
        output_dir
        / "visible"
    )

    visible_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[VIN-NBV Geometry] "
        "Loading mesh"
    )

    clear_scene()

    objects = import_obj(
        input_path
    )

    normalization = (
        normalize_objects(
            objects
        )
    )

    print(
        "[VIN-NBV Geometry] "
        "Extracting normalized mesh triangles"
    )

    triangles = (
        extract_world_triangles(
            objects
        )
    )

    print(
        "[VIN-NBV Geometry] "
        f"Triangles: {len(triangles)}"
    )

    print(
        "[VIN-NBV Geometry] "
        f"Sampling {num_gt_points} "
        "ground-truth surface points"
    )

    ground_truth = (
        sample_surface_points(
            triangles=triangles,
            num_points=(
                num_gt_points
            ),
        )
    )

    ground_truth_path = (
        output_dir
        / "ground_truth.npy"
    )

    np.save(
        ground_truth_path,
        ground_truth,
    )

    print(
        "[VIN-NBV Geometry] "
        "Building BVH"
    )

    bvh = build_bvh(
        triangles
    )

    metadata = load_metadata(
        metadata_path
    )

    views = (
        metadata[
            "views"
        ]
    )

    intrinsics = scaled_intrinsics(
        metadata[
            "camera"
        ][
            "intrinsics"
        ],
        resolution=(
            visibility_resolution
        ),
    )

    visibility_metadata = []

    total_views = len(
        views
    )

    print(
        "[VIN-NBV Geometry] "
        f"Ray casting {total_views} "
        f"views at "
        f"{visibility_resolution}x"
        f"{visibility_resolution}"
    )

    for position, view in enumerate(
        views,
        start=1,
    ):
        index = int(
            view[
                "index"
            ]
        )

        print(
            "[VIN-NBV Geometry] "
            f"{position:03d}/"
            f"{total_views:03d} "
            f"view {index:03d}"
        )

        visible_points = (
            raycast_visible_points(
                bvh=bvh,
                view=view,
                intrinsics=(
                    intrinsics
                ),
                resolution=(
                    visibility_resolution
                ),
            )
        )

        visible_path = (
            visible_dir
            / f"view_{index:03d}.npy"
        )

        np.save(
            visible_path,
            visible_points,
        )

        visibility_metadata.append(
            {
                "index": index,
                "path": str(
                    visible_path
                ),
                "num_visible_points": int(
                    len(
                        visible_points
                    )
                ),
            }
        )

    geometry_metadata = {
        "input": str(
            input_path
        ),
        "candidate_metadata": str(
            metadata_path
        ),
        "num_ground_truth_points": int(
            len(
                ground_truth
            )
        ),
        "ground_truth": str(
            ground_truth_path
        ),
        "visibility_resolution": int(
            visibility_resolution
        ),
        "normalization": (
            normalization
        ),
        "views": (
            visibility_metadata
        ),
    }

    geometry_metadata_path = (
        output_dir
        / "geometry_metadata.json"
    )

    with geometry_metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            geometry_metadata,
            file,
            indent=2,
        )

    print()
    print(
        "[VIN-NBV Geometry] Done."
    )

    print(
        "[VIN-NBV Geometry] GT: "
        f"{ground_truth_path}"
    )

    print(
        "[VIN-NBV Geometry] Visible: "
        f"{visible_dir}"
    )

    print(
        "[VIN-NBV Geometry] Metadata: "
        f"{geometry_metadata_path}"
    )


if __name__ == "__main__":
    main()