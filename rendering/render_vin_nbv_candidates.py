import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Vector


IMAGE_SIZE = 512

CAMERA_LENS_MM = 50.0
CAMERA_SENSOR_WIDTH_MM = 36.0

WORLD_COLOR = (0.95, 0.95, 0.95, 1.0)

OBJECT_COLOR = (0.08, 0.08, 0.08, 1.0)

TARGET_MAX_EXTENT = 1.0


def get_script_arguments():
    if "--" not in sys.argv:
        raise RuntimeError(
            "Expected arguments after '--': "
            "<input_obj> <output_dir>"
        )

    args = sys.argv[
        sys.argv.index("--") + 1 :
    ]

    if len(args) != 2:
        raise RuntimeError(
            "Usage: blender --background "
            "--python render_vin_nbv_candidates.py "
            "-- <input_obj> <output_dir>"
        )

    input_path = Path(args[0]).resolve()
    output_dir = Path(args[1]).resolve()

    return input_path, output_dir


def add_project_root_to_path():
    script_path = Path(__file__).resolve()

    project_root = script_path.parent.parent

    project_root_string = str(
        project_root
    )

    if project_root_string not in sys.path:
        sys.path.insert(
            0,
            project_root_string,
        )


def clear_scene():
    bpy.ops.object.select_all(
        action="SELECT"
    )

    bpy.ops.object.delete(
        use_global=False
    )

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(
            datablocks
        ):
            if datablock.users == 0:
                datablocks.remove(
                    datablock
                )


def import_obj(input_path):
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
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]

    if not mesh_objects:
        raise RuntimeError(
            f"No mesh objects found in {input_path}"
        )

    return mesh_objects


def get_world_bbox(objects):
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
            min(point.x for point in coordinates),
            min(point.y for point in coordinates),
            min(point.z for point in coordinates),
        )
    )

    maximum = Vector(
        (
            max(point.x for point in coordinates),
            max(point.y for point in coordinates),
            max(point.z for point in coordinates),
        )
    )

    return minimum, maximum


def normalize_objects(objects):
    bbox_min, bbox_max = (
        get_world_bbox(objects)
    )

    center = (
        bbox_min + bbox_max
    ) * 0.5

    extents = (
        bbox_max - bbox_min
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
            obj.location - center
        ) * scale

        obj.scale = (
            obj.scale * scale
        )

    bpy.context.view_layer.update()

    normalized_min, normalized_max = (
        get_world_bbox(objects)
    )

    normalized_extents = (
        normalized_max
        - normalized_min
    )

    return {
        "original_bbox_min": list(
            bbox_min
        ),
        "original_bbox_max": list(
            bbox_max
        ),
        "original_extents": list(
            extents
        ),
        "original_center": list(
            center
        ),
        "normalization_scale": scale,
        "normalized_bbox_min": list(
            normalized_min
        ),
        "normalized_bbox_max": list(
            normalized_max
        ),
        "normalized_extents": list(
            normalized_extents
        ),
    }


def create_material():
    material = bpy.data.materials.new(
        name="VIN_NBV_Material"
    )

    material.diffuse_color = (
        OBJECT_COLOR
    )

    material.use_nodes = True

    principled = (
        material.node_tree.nodes.get(
            "Principled BSDF"
        )
    )

    if principled is not None:
        if "Base Color" in principled.inputs:
            principled.inputs[
                "Base Color"
            ].default_value = (
                OBJECT_COLOR
            )

        if "Roughness" in principled.inputs:
            principled.inputs[
                "Roughness"
            ].default_value = 0.8

    return material


def apply_material(
    objects,
    material,
):
    for obj in objects:
        obj.data.materials.clear()

        obj.data.materials.append(
            material
        )


def setup_world():
    world = bpy.data.worlds.get(
        "World"
    )

    if world is None:
        world = bpy.data.worlds.new(
            "World"
        )

        bpy.context.scene.world = world

    world.use_nodes = True

    background = (
        world.node_tree.nodes.get(
            "Background"
        )
    )

    if background is not None:
        background.inputs[
            "Color"
        ].default_value = (
            WORLD_COLOR
        )

        background.inputs[
            "Strength"
        ].default_value = 0.8


def add_area_light(
    name,
    location,
    energy,
    size,
):
    light_data = (
        bpy.data.lights.new(
            name=name,
            type="AREA",
        )
    )

    light_data.energy = energy

    light_data.shape = "DISK"

    light_data.size = size

    light_object = (
        bpy.data.objects.new(
            name,
            light_data,
        )
    )

    bpy.context.collection.objects.link(
        light_object
    )

    light_object.location = (
        location
    )

    direction = (
        Vector((0.0, 0.0, 0.0))
        - light_object.location
    )

    light_object.rotation_euler = (
        direction.to_track_quat(
            "-Z",
            "Y",
        ).to_euler()
    )

    return light_object


def setup_lighting():
    add_area_light(
        name="KeyLight",
        location=(
            4.0,
            -4.0,
            5.0,
        ),
        energy=700.0,
        size=5.0,
    )

    add_area_light(
        name="FillLight",
        location=(
            -4.0,
            4.0,
            3.0,
        ),
        energy=500.0,
        size=5.0,
    )


def create_camera():
    camera_data = (
        bpy.data.cameras.new(
            name="VIN_NBV_Camera"
        )
    )

    camera_data.type = (
        "PERSP"
    )

    camera_data.lens = (
        CAMERA_LENS_MM
    )

    camera_data.sensor_width = (
        CAMERA_SENSOR_WIDTH_MM
    )

    camera_data.clip_start = 0.01

    camera_data.clip_end = 100.0

    camera_object = (
        bpy.data.objects.new(
            "VIN_NBV_Camera",
            camera_data,
        )
    )

    bpy.context.collection.objects.link(
        camera_object
    )

    bpy.context.scene.camera = (
        camera_object
    )

    return camera_object


def point_camera_at_origin(
    camera,
):
    direction = (
        Vector(
            (
                0.0,
                0.0,
                0.0,
            )
        )
        - camera.location
    )

    camera.rotation_euler = (
        direction.to_track_quat(
            "-Z",
            "Y",
        ).to_euler()
    )


def setup_render_settings():
    scene = bpy.context.scene

    scene.render.engine = (
        "BLENDER_EEVEE"
    )

    scene.render.resolution_x = (
        IMAGE_SIZE
    )

    scene.render.resolution_y = (
        IMAGE_SIZE
    )

    scene.render.resolution_percentage = (
        100
    )

    scene.render.image_settings.file_format = (
        "PNG"
    )

    scene.render.image_settings.color_mode = (
        "RGB"
    )

    scene.render.film_transparent = (
        False
    )

    # Blender 5 prefixes the AgX look names, while older
    # versions expose the same preset without the prefix.
    for look in (
        "AgX - Medium High Contrast",
        "Medium High Contrast",
    ):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    else:
        raise RuntimeError(
            "The Medium High Contrast color-management "
            "look is unavailable in this Blender version."
        )

    scene.render.use_file_extension = (
        True
    )


def enable_depth_pass():
    view_layer = (
        bpy.context.scene.view_layers[
            0
        ]
    )

    view_layer.use_pass_z = True


def setup_depth_compositor(
    depth_dir,
):
    scene = bpy.context.scene

    if hasattr(
        scene,
        "compositing_node_group",
    ):
        node_tree = bpy.data.node_groups.new(
            "VIN-NBV Depth Compositor",
            "CompositorNodeTree",
        )
        scene.compositing_node_group = node_tree
    else:
        scene.use_nodes = True
        node_tree = scene.node_tree
        node_tree.nodes.clear()

    render_layers = node_tree.nodes.new(
        "CompositorNodeRLayers"
    )
    file_output = node_tree.nodes.new(
        "CompositorNodeOutputFile"
    )

    file_output.name = "VIN-NBV Depth Output"
    file_output.directory = str(depth_dir)
    file_output.file_name = ".vin_nbv_depth"
    file_output.format.file_format = (
        "OPEN_EXR_MULTILAYER"
    )
    file_output.format.color_mode = "RGB"
    file_output.format.color_depth = "32"

    if hasattr(
        file_output,
        "file_output_items",
    ):
        file_output.file_output_items.new(
            "FLOAT",
            "Depth",
        )
        depth_input = file_output.inputs["Depth"]
    else:
        file_output.file_slots[0].path = (
            ".vin_nbv_depth"
        )
        depth_input = file_output.inputs[0]

    node_tree.links.new(
        render_layers.outputs["Depth"],
        depth_input,
    )


def get_camera_intrinsics(
    camera,
):
    scene = bpy.context.scene

    width = (
        scene.render.resolution_x
        * scene.render.resolution_percentage
        / 100.0
    )

    height = (
        scene.render.resolution_y
        * scene.render.resolution_percentage
        / 100.0
    )

    sensor_width = (
        camera.data.sensor_width
    )

    focal_length = (
        camera.data.lens
    )

    fx = (
        focal_length
        / sensor_width
        * width
    )

    fy = fx

    cx = width / 2.0

    cy = height / 2.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height),
        "lens_mm": float(
            focal_length
        ),
        "sensor_width_mm": float(
            sensor_width
        ),
    }


def matrix_to_list(matrix):
    return [
        [
            float(value)
            for value in row
        ]
        for row in matrix
    ]


def get_camera_matrices(
    camera,
):
    camera_to_world = (
        camera.matrix_world.copy()
    )

    world_to_camera = (
        camera_to_world.inverted()
    )

    return {
        "camera_to_world": (
            matrix_to_list(
                camera_to_world
            )
        ),
        "world_to_camera": (
            matrix_to_list(
                world_to_camera
            )
        ),
    }


def get_depth_pass(
    depth_dir,
):
    depth_files = sorted(
        depth_dir.glob(
            ".vin_nbv_depth*.exr"
        )
    )

    if not depth_files:
        raise RuntimeError(
            "Depth compositor EXR output not found."
        )

    depth_file = depth_files[-1]
    depth_input = oiio.ImageInput.open(
        str(depth_file)
    )

    if depth_input is None:
        raise RuntimeError(
            f"Could not open depth EXR: {depth_file}"
        )

    depth_spec = depth_input.spec()

    width = (
        bpy.context.scene.render.resolution_x
    )

    height = (
        bpy.context.scene.render.resolution_y
    )

    image_width, image_height = (
        depth_spec.width,
        depth_spec.height,
    )

    if (
        image_width != width
        or image_height != height
    ):
        raise RuntimeError(
            "Unexpected depth image size: "
            f"{image_width}x{image_height}, "
            f"expected {width}x{height}."
        )

    depth_pixels = depth_input.read_image(
        format=oiio.FLOAT,
    )
    depth_input.close()

    depth_file.unlink()

    depth = np.asarray(
        depth_pixels[
            :, :, 0
        ],
        dtype=np.float32,
    )

    return depth


def clean_depth(
    depth,
    clip_end,
):
    depth = depth.astype(
        np.float32,
        copy=True,
    )

    invalid = (
        ~np.isfinite(depth)
        | (depth <= 0.0)
        | (
            depth
            >= clip_end
            * 0.999
        )
    )

    depth[
        invalid
    ] = np.nan

    return depth


def render_view(
    camera,
    view,
    rgb_dir,
    depth_dir,
):
    index = view[
        "index"
    ]

    camera.location = (
        view[
            "camera_position"
        ]
    )

    point_camera_at_origin(
        camera
    )

    bpy.context.view_layer.update()

    rgb_path = (
        rgb_dir
        / f"view_{index:03d}.png"
    )

    depth_path = (
        depth_dir
        / f"view_{index:03d}.npy"
    )

    scene = bpy.context.scene

    scene.render.filepath = str(
        rgb_path
    )

    bpy.ops.render.render(
        write_still=True
    )

    depth = get_depth_pass(
        depth_dir,
    )

    depth = clean_depth(
        depth,
        camera.data.clip_end,
    )

    np.save(
        depth_path,
        depth,
    )

    matrices = get_camera_matrices(
        camera
    )

    valid_depth = depth[
        np.isfinite(depth)
    ]

    if valid_depth.size > 0:
        min_depth = float(
            valid_depth.min()
        )

        max_depth = float(
            valid_depth.max()
        )

        valid_pixels = int(
            valid_depth.size
        )
    else:
        min_depth = None
        max_depth = None
        valid_pixels = 0

    return {
        **view,
        "rgb": str(
            rgb_path
        ),
        "depth": str(
            depth_path
        ),
        "camera_to_world": (
            matrices[
                "camera_to_world"
            ]
        ),
        "world_to_camera": (
            matrices[
                "world_to_camera"
            ]
        ),
        "depth_statistics": {
            "valid_pixels": (
                valid_pixels
            ),
            "min_depth": (
                min_depth
            ),
            "max_depth": (
                max_depth
            ),
        },
    }


def main():
    add_project_root_to_path()

    from baselines.vin_nbv.cameras import (
        generate_hemisphere_views,
    )

    input_path, output_dir = (
        get_script_arguments()
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input OBJ does not exist: "
            f"{input_path}"
        )

    rgb_dir = (
        output_dir
        / "rgb"
    )

    depth_dir = (
        output_dir
        / "depth"
    )

    rgb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    depth_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    material = (
        create_material()
    )

    apply_material(
        objects,
        material,
    )

    setup_world()

    setup_lighting()

    camera = create_camera()

    setup_render_settings()

    enable_depth_pass()
    setup_depth_compositor(
        depth_dir,
    )

    candidate_views = (
        generate_hemisphere_views()
    )

    intrinsics = (
        get_camera_intrinsics(
            camera
        )
    )

    metadata = []

    total_views = len(
        candidate_views
    )

    print(
        "[VIN-NBV] "
        f"Rendering {total_views} "
        "candidate views"
    )

    for position, view in enumerate(
        candidate_views,
        start=1,
    ):
        print(
            "[VIN-NBV] "
            f"Rendering "
            f"{position:03d}/"
            f"{total_views:03d} "
            f"(view "
            f"{view['index']:03d})"
        )

        rendered_view = (
            render_view(
                camera=camera,
                view=view,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
            )
        )

        metadata.append(
            rendered_view
        )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    metadata_payload = {
        "input": str(
            input_path
        ),
        "num_candidates": (
            total_views
        ),
        "image_size": (
            IMAGE_SIZE
        ),
        "camera": {
            "type": "perspective",
            "intrinsics": (
                intrinsics
            ),
            "clip_start": float(
                camera.data.clip_start
            ),
            "clip_end": float(
                camera.data.clip_end
            ),
        },
        "normalization": (
            normalization
        ),
        "views": (
            metadata
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata_payload,
            file,
            indent=2,
        )

    print()
    print(
        "[VIN-NBV] Done."
    )

    print(
        "[VIN-NBV] RGB: "
        f"{rgb_dir}"
    )

    print(
        "[VIN-NBV] Depth: "
        f"{depth_dir}"
    )

    print(
        "[VIN-NBV] Metadata: "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()
