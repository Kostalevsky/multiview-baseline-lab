import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

IMAGE_SIZE = 512
ORTHO_SCALE = 1.8

OBJECT_COLOR = (
    0.08,
    0.08,
    0.08,
    1.0,
)

WORLD_COLOR = (
    0.95,
    0.95,
    0.95,
    1.0,
)

TARGET_MAX_EXTENT = 1.0


VIEWS = {
    "pos_x": (
        3.0,
        0.0,
        0.0,
    ),
    "neg_x": (
        -3.0,
        0.0,
        0.0,
    ),
    "pos_y": (
        0.0,
        3.0,
        0.0,
    ),
    "neg_y": (
        0.0,
        -3.0,
        0.0,
    ),
    "pos_z": (
        0.0,
        0.0,
        3.0,
    ),
    "neg_z": (
        0.0,
        0.0,
        -3.0,
    ),
    "iso_1": (
        3.0,
        -3.0,
        3.0,
    ),
    "iso_2": (
        -3.0,
        -3.0,
        3.0,
    ),
}


def get_script_arguments():
    if "--" not in sys.argv:
        raise RuntimeError("Expected arguments after '--': " "<input_obj> <output_dir>")

    args = sys.argv[sys.argv.index("--") + 1 :]

    if len(args) != 2:
        raise RuntimeError(
            "Expected exactly two arguments: " "<input_obj> <output_dir>"
        )

    return (
        Path(args[0]).resolve(),
        Path(args[1]).resolve(),
    )


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")

    bpy.ops.object.delete(use_global=False)


def import_obj(input_path):
    try:
        bpy.ops.wm.obj_import(filepath=str(input_path))
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=str(input_path))

    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not objects:
        raise RuntimeError("No mesh objects were imported.")

    return objects


def get_world_bbox(objects):
    points = []

    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))

    bbox_min = Vector(
        (
            min(p.x for p in points),
            min(p.y for p in points),
            min(p.z for p in points),
        )
    )

    bbox_max = Vector(
        (
            max(p.x for p in points),
            max(p.y for p in points),
            max(p.z for p in points),
        )
    )

    return (
        bbox_min,
        bbox_max,
    )


def normalize_objects(objects):
    bbox_min, bbox_max = get_world_bbox(objects)

    center = (bbox_min + bbox_max) * 0.5

    extents = bbox_max - bbox_min

    max_extent = max(extents)

    if max_extent <= 0:
        raise RuntimeError("Invalid mesh extent.")

    scale = TARGET_MAX_EXTENT / max_extent

    for obj in objects:
        obj.location = (obj.location - center) * scale

        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()


def create_material():
    material = bpy.data.materials.new(name="CanonicalMaterial")

    material.diffuse_color = OBJECT_COLOR

    material.use_nodes = True

    principled = material.node_tree.nodes.get("Principled BSDF")

    if principled:
        principled.inputs["Base Color"].default_value = OBJECT_COLOR

        principled.inputs["Roughness"].default_value = 0.8

    return material


def apply_material(
    objects,
    material,
):
    for obj in objects:
        obj.data.materials.clear()

        obj.data.materials.append(material)


def setup_world():
    world = bpy.data.worlds.get("World")

    if world is None:
        world = bpy.data.worlds.new("World")

        bpy.context.scene.world = world

    world.use_nodes = True

    background = world.node_tree.nodes.get("Background")

    background.inputs["Color"].default_value = WORLD_COLOR

    background.inputs["Strength"].default_value = 0.8


def add_area_light(
    name,
    location,
    energy,
):
    light_data = bpy.data.lights.new(
        name=name,
        type="AREA",
    )

    light_data.energy = energy

    light_data.shape = "DISK"

    light_data.size = 5.0

    light = bpy.data.objects.new(
        name,
        light_data,
    )

    bpy.context.collection.objects.link(light)

    light.location = location

    direction = (
        Vector(
            (
                0.0,
                0.0,
                0.0,
            )
        )
        - light.location
    )

    light.rotation_euler = direction.to_track_quat(
        "-Z",
        "Y",
    ).to_euler()


def setup_lighting():
    add_area_light(
        "KeyLight",
        (
            4.0,
            -4.0,
            5.0,
        ),
        700.0,
    )

    add_area_light(
        "FillLight",
        (
            -4.0,
            4.0,
            3.0,
        ),
        500.0,
    )


def create_camera():
    camera_data = bpy.data.cameras.new("CanonicalCamera")

    camera_data.type = "ORTHO"

    camera_data.ortho_scale = ORTHO_SCALE

    camera = bpy.data.objects.new(
        "CanonicalCamera",
        camera_data,
    )

    bpy.context.collection.objects.link(camera)

    bpy.context.scene.camera = camera

    return camera


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

    camera.rotation_euler = direction.to_track_quat(
        "-Z",
        "Y",
    ).to_euler()


def setup_render():
    scene = bpy.context.scene

    scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = IMAGE_SIZE

    scene.render.resolution_y = IMAGE_SIZE

    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"

    scene.render.image_settings.color_mode = "RGB"

    scene.render.film_transparent = False


def render_views(
    camera,
    output_dir,
):
    views_dir = output_dir / "views"

    views_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []

    for index, (
        name,
        position,
    ) in enumerate(VIEWS.items()):
        print("[Canonical] " f"Rendering " f"{index + 1}/8: " f"{name}")

        camera.location = position

        point_camera_at_origin(camera)

        bpy.context.view_layer.update()

        output_path = views_dir / f"{name}.png"

        bpy.context.scene.render.filepath = str(output_path)

        bpy.ops.render.render(write_still=True)

        metadata.append(
            {
                "index": index,
                "name": name,
                "camera_type": ("orthographic"),
                "camera_position": [float(value) for value in position],
                "ortho_scale": (ORTHO_SCALE),
                "image": str(output_path),
            }
        )

    return metadata


def main():
    (
        input_path,
        output_dir,
    ) = get_script_arguments()

    if not input_path.exists():
        raise FileNotFoundError(f"Input OBJ not found: " f"{input_path}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clear_scene()

    objects = import_obj(input_path)

    normalize_objects(objects)

    material = create_material()

    apply_material(
        objects,
        material,
    )

    setup_world()

    setup_lighting()

    camera = create_camera()

    setup_render()

    metadata = render_views(
        camera=camera,
        output_dir=output_dir,
    )

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "baseline": ("canonical"),
                "num_views": 8,
                "views": metadata,
            },
            file,
            indent=2,
        )

    print()
    print("[Canonical] Done.")

    print("[Canonical] Metadata: " f"{metadata_path}")


if __name__ == "__main__":
    main()
