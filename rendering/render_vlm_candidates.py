import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

AZIMUTHS = [
    0,
    45,
    90,
    135,
    180,
    225,
    270,
    315,
]

ELEVATIONS = [
    -35,
    15,
    45,
    70,
]

CAMERA_RADIUS = 3.0
ORTHO_SCALE = 1.8


def parse_args():
    argv = sys.argv

    if "--" not in argv:
        raise RuntimeError("Expected arguments after --")

    args = argv[argv.index("--") + 1 :]

    if len(args) != 2:
        raise RuntimeError("Expected: model_path output_dir")

    model_path = Path(args[0]).resolve()

    output_dir = Path(args[1]).resolve()

    return (
        model_path,
        output_dir,
    )


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")

    bpy.ops.object.delete(use_global=False)

    for datablock in bpy.data.meshes:
        if datablock.users == 0:
            bpy.data.meshes.remove(datablock)


def import_obj(model_path):
    bpy.ops.wm.obj_import(filepath=str(model_path))

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not mesh_objects:
        raise RuntimeError("No mesh objects found " "after OBJ import.")

    return mesh_objects


def get_world_bbox(
    objects,
):
    points = []

    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)

            points.append(point)

    xs = [point.x for point in points]

    ys = [point.y for point in points]

    zs = [point.z for point in points]

    bbox_min = Vector(
        (
            min(xs),
            min(ys),
            min(zs),
        )
    )

    bbox_max = Vector(
        (
            max(xs),
            max(ys),
            max(zs),
        )
    )

    return (
        bbox_min,
        bbox_max,
    )


def normalize_objects(
    objects,
):
    bbox_min, bbox_max = get_world_bbox(objects)

    center = (bbox_min + bbox_max) * 0.5

    extents = bbox_max - bbox_min

    max_extent = max(
        extents.x,
        extents.y,
        extents.z,
    )

    if max_extent <= 0:
        raise RuntimeError("Invalid mesh dimensions.")

    scale = 1.0 / max_extent

    for obj in objects:
        obj.location = (obj.location - center) * scale

        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()


def create_material():
    material = bpy.data.materials.new(name="VLMObjectMaterial")

    material.diffuse_color = (
        0.08,
        0.08,
        0.08,
        1.0,
    )

    material.use_nodes = True

    principled = material.node_tree.nodes.get("Principled BSDF")

    if principled is not None:
        principled.inputs["Base Color"].default_value = (
            0.08,
            0.08,
            0.08,
            1.0,
        )

        principled.inputs["Roughness"].default_value = 0.8

    return material


def apply_material(
    objects,
    material,
):
    for obj in objects:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def setup_world():
    world = bpy.context.scene.world

    if world is None:
        world = bpy.data.worlds.new("World")

        bpy.context.scene.world = world

    world.use_nodes = True

    background = world.node_tree.nodes.get("Background")

    if background is not None:
        background.inputs["Color"].default_value = (
            0.95,
            0.95,
            0.95,
            1.0,
        )

        background.inputs["Strength"].default_value = 0.8


def create_area_light(
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

    light_object = bpy.data.objects.new(
        name,
        light_data,
    )

    light_object.location = location

    bpy.context.collection.objects.link(light_object)

    return light_object


def setup_lights():
    create_area_light(
        "KeyLight",
        (
            4.0,
            -4.0,
            5.0,
        ),
        700.0,
    )

    create_area_light(
        "FillLight",
        (
            -4.0,
            4.0,
            3.0,
        ),
        500.0,
    )


def create_camera():
    camera_data = bpy.data.cameras.new("VLMCamera")

    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE

    camera = bpy.data.objects.new(
        "VLMCamera",
        camera_data,
    )

    bpy.context.collection.objects.link(camera)

    bpy.context.scene.camera = camera

    return camera


def look_at_origin(
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


def camera_position(
    azimuth,
    elevation,
):
    azimuth_rad = math.radians(azimuth)

    elevation_rad = math.radians(elevation)

    horizontal_radius = CAMERA_RADIUS * math.cos(elevation_rad)

    x = horizontal_radius * math.sin(azimuth_rad)

    y = -horizontal_radius * math.cos(azimuth_rad)

    z = CAMERA_RADIUS * math.sin(elevation_rad)

    return (
        x,
        y,
        z,
    )


def setup_render():
    scene = bpy.context.scene

    scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"

    scene.render.film_transparent = False


def render_candidates(
    camera,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = []

    index = 0

    for elevation in ELEVATIONS:
        for azimuth in AZIMUTHS:
            position = camera_position(
                azimuth,
                elevation,
            )

            camera.location = position

            look_at_origin(camera)

            output_path = output_dir / f"view_{index:02d}.png"

            bpy.context.scene.render.filepath = str(output_path)

            bpy.ops.render.render(write_still=True)

            metadata.append(
                {
                    "index": index,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "camera_position": [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ],
                    "image": str(output_path),
                }
            )

            print(
                "[VLM Iterative] "
                f"Rendered candidate {index:02d}: "
                f"azimuth={azimuth}, "
                f"elevation={elevation}"
            )

            index += 1

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("[VLM Iterative] " f"Rendered {len(metadata)} candidates")

    print("[VLM Iterative] Metadata: " f"{metadata_path}")


def main():
    (
        model_path,
        output_dir,
    ) = parse_args()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: " f"{model_path}")

    print()
    print("=" * 50)
    print("VLM Iterative candidate renderer")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()

    clear_scene()

    objects = import_obj(model_path)

    normalize_objects(objects)

    material = create_material()

    apply_material(
        objects,
        material,
    )

    setup_world()
    setup_lights()
    setup_render()

    camera = create_camera()

    render_candidates(
        camera,
        output_dir,
    )


if __name__ == "__main__":
    main()
