import bpy
import sys
from pathlib import Path

from mathutils import Vector


def get_args():
    if "--" not in sys.argv:
        raise RuntimeError("Arguments were not provided.")

    args = sys.argv[sys.argv.index("--") + 1:]

    if len(args) < 2:
        raise RuntimeError("Expected: model_path output_path")

    return Path(args[0]).resolve(), Path(args[1]).resolve()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_obj(model_path):
    bpy.ops.wm.obj_import(filepath=str(model_path))


def get_mesh_objects():
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
    ]


def scene_bbox(mesh_objects):
    bbox_min = Vector((float("inf"), float("inf"), float("inf")))
    bbox_max = Vector((-float("inf"), -float("inf"), -float("inf")))

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)

            bbox_min.x = min(bbox_min.x, world_corner.x)
            bbox_min.y = min(bbox_min.y, world_corner.y)
            bbox_min.z = min(bbox_min.z, world_corner.z)

            bbox_max.x = max(bbox_max.x, world_corner.x)
            bbox_max.y = max(bbox_max.y, world_corner.y)
            bbox_max.z = max(bbox_max.z, world_corner.z)

    return bbox_min, bbox_max


def normalize_scene(mesh_objects):
    bbox_min, bbox_max = scene_bbox(mesh_objects)

    size = bbox_max - bbox_min
    scale = 1.0 / max(size)

    for obj in mesh_objects:
        obj.scale *= scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox(mesh_objects)

    center = (bbox_min + bbox_max) / 2

    for obj in mesh_objects:
        obj.location -= center

    bpy.context.view_layer.update()


def apply_material(mesh_objects):
    material = bpy.data.materials.new(
        name="BaselineMaterial"
    )

    material.use_nodes = True

    bsdf = material.node_tree.nodes.get(
        "Principled BSDF"
    )

    bsdf.inputs["Base Color"].default_value = (
        0.08,
        0.08,
        0.08,
        1.0,
    )

    bsdf.inputs["Roughness"].default_value = 0.8

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def create_camera():
    camera_data = bpy.data.cameras.new("Camera")

    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.7

    camera = bpy.data.objects.new(
        "Camera",
        camera_data,
    )

    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    return camera


def set_isometric_view(camera):
    camera.location = (
        3.0,
        -3.0,
        3.0,
    )

    direction = Vector((0, 0, 0)) - camera.location

    camera.rotation_euler = direction.to_track_quat(
        "-Z",
        "Y",
    ).to_euler()


def create_light():
    positions = [
        (4, -4, 5),
        (-4, 4, 3),
    ]

    energies = [
        700,
        500,
    ]

    for i, (position, energy) in enumerate(
        zip(positions, energies)
    ):
        light_data = bpy.data.lights.new(
            name=f"Light_{i}",
            type="AREA",
        )

        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = 5

        light = bpy.data.objects.new(
            f"Light_{i}",
            light_data,
        )

        bpy.context.scene.collection.objects.link(light)

        light.location = position


def setup_render(output_path):
    scene = bpy.context.scene

    scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path)

    scene.world.use_nodes = True

    background = scene.world.node_tree.nodes.get("Background")

    background.inputs["Color"].default_value = (
        0.95,
        0.95,
        0.95,
        1.0,
    )

    background.inputs["Strength"].default_value = 0.8


def render_isometric_view():
    print("[ISO Isometric] Rendering isometric view")

    bpy.ops.render.render(
        write_still=True
    )


def main():
    model_path, output_path = get_args()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[ISO Isometric] Model: {model_path}")
    print(f"[ISO Isometric] Output: {output_path}")

    clear_scene()
    import_obj(model_path)

    mesh_objects = get_mesh_objects()

    if not mesh_objects:
        raise RuntimeError("No mesh objects found.")

    normalize_scene(mesh_objects)
    apply_material(mesh_objects)

    camera = create_camera()
    set_isometric_view(camera)

    create_light()
    setup_render(output_path)

    render_isometric_view()

    print("[ISO Isometric] Rendering completed.")


if __name__ == "__main__":
    main()