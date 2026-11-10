import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 1:
        raise ValueError("Expected one render_request.json path")
    request = json.loads(Path(args[0]).read_text())
    geometry_path = Path(request["geometry"])
    with geometry_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != request["geometry_sha256"]:
        raise ValueError("Geometry changed before rendering")
    output_dir = Path(request["output_dir"])
    views_dir = output_dir / "views"
    views_dir.mkdir(exist_ok=False)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    with np.load(geometry_path, allow_pickle=False) as archive:
        mesh = bpy.data.meshes.new("MVTNMesh")
        mesh.from_pydata(archive["vertices"].tolist(), [], archive["faces"].tolist())
        mesh.update()
    obj = bpy.data.objects.new("MVTNObject", mesh)
    bpy.context.collection.objects.link(obj)
    material = bpy.data.materials.new("MVTNMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    obj.data.materials.append(material)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = request["samples"]
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False
    scene.render.resolution_x = request["image_size"]
    scene.render.resolution_y = request["image_size"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1)
    background.inputs["Strength"].default_value = 0.8
    for index, (position, energy) in enumerate([((4, 4, 4), 700), ((-4, 2, -3), 500)]):
        data = bpy.data.lights.new(f"MVTNLight{index}", "AREA")
        data.energy = energy
        data.size = 5
        light = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(light)
        light.location = position
        light.rotation_euler = (-Vector(position)).to_track_quat("-Z", "Y").to_euler()
    data = bpy.data.cameras.new("MVTNCamera")
    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = 18 / math.tan(math.radians(request["fov_degrees"]) / 2)
    data.clip_start = request["clip_start"]
    data.clip_end = request["clip_end"]
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    rendered = []
    for pose in request["poses"]:
        camera.matrix_world = Matrix(pose["matrix_world"])
        bpy.context.view_layer.update()
        path = views_dir / f"view_{pose['index']:03d}.png"
        if path.exists():
            raise FileExistsError(path)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append(
            {
                **pose,
                "image": str(path),
                "actual_matrix_world": [list(row) for row in camera.matrix_world],
                "projection_probes": [
                    {
                        "point": list(point),
                        "ndc": list(world_to_camera_view(scene, camera, Vector(point))),
                    }
                    for point in ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1))
                ],
            }
        )
    metadata = {
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "device": "CPU",
        "image_size": request["image_size"],
        "samples": request["samples"],
        "seed": 42,
        "coordinate_frame": "mvtn_y_up",
        "projection": "perspective",
        "fov_degrees": request["fov_degrees"],
        "clip_start": data.clip_start,
        "clip_end": data.clip_end,
        "mesh_normalized_again": False,
        "geometry_sha256": digest,
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.polygons),
        "mesh_matrix_world": [list(row) for row in obj.matrix_world],
        "mesh_bounds": {
            "min": [
                min(vertex.co[axis] for vertex in mesh.vertices) for axis in range(3)
            ],
            "max": [
                max(vertex.co[axis] for vertex in mesh.vertices) for axis in range(3)
            ],
        },
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "view_transform": scene.view_settings.view_transform,
        "material_rgba": [0.08, 0.08, 0.08, 1.0],
        "roughness": 0.8,
        "world_rgba": [0.95, 0.95, 0.95, 1.0],
        "world_strength": 0.8,
        "area_lights": [
            {"position": [4, 4, 4], "energy": 700, "size": 5},
            {"position": [-4, 2, -3], "energy": 500, "size": 5},
        ],
        "views": rendered,
    }
    (output_dir / "blender_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
