import json
import math
import sys
from pathlib import Path

import bpy


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import render_candidates as renderer

    input_path, output_dir, view_spec = sys.argv[sys.argv.index("--") + 1 :]
    views = json.loads(view_spec)

    renderer.clear_scene()
    renderer.import_obj(Path(input_path))
    mesh_objects = renderer.get_mesh_objects()
    if not mesh_objects:
        raise RuntimeError("No mesh objects found.")

    renderer.normalize_scene(mesh_objects)
    renderer.apply_material(mesh_objects)
    camera = renderer.create_camera()
    renderer.create_light()
    renderer.setup_render()

    for view in views:
        camera.location = renderer.spherical_to_cartesian(
            view["radius"],
            math.radians(view["azimuth_deg"]),
            math.radians(view["elevation_deg"]),
        )
        renderer.point_camera_at_origin(camera)
        bpy.context.scene.render.filepath = str(Path(output_dir) / view["image"])
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
