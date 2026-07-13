"""Render a qualitative Smithsonian reference/reconstruction case in Blender."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
ASSET = "SI_1f1ec6af-193f-4969-9185-dc5dd587239d"
REFERENCE = ROOT / "data" / "raw" / "smithsonian" / f"{ASSET}.glb"
CLEAN = (
    ROOT
    / "experiments"
    / "exp005_real_data_robustness"
    / "meshes"
    / "r512"
    / "clean"
    / ASSET
    / "view_270"
    / "mesh.obj"
)
OCCLUDED = (
    ROOT
    / "experiments"
    / "exp005_real_data_robustness"
    / "meshes"
    / "r512"
    / "occlusion_30"
    / ASSET
    / "view_270"
    / "mesh.obj"
)
OUT = ROOT / "figures" / "qualitative_case"

GALLERY_CASES = (
    (
        "pottery",
        "SI_fc2bca6b-1f44-4d2c-a1b9-713a33ee0596",
        "exp005_real_data_robustness",
        "blur_high",
        225,
    ),
    (
        "crab",
        "SI_451bcb49-01f9-4030-9d89-6f976b64572b",
        "exp005_real_data_robustness",
        "blur_high",
        225,
    ),
    (
        "helmet",
        "SI_69481624-3312-4f02-be4c-e8bb10ee84c4",
        "exp008_external_holdout",
        "occlusion_15",
        225,
    ),
    (
        "statue",
        "SI_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f",
        "exp008_external_holdout",
        "occlusion_15",
        225,
    ),
)

ATLAS_CASES = (
    ("pottery", "SI_fc2bca6b-1f44-4d2c-a1b9-713a33ee0596", 225),
    ("fossil", "SI_1f1ec6af-193f-4969-9185-dc5dd587239d", 225),
    ("mandible", "SI_b1a04b98-c12f-4123-8093-af32c20ce5f3", 225),
    ("crab", "SI_451bcb49-01f9-4030-9d89-6f976b64572b", 225),
    ("skull", "SI_6bd320bc-25b0-4ef8-9654-a6bdeea28636", 225),
    ("helmet", "SI_69481624-3312-4f02-be4c-e8bb10ee84c4", 225),
    ("statue", "SI_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f", 225),
    ("shell", "SI_c29a9ede-c224-46c9-8791-69ff36c23828", 225),
    ("plow", "SI_d3ddac58-b467-40a4-93e6-8664b6a33997", 225),
    ("insect", "SI_4cc2228f-6772-4878-971d-04e4893c94cd", 225),
)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def import_geometry(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    if path.suffix.lower() == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        bpy.ops.wm.obj_import(filepath=str(path))
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def normalize(objects: list[bpy.types.Object]) -> None:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    lower = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    upper = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (lower + upper) / 2
    scale = 2.2 / max((upper - lower))
    for obj in objects:
        obj.location -= center
        obj.scale *= scale
    bpy.context.view_layer.update()


def apply_gray_material(objects: list[bpy.types.Object], color: tuple[float, float, float, float]) -> None:
    material = bpy.data.materials.new("Reconstruction material")
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.72
    bsdf.inputs["Metallic"].default_value = 0.0
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_scene(camera_azimuth: float) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 900
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.world.color = (0.96, 0.96, 0.96)

    angle = math.radians(camera_azimuth)
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (3.6 * math.cos(angle), 3.6 * math.sin(angle), 2.5)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.25
    look_at(camera, Vector((0, 0, 0)))
    scene.camera = camera

    for name, location, energy, size in (
        ("Key", (4.0, -3.0, 5.0), 1050, 4.0),
        ("Fill", (-4.0, -1.0, 3.0), 700, 3.5),
        ("Rim", (1.0, 4.0, 4.0), 850, 3.0),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        look_at(light, Vector((0, 0, 0)))
        bpy.context.collection.objects.link(light)


def render(path: Path, output_name: str, azimuth: float, gray: bool) -> None:
    clear_scene()
    objects = import_geometry(path)
    normalize(objects)
    if gray:
        apply_gray_material(objects, (0.48, 0.64, 0.78, 1.0))
    setup_scene(azimuth)
    bpy.context.scene.render.filepath = str(OUT / output_name)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render(REFERENCE, "reference_3d.png", 225, gray=False)
    render(CLEAN, "reconstruction_clean.png", 45, gray=True)
    render(OCCLUDED, "reconstruction_occluded.png", 225, gray=True)
    render(OCCLUDED, "reconstruction_occluded_back.png", 45, gray=True)
    for label, asset_id, experiment, condition, azimuth in GALLERY_CASES:
        reference = ROOT / "data" / "raw" / "smithsonian" / f"{asset_id}.glb"
        reconstruction = (
            ROOT
            / "experiments"
            / experiment
            / "meshes"
            / "r512"
            / condition
            / asset_id
            / "view_000"
            / "mesh.obj"
        )
        render(reference, f"gallery_{label}_reference.png", azimuth, gray=False)
        render(reconstruction, f"gallery_{label}_reconstruction.png", 45, gray=True)
    for label, asset_id, azimuth in ATLAS_CASES:
        reference = ROOT / "data" / "raw" / "smithsonian" / f"{asset_id}.glb"
        render(reference, f"atlas_{label}.png", azimuth, gray=False)


if __name__ == "__main__":
    main()
