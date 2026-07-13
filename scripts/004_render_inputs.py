"""Render deterministic RGBA input views from Smithsonian GLB references in Blender."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv"
OUT = ROOT / "data" / "processed" / "exp004_input_views"
AZIMUTHS = (0, 90, 180, 270)


def reset() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (low + high) / 2
    return center, max((high - low).length, 1e-6)


def point_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def render_asset(asset_id: str, source: Path, legacy_clipping: bool = False) -> None:
    reset()
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    center, diagonal = bounds(meshes)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = diagonal * 1.05
    if not legacy_clipping:
        camera.data.clip_start = max(diagonal * 1e-4, 1e-5)
        camera.data.clip_end = diagonal * 10
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(4, -4, 6))
    bpy.context.object.data.energy = 1200
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5
    bpy.ops.object.light_add(type="AREA", location=(-4, 2, 3))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.size = 4
    asset_dir = OUT / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    for azimuth in AZIMUTHS:
        angle = math.radians(azimuth)
        camera.location = center + Vector((math.sin(angle), -math.cos(angle), 0.2)) * diagonal * 1.8
        point_at(camera, center)
        bpy.context.scene.render.filepath = str(asset_dir / f"view_{azimuth:03d}.png")
        bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--legacy-clipping", action="store_true")
    script_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(script_args)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    for row in rows:
        render_asset(
            row["asset_id"], ROOT / row["local_path"], legacy_clipping=args.legacy_clipping
        )


if __name__ == "__main__":
    main()
