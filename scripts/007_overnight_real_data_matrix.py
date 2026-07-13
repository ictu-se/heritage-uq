"""Run the overnight real-data reconstruction robustness matrix with checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
TRIPOSR = ROOT / "external" / "TripoSR"
WEIGHTS = ROOT / "external" / "TripoSR_weights"
SOURCE = ROOT / "data" / "processed" / "exp004_input_views"
INPUTS = ROOT / "data" / "processed" / "exp005_robustness_inputs"
OUT = ROOT / "experiments" / "exp005_real_data_robustness"
LOG = ROOT / "logs" / "overnight_experiment.log"
RESOLUTIONS = (128, 192, 256, 320, 384, 448, 512, 640)
CONDITIONS = (
    "clean",
    "blur_low",
    "blur_high",
    "jpeg_70",
    "jpeg_35",
    "occlusion_15",
    "occlusion_30",
)
ASSET_FILTER: set[str] | None = None
MODEL_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"
MODEL_CHECKPOINT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_inputs() -> list[dict]:
    rows = []
    for source in sorted(SOURCE.rglob("view_*.png")):
        asset_id, view = source.parent.name, source.stem
        if ASSET_FILTER is not None and asset_id not in ASSET_FILTER:
            continue
        base = Image.open(source).convert("RGBA")
        for condition in CONDITIONS:
            target = INPUTS / condition / asset_id / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if condition == "clean":
                image = base.copy()
                image.save(target)
            elif condition.startswith("blur"):
                radius = 1.25 if condition == "blur_low" else 3.5
                base.filter(ImageFilter.GaussianBlur(radius=radius)).save(target)
            elif condition.startswith("jpeg"):
                temporary = target.with_suffix(".jpg")
                background = Image.new("RGB", base.size, "white")
                background.paste(base, mask=base.getchannel("A"))
                quality = 70 if condition == "jpeg_70" else 35
                background.save(temporary, quality=quality, optimize=True)
                Image.open(temporary).convert("RGBA").save(target)
                temporary.unlink()
            else:
                image = base.copy()
                draw = ImageDraw.Draw(image)
                width, height = image.size
                fraction = 0.15 if condition == "occlusion_15" else 0.30
                draw.rectangle(
                    ((1 - fraction) * width, 0.2 * height, width, 0.8 * height),
                    fill=(255, 255, 255, 255),
                )
                image.save(target)
            rows.append(
                {
                    "asset_id": asset_id,
                    "view": view,
                    "condition": condition,
                    "path": target.relative_to(ROOT).as_posix(),
                    "sha256": sha256(target),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "input_manifest.csv", index=False)
    return rows


def run_batch(resolution: int, condition: str, inputs: list[dict]) -> list[dict]:
    selected = [row for row in inputs if row["condition"] == condition]
    destination = OUT / "meshes" / f"r{resolution}" / condition
    completed_marker = destination / "_COMPLETE.json"
    if completed_marker.exists():
        return json.loads(completed_marker.read_text())
    temporary = OUT / "_working" / f"r{resolution}_{condition}"
    if temporary.exists():
        shutil.rmtree(temporary)
    paths = [ROOT / row["path"] for row in selected]
    # TripoSR retains enough per-image state that a 24-image invocation can
    # exhaust host RAM during high-resolution marching cubes.  Chunking changes
    # neither inputs nor model settings; it only bounds peak process memory.
    chunk_size = 8 if resolution >= 448 else len(paths)
    log(f"START resolution={resolution} condition={condition} inputs={len(paths)}")
    started = time.perf_counter()
    chunk_outputs: list[tuple[Path, int]] = []
    for chunk_start in range(0, len(paths), chunk_size):
        chunk_paths = paths[chunk_start : chunk_start + chunk_size]
        chunk_dir = temporary / f"chunk_{chunk_start:03d}"
        command = [sys.executable, str(TRIPOSR / "run.py"), *map(str, chunk_paths)]
        command += [
            "--output-dir",
            str(chunk_dir),
            "--pretrained-model-name-or-path",
            str(WEIGHTS),
            "--device",
            "cuda:0",
            "--mc-resolution",
            str(resolution),
        ]
        process = subprocess.run(command, cwd=TRIPOSR, text=True, capture_output=True)
        if process.returncode:
            log(
                f"FAILED resolution={resolution} condition={condition} "
                f"chunk_start={chunk_start}: {process.stderr[-3000:]}"
            )
            raise RuntimeError(process.stderr[-3000:])
        chunk_outputs.extend((chunk_dir, index) for index in range(len(chunk_paths)))
    elapsed = time.perf_counter() - started
    records = []
    for index, row in enumerate(selected):
        chunk_dir, local_index = chunk_outputs[index]
        source_mesh = chunk_dir / str(local_index) / "mesh.obj"
        target_mesh = destination / row["asset_id"] / row["view"] / "mesh.obj"
        target_mesh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_mesh, target_mesh)
        records.append(
            {
                **row,
                "resolution": resolution,
                "mesh_path": target_mesh.relative_to(ROOT).as_posix(),
                "mesh_sha256": sha256(target_mesh),
                "seconds_per_input": elapsed / len(selected),
                "status": "success",
            }
        )
    shutil.rmtree(temporary)
    completed_marker.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log(f"DONE resolution={resolution} condition={condition} seconds={elapsed:.1f}")
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs()
    records = []
    for resolution in RESOLUTIONS:
        for condition in CONDITIONS:
            records.extend(run_batch(resolution, condition, inputs))
            pd.DataFrame(records).to_csv(OUT / "run_manifest.csv", index=False)
    summary = {
        "real_source_assets": len({row["asset_id"] for row in inputs}),
        "input_images": len(inputs),
        "reconstructions": len(records),
        "resolutions": list(RESOLUTIONS),
        "conditions": list(CONDITIONS),
        "model": "stabilityai/TripoSR",
        "model_commit": MODEL_COMMIT,
        "model_checkpoint_bytes": (WEIGHTS / "model.ckpt").stat().st_size,
        "model_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "claim_boundary": "Real Smithsonian source meshes with controlled image perturbations; no cultural-authenticity claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"ALL_DONE reconstructions={len(records)}")


if __name__ == "__main__":
    main()
