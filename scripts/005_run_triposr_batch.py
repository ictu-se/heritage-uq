"""Run the official TripoSR implementation on every deterministic input view."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
TRIPOSR = ROOT / "external" / "TripoSR"
WEIGHTS = ROOT / "external" / "TripoSR_weights"
INPUTS = ROOT / "data" / "processed" / "exp004_input_views"
OUT = ROOT / "experiments" / "exp004_lrm_reconstruction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    images = sorted(INPUTS.rglob("view_*.png"))
    if args.limit:
        images = images[: args.limit]
    rows = []
    pending = []
    for image in images:
        asset_id = image.parent.name
        view = image.stem
        target = OUT / "meshes" / asset_id / view
        mesh = target / "0" / "mesh.obj"
        status = "cached"
        if not mesh.exists():
            status = "pending"
            pending.append((image, mesh))
        rows.append(
            {
                "asset_id": asset_id,
                "view": view,
                "input_path": image.relative_to(ROOT).as_posix(),
                "mesh_path": mesh.relative_to(ROOT).as_posix(),
                "status": status,
                "seconds": 0.0,
                "mc_resolution": args.resolution,
                "model": "stabilityai/TripoSR",
                "model_commit": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "error": "",
            }
        )
    if pending:
        temporary = OUT / "_batch_output"
        if temporary.exists():
            shutil.rmtree(temporary)
        command = [sys.executable, str(TRIPOSR / "run.py"), *[str(item[0]) for item in pending]]
        command += [
            "--output-dir",
            str(temporary),
            "--pretrained-model-name-or-path",
            str(WEIGHTS),
            "--device",
            "cuda:0",
            "--mc-resolution",
            str(args.resolution),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=TRIPOSR, text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        for index, (image, destination) in enumerate(pending):
            source = temporary / str(index) / "mesh.obj"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            for row in rows:
                if row["input_path"] == image.relative_to(ROOT).as_posix():
                    row["status"] = "success"
                    row["seconds"] = elapsed / len(pending)
        shutil.rmtree(temporary)
    pd.DataFrame(rows).to_csv(OUT / "run_manifest.csv", index=False)
    summary = {
        "inputs": len(images),
        "successful": sum(row["status"] in {"success", "cached"} for row in rows),
        "model": "stabilityai/TripoSR",
        "views_per_asset": 4,
        "mc_resolution": args.resolution,
        "claim_boundary": "Zero-shot single-view technical reconstruction; no cultural-authenticity claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
