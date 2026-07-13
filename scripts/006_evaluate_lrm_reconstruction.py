"""Evaluate zero-shot meshes against references and quantify view instability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "exp004_lrm_reconstruction" / "run_manifest.csv"
ASSETS = ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv"
OUT = ROOT / "experiments" / "exp004_lrm_reconstruction"
SAMPLES = 20_000
SEED = 20260713


def mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    result = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    result.remove_unreferenced_vertices()
    result.vertices -= result.bounding_box.centroid
    result.vertices /= max(float(np.linalg.norm(result.extents)), 1e-12)
    return result


def points(value: trimesh.Trimesh, seed: int) -> np.ndarray:
    return trimesh.sample.sample_surface(value, SAMPLES, seed=seed)[0]


def best_aligned(reference: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, float]:
    best_points, best_cost = reconstruction, float("inf")
    for quarter_turn in range(4):
        angle = quarter_turn * np.pi / 2
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        initial = reconstruction @ rotation.T
        matrix, transformed, cost = trimesh.registration.icp(
            initial, reference, max_iterations=40, reflection=False, scale=False
        )
        del matrix
        if cost < best_cost:
            best_points, best_cost = transformed, float(cost)
    return best_points, best_cost


def distances(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_to_b = cKDTree(b).query(a)[0]
    b_to_a = cKDTree(a).query(b)[0]
    threshold = 0.01
    precision = float(np.mean(a_to_b <= threshold))
    recall = float(np.mean(b_to_a <= threshold))
    return {
        "chamfer_l1": float((a_to_b.mean() + b_to_a.mean()) / 2),
        "hausdorff": float(max(a_to_b.max(), b_to_a.max())),
        "fscore_001": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def main() -> None:
    runs = pd.read_csv(RUNS)
    assets = pd.read_csv(ASSETS).set_index("asset_id")
    rows = []
    aligned_by_asset: dict[str, list[np.ndarray]] = {}
    for index, run in runs.iterrows():
        reference_mesh = mesh(ROOT / assets.loc[run.asset_id, "local_path"])
        generated_mesh = mesh(ROOT / run.mesh_path)
        reference_points = points(reference_mesh, SEED + index * 2)
        generated_points = points(generated_mesh, SEED + index * 2 + 1)
        aligned, icp_cost = best_aligned(reference_points, generated_points)
        aligned_by_asset.setdefault(run.asset_id, []).append(aligned)
        rows.append(
            {
                "asset_id": run.asset_id,
                "view": run.view,
                "reference_vertices": len(reference_mesh.vertices),
                "generated_vertices": len(generated_mesh.vertices),
                "icp_cost": icp_cost,
                **distances(aligned, reference_points),
            }
        )
    quality = pd.DataFrame(rows)
    quality.to_csv(OUT / "reference_quality.csv", index=False)
    consistency_rows = []
    for asset_id, clouds in aligned_by_asset.items():
        pair_metrics = [
            distances(clouds[left], clouds[right])["chamfer_l1"]
            for left in range(len(clouds))
            for right in range(left + 1, len(clouds))
        ]
        consistency_rows.append(
            {
                "asset_id": asset_id,
                "view_pairs": len(pair_metrics),
                "mean_pairwise_chamfer": float(np.mean(pair_metrics)),
                "max_pairwise_chamfer": float(np.max(pair_metrics)),
            }
        )
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(OUT / "view_consistency.csv", index=False)
    summary = json.loads((OUT / "summary.json").read_text())
    summary.update(
        {
            "mean_reference_chamfer": float(quality.chamfer_l1.mean()),
            "mean_reference_fscore_001": float(quality.fscore_001.mean()),
            "mean_view_pair_chamfer": float(consistency.mean_pairwise_chamfer.mean()),
            "evaluation_surface_samples": SAMPLES,
        }
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
