"""Localize unsupported generated surface against real Smithsonian reference meshes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "exp005_real_data_robustness" / "run_manifest.csv"
ASSETS = ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv"
OUT = ROOT / "experiments" / "exp007_hallucination_localization"
FIG = ROOT / "figures"
CONDITIONS = {"clean", "jpeg_35", "occlusion_30"}
RESOLUTIONS = {128, 256, 512}
SAMPLES = 8_000
REFERENCE_SUPPORT_SAMPLES = 50_000
THRESHOLDS = (0.01, 0.02, 0.05)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    mesh.remove_unreferenced_vertices()
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= max(float(np.linalg.norm(mesh.extents)), 1e-12)
    return mesh


def aligned_vertices(
    reference_points: np.ndarray, generated: trimesh.Trimesh, seed: int
) -> np.ndarray:
    generated_points = trimesh.sample.sample_surface(generated, SAMPLES, seed=seed + 1)[0]
    best_cost = float("inf")
    best_transform = np.eye(4)
    for quarter_turn in range(4):
        angle = quarter_turn * np.pi / 2
        rotation = np.eye(4)
        rotation[:3, :3] = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        initial = trimesh.transform_points(generated_points, rotation)
        matrix, _, cost = trimesh.registration.icp(
            initial, reference_points, max_iterations=30, reflection=False, scale=False
        )
        if cost < best_cost:
            best_cost = float(cost)
            best_transform = matrix @ rotation
    return trimesh.transform_points(generated.vertices, best_transform)


def color_map(distances: np.ndarray) -> np.ndarray:
    normalized = np.clip(distances / 0.05, 0, 1)
    colors = np.zeros((len(distances), 4), dtype=np.uint8)
    colors[:, 0] = (255 * normalized).astype(np.uint8)
    colors[:, 1] = (255 * (1 - normalized)).astype(np.uint8)
    colors[:, 2] = 40
    colors[:, 3] = 255
    return colors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RUNS)
    runs = runs[runs.condition.isin(CONDITIONS) & runs.resolution.isin(RESOLUTIONS)].copy()
    assets = pd.read_csv(ASSETS).set_index("asset_id")
    rows = []
    candidates = []
    reference_cache: dict[str, tuple[np.ndarray, cKDTree]] = {}
    for index, run in runs.reset_index(drop=True).iterrows():
        if run.asset_id not in reference_cache:
            reference = load_mesh(ROOT / assets.loc[run.asset_id, "local_path"])
            support = trimesh.sample.sample_surface(
                reference,
                REFERENCE_SUPPORT_SAMPLES,
                seed=20260713 + len(reference_cache),
            )[0]
            reference_cache[run.asset_id] = support, cKDTree(support)
        support, support_tree = reference_cache[run.asset_id]
        generated = load_mesh(ROOT / run.mesh_path)
        vertices = aligned_vertices(support, generated, 20260713 + index * 2)
        distances = support_tree.query(vertices)[0]
        record = {
            "asset_id": run.asset_id,
            "view": run["view"],
            "condition": run.condition,
            "resolution": int(run.resolution),
            "vertices": len(vertices),
            "mean_distance": float(distances.mean()),
            "p95_distance": float(np.quantile(distances, 0.95)),
            "max_distance": float(distances.max()),
        }
        for threshold in THRESHOLDS:
            record[f"unsupported_rate_{threshold:.2f}"] = float(np.mean(distances > threshold))
        rows.append(record)
        candidates.append((record["unsupported_rate_0.02"], record, generated, vertices, distances))
        candidates.sort(reverse=True, key=lambda item: item[0])
        if len(candidates) > 8:
            candidates.pop()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "localization_metrics.csv", index=False)
    aggregate = frame.groupby(["resolution", "condition"], as_index=False).agg(
        unsupported_rate_002=("unsupported_rate_0.02", "mean"),
        mean_distance=("mean_distance", "mean"),
        p95_distance=("p95_distance", "mean"),
    )
    aggregate.to_csv(OUT / "aggregate_localization.csv", index=False)

    heatmap_dir = OUT / "representative_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for _, record, mesh, vertices, distances in candidates:
        colored = mesh.copy()
        colored.vertices = vertices
        colored.visual.vertex_colors = color_map(distances)
        name = f"{record['asset_id']}_{record['view']}_{record['condition']}_r{record['resolution']}.ply"
        colored.export(heatmap_dir / name)

    summary = {
        "localized_meshes": len(frame),
        "real_reference_assets": int(frame.asset_id.nunique()),
        "conditions": sorted(frame.condition.unique()),
        "resolutions": sorted(int(value) for value in frame.resolution.unique()),
        "mean_unsupported_rate_002": float(frame["unsupported_rate_0.02"].mean()),
        "worst_unsupported_rate_002": float(frame["unsupported_rate_0.02"].max()),
        "representative_colored_ply": 8,
        "reference_support_samples_per_asset": REFERENCE_SUPPORT_SAMPLES,
        "definition": "Generated vertices farther than a normalized distance threshold from a dense sample of the aligned real reference surface.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition, group in aggregate.groupby("condition"):
        ax.plot(group.resolution, group.unsupported_rate_002, marker="o", label=condition)
    ax.set(
        xlabel="Marching-cubes resolution",
        ylabel="Unsupported vertex rate (>0.02)",
        title="Hallucinated-surface localization",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "exp007_hallucination_localization.pdf", bbox_inches="tight")
    fig.savefig(FIG / "exp007_hallucination_localization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
