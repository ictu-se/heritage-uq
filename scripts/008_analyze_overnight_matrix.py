"""Analyze structural quality and perturbation sensitivity of the overnight matrix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "exp005_real_data_robustness"
OUT = ROOT / "experiments" / "exp006_uncertainty_analysis"
FIG = ROOT / "figures"
FIGURE_STEM = "exp006_robustness_matrix"
SAMPLES = 5_000


def load_points(path: Path, seed: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    loaded = trimesh.load(path, force="scene")
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    mesh.remove_unreferenced_vertices()
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= max(float(np.linalg.norm(mesh.extents)), 1e-12)
    return mesh, trimesh.sample.sample_surface(mesh, SAMPLES, seed=seed)[0]


def chamfer(left: np.ndarray, right: np.ndarray) -> float:
    return float((cKDTree(right).query(left)[0].mean() + cKDTree(left).query(right)[0].mean()) / 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(EXP / "run_manifest.csv")
    cache: dict[str, np.ndarray] = {}
    structures = []
    for index, row in runs.iterrows():
        mesh, cloud = load_points(ROOT / row["mesh_path"], 20260713 + index)
        cache[row["mesh_path"]] = cloud
        structures.append(
            {
                "asset_id": row["asset_id"],
                "view": row["view"],
                "condition": row["condition"],
                "resolution": row["resolution"],
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "components": int(mesh.body_count),
                "watertight": int(mesh.is_watertight),
                "surface_area": float(mesh.area),
                "volume_abs": float(abs(mesh.volume)) if mesh.is_volume else np.nan,
            }
        )
    structure_frame = pd.DataFrame(structures)
    structure_frame.to_csv(OUT / "mesh_structure.csv", index=False)

    comparisons = []
    keys = ["asset_id", "view", "resolution"]
    clean = runs[runs.condition == "clean"].set_index(keys)
    for _, row in runs[runs.condition != "clean"].iterrows():
        clean_row = clean.loc[(row["asset_id"], row["view"], row["resolution"])]
        clean_cloud = cache[clean_row["mesh_path"]]
        perturbed_cloud = cache[row["mesh_path"]]
        comparisons.append(
            {
                "asset_id": row["asset_id"],
                "view": row["view"],
                "resolution": row["resolution"],
                "condition": row["condition"],
                "clean_to_perturbed_chamfer": chamfer(clean_cloud, perturbed_cloud),
            }
        )
    sensitivity = pd.DataFrame(comparisons)
    sensitivity.to_csv(OUT / "perturbation_sensitivity.csv", index=False)
    aggregate = sensitivity.groupby(["resolution", "condition"], as_index=False).agg(
        mean_chamfer=("clean_to_perturbed_chamfer", "mean"),
        std_chamfer=("clean_to_perturbed_chamfer", "std"),
    )
    aggregate.to_csv(OUT / "aggregate_sensitivity.csv", index=False)
    summary = {
        "meshes_analyzed": len(runs),
        "pairwise_clean_perturbed_comparisons": len(sensitivity),
        "watertight_rate": float(structure_frame.watertight.mean()),
        "mean_components": float(structure_frame.components.mean()),
        "most_disruptive_condition": aggregate.loc[aggregate.mean_chamfer.idxmax(), "condition"],
        "max_aggregate_chamfer": float(aggregate.mean_chamfer.max()),
        "claim_boundary": "Cross-condition reconstruction instability on real-source renders; not cultural authenticity.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition, group in aggregate.groupby("condition"):
        ax.errorbar(
            group.resolution,
            group.mean_chamfer,
            yerr=group.std_chamfer,
            marker="o",
            label=condition,
        )
    ax.set(
        xlabel="Marching-cubes resolution",
        ylabel="Clean-to-perturbed Chamfer",
        title="Real-data reconstruction instability",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{FIGURE_STEM}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{FIGURE_STEM}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
