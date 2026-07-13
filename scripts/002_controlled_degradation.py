"""Controlled geometric-degradation baseline on Smithsonian Open Access meshes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

plt.rcParams.update(
    {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 12}
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv"
OUT = ROOT / "experiments" / "exp002_controlled_degradation"
FIG = ROOT / "figures"
SEVERITIES = (0.02, 0.05, 0.10, 0.20, 0.35)
KINDS = ("vertex_noise", "face_dropout", "quantization")
SAMPLES = 12_000
SEED = 20260712


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    else:
        mesh = loaded
    mesh.remove_unreferenced_vertices()
    return mesh


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    result.vertices -= result.bounding_box.centroid
    scale = float(np.linalg.norm(result.extents))
    result.vertices /= max(scale, 1e-12)
    return result


def degrade(
    mesh: trimesh.Trimesh, kind: str, severity: float, rng: np.random.Generator
) -> trimesh.Trimesh:
    result = mesh.copy()
    if kind == "vertex_noise":
        result.vertices += rng.normal(0, severity * 0.02, result.vertices.shape)
    elif kind == "face_dropout":
        keep = rng.random(len(result.faces)) >= severity * 0.65
        result.update_faces(keep)
        result.remove_unreferenced_vertices()
    elif kind == "quantization":
        step = max(severity * 0.025, 1e-6)
        result.vertices = np.round(result.vertices / step) * step
    return result


def metrics(reference: trimesh.Trimesh, test: trimesh.Trimesh, seed: int) -> dict[str, float]:
    ref_points, ref_faces = trimesh.sample.sample_surface(reference, SAMPLES, seed=seed)
    test_points, test_faces = trimesh.sample.sample_surface(test, SAMPLES, seed=seed + 1)
    ref_tree, test_tree = cKDTree(ref_points), cKDTree(test_points)
    test_to_ref, test_index = ref_tree.query(test_points)
    ref_to_test, _ = test_tree.query(ref_points)
    threshold = 0.01
    precision = float(np.mean(test_to_ref <= threshold))
    recall = float(np.mean(ref_to_test <= threshold))
    fscore = 2 * precision * recall / max(precision + recall, 1e-12)
    ref_normals = reference.face_normals[ref_faces][test_index]
    test_normals = test.face_normals[test_faces]
    normal_consistency = float(np.mean(np.abs(np.sum(ref_normals * test_normals, axis=1))))
    return {
        "chamfer_l1": float((test_to_ref.mean() + ref_to_test.mean()) / 2),
        "hausdorff": float(max(test_to_ref.max(), ref_to_test.max())),
        "fscore_001": fscore,
        "normal_consistency": normal_consistency,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows = []
    for asset_index, asset in manifest.iterrows():
        reference = normalize(load_mesh(ROOT / asset.local_path))
        for kind_index, kind in enumerate(KINDS):
            for severity in SEVERITIES:
                seed = SEED + asset_index * 100 + kind_index * 10 + int(severity * 100)
                degraded = degrade(reference, kind, severity, np.random.default_rng(seed))
                rows.append(
                    {
                        "asset_id": asset.asset_id,
                        "kind": kind,
                        "severity": severity,
                        "seed": seed,
                        "reference_vertices": len(reference.vertices),
                        "reference_faces": len(reference.faces),
                        "test_vertices": len(degraded.vertices),
                        "test_faces": len(degraded.faces),
                        **metrics(reference, degraded, seed),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "degradation_metrics.csv", index=False)
    monotonicity = []
    for kind in KINDS:
        grouped = frame[frame.kind == kind].groupby("severity").chamfer_l1.mean()
        monotonicity.append(
            {
                "kind": kind,
                "spearman_severity_chamfer": float(
                    spearmanr(grouped.index, grouped.values).statistic
                ),
                "violations": int(np.sum(np.diff(grouped.values) < 0)),
            }
        )
    pd.DataFrame(monotonicity).to_csv(OUT / "monotonicity.csv", index=False)
    summary = {
        "assets": int(frame.asset_id.nunique()),
        "records": len(frame),
        "degradation_kinds": list(KINDS),
        "severity_levels": list(SEVERITIES),
        "surface_samples_per_comparison": SAMPLES,
        "seed": SEED,
        "claim_boundary": "Controlled geometry baseline; no generative-model or cultural-authenticity claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    means = frame.groupby(["kind", "severity"], as_index=False).chamfer_l1.mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for kind, group in means.groupby("kind"):
        ax.plot(group.severity, group.chamfer_l1, marker="o", label=kind.replace("_", " "))
    ax.set(
        xlabel="Controlled severity", ylabel="Symmetric Chamfer-L1", title="Degradation response"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "exp002_degradation_response.pdf", bbox_inches="tight")
    fig.savefig(FIG / "exp002_degradation_response.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
