"""Audit the full real-data campaign and write a consolidated machine-readable report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
REPORT = ROOT / "FINAL_EXPERIMENT_REPORT.md"
SUMMARY = EXPERIMENTS / "campaign_summary.json"
README = ROOT / "README.md"
NOTE = ROOT / "NOTE.md"
CONDITIONS = {
    "clean",
    "blur_low",
    "blur_high",
    "jpeg_70",
    "jpeg_35",
    "occlusion_15",
    "occlusion_30",
}
RESOLUTIONS = {128, 192, 256, 320, 384, 448, 512, 640}
MODEL_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"
MODEL_CHECKPOINT_SHA256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_matrix(folder: str, expected_assets: int) -> tuple[dict, set[str]]:
    base = EXPERIMENTS / folder
    frame = pd.read_csv(base / "run_manifest.csv")
    expected = expected_assets * 4 * len(CONDITIONS) * len(RESOLUTIONS)
    keys = ["asset_id", "view", "condition", "resolution"]
    assert len(frame) == expected, (folder, len(frame), expected)
    assert frame[keys].drop_duplicates().shape[0] == expected
    assert frame.asset_id.nunique() == expected_assets
    assert set(frame.condition) == CONDITIONS
    assert set(frame.resolution.astype(int)) == RESOLUTIONS
    assert frame.view.nunique() == 4
    assert frame.status.isin(["success", "cached"]).all()
    assert frame.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame.mesh_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    missing = [path for path in frame.mesh_path if not (ROOT / path).is_file()]
    assert not missing, f"{len(missing)} missing meshes in {folder}"
    input_records = frame[["path", "sha256"]].drop_duplicates()
    assert len(input_records) == expected_assets * 4 * len(CONDITIONS)
    for row in input_records.itertuples(index=False):
        assert sha256_file(ROOT / row.path) == row.sha256
    for row in frame[["mesh_path", "mesh_sha256"]].itertuples(index=False):
        assert sha256_file(ROOT / row.mesh_path) == row.mesh_sha256
    return (
        {
            "runs": len(frame),
            "assets": int(frame.asset_id.nunique()),
            "views": int(frame.view.nunique()),
            "conditions": len(CONDITIONS),
            "resolutions": len(RESOLUTIONS),
        },
        set(frame.asset_id),
    )


def main() -> None:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "TripoSR"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert actual_commit == MODEL_COMMIT
    checkpoint = ROOT / "external" / "TripoSR_weights" / "model.ckpt"
    assert checkpoint.stat().st_size == 1_677_246_742
    assert sha256_file(checkpoint) == MODEL_CHECKPOINT_SHA256
    datasets = read_json(ROOT / "config" / "datasets.json")
    ch3d_source = datasets["ch3d_reco"]
    ch3d_source_path = ROOT / ch3d_source["evaluation_file"]
    assert ch3d_source_path.stat().st_size == ch3d_source["evaluation_file_bytes"]
    assert sha256_file(ch3d_source_path) == ch3d_source["evaluation_file_sha256"]
    assets = pd.read_csv(EXPERIMENTS / "exp001_dataset_acquisition" / "asset_manifest.csv")
    assert len(assets) == 10
    assert assets.model_url.nunique() == 10
    assert assets.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    for row in assets.itertuples():
        path = ROOT / row.local_path
        assert path.is_file()
        digest = sha256_file(path)
        assert digest == row.sha256

    development, development_assets = audit_matrix("exp005_real_data_robustness", 4)
    holdout, holdout_assets = audit_matrix("exp008_external_holdout", 6)
    for folder in ("exp005_real_data_robustness", "exp008_external_holdout"):
        matrix_summary = read_json(EXPERIMENTS / folder / "summary.json")
        matrix_runs = pd.read_csv(EXPERIMENTS / folder / "run_manifest.csv")
        matrix_inputs = pd.read_csv(EXPERIMENTS / folder / "input_manifest.csv")
        assert matrix_summary["real_source_assets"] == matrix_runs.asset_id.nunique()
        assert matrix_summary["input_images"] == len(matrix_inputs)
        assert matrix_summary["reconstructions"] == len(matrix_runs)
        assert matrix_summary["model_commit"] == MODEL_COMMIT
        assert matrix_summary["model_checkpoint_sha256"] == MODEL_CHECKPOINT_SHA256
    assert development_assets.isdisjoint(holdout_assets)
    assert development_assets | holdout_assets == set(assets.asset_id)
    development_uncertainty = read_json(
        EXPERIMENTS / "exp006_uncertainty_analysis" / "summary.json"
    )
    holdout_uncertainty = read_json(
        EXPERIMENTS / "exp009_holdout_uncertainty" / "summary.json"
    )
    localization = read_json(
        EXPERIMENTS / "exp007_hallucination_localization" / "summary.json"
    )
    ch3d = read_json(EXPERIMENTS / "exp003_ch3d_quality_calibration" / "summary.json")
    pilot = read_json(EXPERIMENTS / "exp004_lrm_reconstruction" / "summary.json")
    cross_dataset = read_json(
        EXPERIMENTS / "exp010_cross_dataset_calibration" / "summary.json"
    )
    assert development_uncertainty["meshes_analyzed"] == 896
    assert development_uncertainty["pairwise_clean_perturbed_comparisons"] == 768
    assert holdout_uncertainty["meshes_analyzed"] == 1344
    assert holdout_uncertainty["pairwise_clean_perturbed_comparisons"] == 1152
    assert localization["localized_meshes"] == 144
    assert localization["real_reference_assets"] == 4
    assert localization["representative_colored_ply"] == 8
    heatmaps = list(
        (EXPERIMENTS / "exp007_hallucination_localization" / "representative_heatmaps").glob(
            "*.ply"
        )
    )
    assert len(heatmaps) == 8 and all(path.stat().st_size > 0 for path in heatmaps)
    required_figures = (
        "exp002_degradation_response.pdf",
        "exp003_ch3d_calibration.pdf",
        "exp006_robustness_matrix.pdf",
        "exp007_hallucination_localization.pdf",
        "exp009_holdout_robustness_matrix.pdf",
        "exp010_cross_dataset_calibration.pdf",
    )
    for figure in required_figures:
        path = ROOT / "figures" / figure
        assert path.is_file() and path.stat().st_size > 0
    assert ch3d["models"] == 45 and ch3d["sites"] == 5
    assert pilot["inputs"] == 16 and pilot["successful"] == 16
    assert cross_dataset["development"]["records"] == 768
    assert cross_dataset["external_holdout"]["records"] == 1152

    result = {
        "status": "complete",
        "real_smithsonian_assets": 10,
        "smithsonian_bytes": int(sum((ROOT / path).stat().st_size for path in assets.local_path)),
        "triposr_source_commit": MODEL_COMMIT,
        "triposr_checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "development_matrix": development,
        "external_holdout_matrix": holdout,
        "factorial_reconstructions": development["runs"] + holdout["runs"],
        "pilot_reconstructions": pilot["successful"],
        "total_generated_reconstructions": (
            development["runs"] + holdout["runs"] + pilot["successful"]
        ),
        "ch3d_models": ch3d["models"],
        "ch3d_sites": ch3d["sites"],
        "ch3d_interaction_rows": ch3d["interaction_log_rows"],
        "ch3d_source_sha256": ch3d_source["evaluation_file_sha256"],
        "ch3d_site_disjoint_spearman": ch3d["site_disjoint_linear_spearman"],
        "development_uncertainty": development_uncertainty,
        "holdout_uncertainty": holdout_uncertainty,
        "hallucination_localization": localization,
        "cross_dataset_calibration": cross_dataset,
    }
    SUMMARY.write_text(json.dumps(result, indent=2), encoding="utf-8")
    REPORT.write_text(
        "# Final experiment report\n\n"
        "The campaign completed on real Smithsonian Open Access 3D assets and the "
        "published CH3D-Reco evaluation data.\n\n"
        f"- Smithsonian assets: **10** ({result['smithsonian_bytes']:,} bytes)\n"
        f"- Factorial reconstruction runs: **{result['factorial_reconstructions']:,}** "
        "(896 development + 1,344 external holdout)\n"
        f"- Total generated reconstructions including the 16-run pilot: "
        f"**{result['total_generated_reconstructions']:,}**\n"
        "- Factorial design: 4 views, 7 input conditions, and 8 extraction resolutions\n"
        f"- CH3D-Reco: **{ch3d['models']} models**, **{ch3d['sites']} sites**, "
        f"**{ch3d['interaction_log_rows']:,} interaction rows**\n"
        f"- CH3D site-disjoint calibration: Spearman "
        f"**{ch3d['site_disjoint_linear_spearman']:.3f}**\n"
        f"- Development most disruptive condition: "
        f"**{development_uncertainty['most_disruptive_condition']}**\n"
        f"- Holdout most disruptive condition: "
        f"**{holdout_uncertainty['most_disruptive_condition']}**\n"
        f"- Localized meshes: **{localization['localized_meshes']}**, mean unsupported "
        f"vertex rate at 0.02: **{localization['mean_unsupported_rate_002']:.3f}**\n\n"
        f"- Category-holdout calibration: AUROC "
        f"**{cross_dataset['external_holdout']['auroc']:.3f}**, Brier "
        f"**{cross_dataset['external_holdout']['brier']:.3f}**, ECE "
        f"**{cross_dataset['external_holdout']['ece_10_bin']:.3f}**\n\n"
        "All counts above are enforced by `scripts/014_finalize_campaign.py`; a missing "
        "factor combination, mesh, provenance hash, analysis output, or localization "
        "artifact causes finalization to fail. Results measure geometric robustness and "
        "calibration, not cultural authenticity or human judgment.\n",
        encoding="utf-8",
    )
    README.write_text(
        "# 07 - Uncertainty-Aware 3D Heritage Reconstruction\n\n"
        "Reproducible experiments for uncertainty-aware geometric quality assessment "
        "of single-image 3D cultural-heritage reconstruction.\n\n"
        "## Completed real-data campaign\n\n"
        f"The audited factorial campaign contains **{result['factorial_reconstructions']:,} "
        "TripoSR reconstructions** from 10 Smithsonian Open Access 3D assets, plus a "
        "16-run pilot (2,256 generated reconstructions in total). Development and "
        "category-holdout sets are disjoint. Each asset uses four rendered views, seven "
        "controlled input conditions, and eight extraction resolutions. The independent "
        "CH3D-Reco analysis covers 45 models from five sites and 182,023 published "
        "interaction-log rows.\n\n"
        "Completed experiments:\n\n"
        "- `exp001`: Smithsonian acquisition and cryptographic provenance\n"
        "- `exp002`: controlled mesh degradation\n"
        "- `exp003`: CH3D-Reco MOS calibration with site-disjoint validation\n"
        "- `exp004`: initial zero-shot multiview reconstruction\n"
        "- `exp005`/`exp008`: development and external-holdout robustness matrices\n"
        "- `exp006`/`exp009`: reconstruction uncertainty and perturbation sensitivity\n"
        "- `exp007`: unsupported-surface localization and colored PLY heatmaps\n"
        "- `exp010`: development-to-category-holdout probability calibration\n\n"
        "See `FINAL_EXPERIMENT_REPORT.md` and `experiments/campaign_summary.json` for "
        "audited results. Run `python scripts/014_finalize_campaign.py` to repeat the "
        "integrity audit and `python -m pytest -q` for regression tests.\n\n"
        "## Claim boundary\n\n"
        "These experiments measure geometric robustness, instability, unsupported "
        "surface, and objective calibration. They do not establish cultural authenticity "
        "and introduce no new human-subject evaluation.\n",
        encoding="utf-8",
    )
    note = NOTE.read_text(encoding="utf-8") if NOTE.exists() else "# Experiment log\n"
    marker = "## 2026-07-13 — Full campaign completed"
    if marker not in note:
        note += (
            f"\n{marker}\n\n"
            f"Final integrity audit passed for {result['factorial_reconstructions']:,} "
            "factorial reconstructions plus the 16-run pilot, 10 real Smithsonian "
            "assets, the CH3D-Reco benchmark, "
            "uncertainty analysis, category-holdout calibration, and hallucination "
            "localization. Ruff and pytest are executed by the final watcher after this "
            "report is generated.\n"
        )
        NOTE.write_text(note, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
