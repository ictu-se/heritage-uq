import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_exp001_manifest() -> None:
    frame = pd.read_csv(ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv")
    assert len(frame) >= 3
    assert frame.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame.model_url.nunique() == len(frame)


def test_exp002_outputs() -> None:
    summary = json.loads(
        (ROOT / "experiments" / "exp002_controlled_degradation" / "summary.json").read_text()
    )
    assert summary["records"] == summary["assets"] * 3 * 5
    metrics = pd.read_csv(
        ROOT / "experiments" / "exp002_controlled_degradation" / "degradation_metrics.csv"
    )
    assert metrics.chamfer_l1.ge(0).all()
    assert metrics.fscore_001.between(0, 1).all()


def test_exp003_calibration() -> None:
    summary = json.loads(
        (ROOT / "experiments" / "exp003_ch3d_quality_calibration" / "summary.json").read_text()
    )
    assert summary["models"] == 45
    assert summary["sites"] >= 3
    assert -1 <= summary["site_disjoint_linear_spearman"] <= 1


def test_exp004_lrm_outputs() -> None:
    folder = ROOT / "experiments" / "exp004_lrm_reconstruction"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["inputs"] == 16
    assert summary["successful"] == 16
    assert summary["mean_reference_chamfer"] > 0
    assert 0 <= summary["mean_reference_fscore_001"] <= 1
    runs = pd.read_csv(folder / "run_manifest.csv")
    assert len(runs) == 16
    assert runs.status.isin(["success", "cached"]).all()


def test_full_campaign_when_finalized() -> None:
    summary_path = ROOT / "experiments" / "campaign_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["real_smithsonian_assets"] == 10
    assert summary["development_matrix"]["runs"] == 896
    assert summary["external_holdout_matrix"]["runs"] == 1344
    assert summary["factorial_reconstructions"] == 2240
    assert summary["pilot_reconstructions"] == 16
    assert summary["total_generated_reconstructions"] == 2256
    assert summary["ch3d_models"] == 45
    assert summary["hallucination_localization"]["localized_meshes"] == 144
    calibration = summary["cross_dataset_calibration"]
    assert calibration["development"]["records"] == 768
    assert calibration["external_holdout"]["records"] == 1152
    assert 0 <= calibration["external_holdout"]["brier"] <= 1
    assert 0 <= calibration["external_holdout"]["auroc"] <= 1
    assert (ROOT / "FINAL_EXPERIMENT_REPORT.md").is_file()
    assert "2,240" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_analysis_uses_literal_view_column() -> None:
    source = (ROOT / "scripts" / "008_analyze_overnight_matrix.py").read_text(
        encoding="utf-8"
    )
    assert "row.view" not in source
    assert 'row["view"]' in source
