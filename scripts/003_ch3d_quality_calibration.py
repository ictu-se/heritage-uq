"""Calibrate objective quality signals against published CH3D-Reco MOS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "ch3d_reco" / "Evaluation_results.xlsx"
OUT = ROOT / "experiments" / "exp003_ch3d_quality_calibration"
FIG = ROOT / "figures"
FEATURES = [
    "SSIM_mean",
    "PSNR_mean",
    "LPIPS_Alex_mean",
    "BRISQE_mean",
    "NIQE_mean",
    "L2_mean",
    "Haussdorf_mean",
    "FID_mean",
    "Number of Images",
    "Number of Triangles",
    "Texture Resolution",
]


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(SOURCE, sheet_name="MOS")
    data["site"] = data.DistortedModel.str.extract(r"^(S\d+)")
    correlations = []
    for feature in FEATURES:
        rho, p_value = spearmanr(data[feature], data.MOS)
        correlations.append({"feature": feature, "spearman_mos": rho, "p_value": p_value})
    corr = pd.DataFrame(correlations).sort_values("spearman_mos", key=abs, ascending=False)
    corr.to_csv(OUT / "metric_correlations.csv", index=False)

    predictions = []
    for site in sorted(data.site.unique()):
        train = data[data.site != site]
        test = data[data.site == site]
        x_train, x_test = standardize(
            train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float)
        )
        design_train = np.column_stack([np.ones(len(train)), x_train])
        design_test = np.column_stack([np.ones(len(test)), x_test])
        coefficients = np.linalg.lstsq(design_train, train.MOS.to_numpy(float), rcond=None)[0]
        predicted = design_test @ coefficients
        for model, observed, estimate in zip(test.DistortedModel, test.MOS, predicted, strict=True):
            predictions.append(
                {"site": site, "model": model, "mos": observed, "predicted_mos": estimate}
            )
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(OUT / "site_disjoint_predictions.csv", index=False)
    rho = float(spearmanr(prediction_frame.mos, prediction_frame.predicted_mos).statistic)
    mae = float(np.mean(np.abs(prediction_frame.mos - prediction_frame.predicted_mos)))
    summary = {
        "models": len(data),
        "sites": int(data.site.nunique()),
        "interaction_log_rows": int(
            len(pd.read_excel(SOURCE, sheet_name="interaction_logs", usecols=[0]))
        ),
        "best_single_metric": corr.iloc[0].feature,
        "best_single_metric_spearman": float(corr.iloc[0].spearman_mos),
        "site_disjoint_linear_spearman": rho,
        "site_disjoint_linear_mae": mae,
        "claim_boundary": "Analysis of published CH3D-Reco ratings; no new human study.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    top = corr.head(8).sort_values("spearman_mos")
    axes[0].barh(top.feature, top.spearman_mos)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set(title="Single-metric association", xlabel="Spearman with MOS")
    axes[1].scatter(prediction_frame.mos, prediction_frame.predicted_mos, c="tab:blue")
    lo = min(prediction_frame.mos.min(), prediction_frame.predicted_mos.min())
    hi = max(prediction_frame.mos.max(), prediction_frame.predicted_mos.max())
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", color="black")
    axes[1].set(
        xlabel="Observed MOS", ylabel="Site-disjoint prediction", title="Cross-site calibration"
    )
    fig.tight_layout()
    fig.savefig(FIG / "exp003_ch3d_calibration.pdf", bbox_inches="tight")
    fig.savefig(FIG / "exp003_ch3d_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
