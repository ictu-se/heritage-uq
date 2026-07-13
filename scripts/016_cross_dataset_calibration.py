"""Calibrate degradation risk on development assets and test it on category holdout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments" / "exp006_uncertainty_analysis" / "perturbation_sensitivity.csv"
TEST = ROOT / "experiments" / "exp009_holdout_uncertainty" / "perturbation_sensitivity.csv"
CH3D = ROOT / "experiments" / "exp003_ch3d_quality_calibration" / "summary.json"
OUT = ROOT / "experiments" / "exp010_cross_dataset_calibration"
FIG = ROOT / "figures"
SEVERE = {"blur_high", "jpeg_35", "occlusion_30"}


def design(frame: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    raw = np.column_stack(
        [
            np.log1p(frame.clean_to_perturbed_chamfer.to_numpy()),
            np.log2(frame.resolution.to_numpy() / 128),
        ]
    )
    return np.column_stack([np.ones(len(frame)), (raw - mean) / scale])


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1 / (1 + np.exp(-values))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        selected = (probabilities >= lower) & (
            (probabilities <= upper) if upper == 1 else (probabilities < upper)
        )
        if selected.any():
            ece += selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
    return {
        "records": int(len(labels)),
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "auroc": auc(labels, probabilities),
        "ece_10_bin": float(ece),
    }


def reliability(labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], list[float]]:
    predicted = []
    observed = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        selected = (probabilities >= lower) & (
            (probabilities <= upper) if upper == 1 else (probabilities < upper)
        )
        if selected.any():
            predicted.append(float(probabilities[selected].mean()))
            observed.append(float(labels[selected].mean()))
    return predicted, observed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(DEV)
    holdout = pd.read_csv(TEST)
    raw_development = np.column_stack(
        [
            np.log1p(development.clean_to_perturbed_chamfer.to_numpy()),
            np.log2(development.resolution.to_numpy() / 128),
        ]
    )
    mean = raw_development.mean(axis=0)
    scale = np.maximum(raw_development.std(axis=0), 1e-12)
    x_development = design(development, mean, scale)
    x_holdout = design(holdout, mean, scale)
    y_development = development.condition.isin(SEVERE).astype(int).to_numpy()
    y_holdout = holdout.condition.isin(SEVERE).astype(int).to_numpy()

    def loss(weights: np.ndarray) -> float:
        probabilities = sigmoid(x_development @ weights)
        likelihood = -np.mean(
            y_development * np.log(probabilities + 1e-12)
            + (1 - y_development) * np.log(1 - probabilities + 1e-12)
        )
        return float(likelihood + 1e-4 * np.sum(weights[1:] ** 2))

    fit = minimize(loss, np.zeros(x_development.shape[1]), method="L-BFGS-B")
    assert fit.success, fit.message
    development_probability = sigmoid(x_development @ fit.x)
    holdout_probability = sigmoid(x_holdout @ fit.x)
    predictions = holdout.copy()
    predictions["is_severe"] = y_holdout
    predictions["calibrated_severe_probability"] = holdout_probability
    predictions.to_csv(OUT / "holdout_predictions.csv", index=False)
    ch3d = json.loads(CH3D.read_text(encoding="utf-8"))
    summary = {
        "calibration_train_split": "four Smithsonian development assets",
        "calibration_test_split": "six disjoint Smithsonian category-holdout assets",
        "target": "high versus low controlled degradation within each degradation family",
        "features": ["clean-to-perturbed Chamfer", "extraction resolution"],
        "development": metrics(y_development, development_probability),
        "external_holdout": metrics(y_holdout, holdout_probability),
        "ch3d_reco_reference": {
            "models": ch3d["models"],
            "sites": ch3d["sites"],
            "site_disjoint_spearman": ch3d["site_disjoint_linear_spearman"],
            "site_disjoint_mae": ch3d["site_disjoint_linear_mae"],
        },
        "claim_boundary": "The Smithsonian classifier transfers across held-out object categories. CH3D-Reco is reported as an independent MOS calibration benchmark because its published feature and rating scales are not commensurate with the reconstruction-instability score.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for label, labels, probabilities in (
        ("development", y_development, development_probability),
        ("category holdout", y_holdout, holdout_probability),
    ):
        predicted, observed = reliability(labels, probabilities)
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.9)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted probability of severe degradation",
        ylabel="Observed severe-degradation frequency",
        title="Development-to-holdout calibration",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "exp010_cross_dataset_calibration.pdf", bbox_inches="tight")
    fig.savefig(FIG / "exp010_cross_dataset_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
