"""Regenerate aggregate publication figures from audited compact result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
COLORS = {
    "blur_high": "#0072B2",
    "blur_low": "#56B4E9",
    "jpeg_35": "#009E73",
    "jpeg_70": "#F0E442",
    "occlusion_15": "#E69F00",
    "occlusion_30": "#D55E00",
    "clean": "#0072B2",
}
plt.rcParams.update(
    {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 11}
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def robustness(source: Path, stem: str) -> None:
    frame = pd.read_csv(source)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition, group in frame.groupby("condition"):
        ax.errorbar(
            group.resolution,
            group.mean_chamfer,
            yerr=group.std_chamfer,
            marker="o",
            color=COLORS[condition],
            label=condition.replace("_", " "),
        )
    ax.set(xlabel="Marching-cubes resolution", ylabel="Clean-to-perturbed Chamfer")
    ax.legend(frameon=False, ncol=2)
    save(fig, stem)


def localization() -> None:
    source = ROOT / "experiments/exp007_hallucination_localization/aggregate_localization.csv"
    frame = pd.read_csv(source)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition, group in frame.groupby("condition"):
        ax.plot(
            group.resolution,
            group.unsupported_rate_002,
            marker="o",
            color=COLORS[condition],
            label=condition.replace("_", " "),
        )
    ax.set(xlabel="Marching-cubes resolution", ylabel="Unsupported vertex rate (>0.02)")
    ax.legend(frameon=False)
    save(fig, "exp007_hallucination_localization")


def main() -> None:
    robustness(
        ROOT / "experiments/exp006_uncertainty_analysis/aggregate_sensitivity.csv",
        "exp006_robustness_matrix",
    )
    robustness(
        ROOT / "experiments/exp009_holdout_uncertainty/aggregate_sensitivity.csv",
        "exp009_holdout_robustness_matrix",
    )
    localization()


if __name__ == "__main__":
    main()
