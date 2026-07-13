"""Refresh model provenance in completed matrix summaries from verified local artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "external" / "TripoSR_weights" / "model.ckpt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "TripoSR"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    checkpoint_hash = sha256_file(CHECKPOINT)
    for folder in ("exp005_real_data_robustness", "exp008_external_holdout"):
        experiment = ROOT / "experiments" / folder
        path = experiment / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        runs = pd.read_csv(experiment / "run_manifest.csv")
        input_columns = ["asset_id", "view", "condition", "path", "sha256"]
        inputs = runs[input_columns].drop_duplicates().sort_values(input_columns[:3])
        inputs.to_csv(experiment / "input_manifest.csv", index=False)
        summary["real_source_assets"] = int(runs.asset_id.nunique())
        summary["input_images"] = len(inputs)
        summary["reconstructions"] = len(runs)
        summary["model_commit"] = commit
        summary["model_checkpoint_bytes"] = CHECKPOINT.stat().st_size
        summary["model_checkpoint_sha256"] = checkpoint_hash
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
