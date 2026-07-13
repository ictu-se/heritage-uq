"""Resumable overnight pipeline; runs every remaining stage in order."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "overnight_supervisor.log"
STAGES = [
    "007_overnight_real_data_matrix.py",
    "008_analyze_overnight_matrix.py",
]


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def main() -> None:
    for stage in STAGES:
        log(f"START {stage}")
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / stage)], cwd=ROOT)
        log(f"END {stage} exit={completed.returncode}")
        if completed.returncode:
            raise SystemExit(completed.returncode)
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    log(f"END pytest exit={completed.returncode}")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
