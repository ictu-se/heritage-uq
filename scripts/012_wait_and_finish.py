"""Wait for the primary overnight pipeline, then run holdout and localization stages."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_PID = ROOT / "logs" / "overnight_pid.txt"
LOG = ROOT / "logs" / "finisher.log"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def alive(pid: int) -> bool:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-Process -Id {pid} -ErrorAction SilentlyContinue",
        ],
        capture_output=True,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def main() -> None:
    pid = int(PRIMARY_PID.read_text().strip())
    while alive(pid):
        time.sleep(30)
    for stage in (
        "011_external_holdout_matrix.py",
        "013_analyze_external_holdout.py",
        "010_hallucination_localization.py",
    ):
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
