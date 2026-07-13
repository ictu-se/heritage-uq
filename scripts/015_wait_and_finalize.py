"""Wait for the finisher supervisor, then audit, lint, and test the complete campaign."""

from __future__ import annotations

import subprocess
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "logs" / "finisher_pid.txt"
LOG = ROOT / "logs" / "final_audit.log"
SUCCESS = ROOT / "logs" / "FINAL_AUDIT_PASSED.json"


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def alive(pid: int) -> bool:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
        capture_output=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def run(command: list[str]) -> None:
    log(f"START {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT)
    log(f"END exit={result.returncode} {' '.join(command)}")
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    while alive(pid):
        time.sleep(30)
    run([sys.executable, str(ROOT / "scripts" / "018_refresh_model_provenance.py")])
    run([sys.executable, str(ROOT / "scripts" / "016_cross_dataset_calibration.py")])
    run([sys.executable, str(ROOT / "scripts" / "014_finalize_campaign.py")])
    run([sys.executable, "-m", "ruff", "check", "scripts", "tests"])
    run([sys.executable, "-m", "pytest", "-q"])
    SUCCESS.write_text(
        json.dumps(
            {
                "status": "passed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "gates": ["campaign integrity audit", "ruff", "pytest"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"PASS marker={SUCCESS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
