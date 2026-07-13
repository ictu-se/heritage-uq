"""Apply the same uncertainty analysis to the six-asset external holdout matrix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "008_analyze_overnight_matrix.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("matrix_analysis", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXP = ROOT / "experiments" / "exp008_external_holdout"
    module.OUT = ROOT / "experiments" / "exp009_holdout_uncertainty"
    module.FIGURE_STEM = "exp009_holdout_robustness_matrix"
    module.main()


if __name__ == "__main__":
    main()
