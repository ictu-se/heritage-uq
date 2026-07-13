"""Run a separate matrix on the six newly acquired Smithsonian holdout assets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "007_overnight_real_data_matrix.py"
ORIGINAL_ASSETS = {
    "SI_fc2bca6b-1f44-4d2c-a1b9-713a33ee0596",
    "SI_1f1ec6af-193f-4969-9185-dc5dd587239d",
    "SI_b1a04b98-c12f-4123-8093-af32c20ce5f3",
    "SI_451bcb49-01f9-4030-9d89-6f976b64572b",
}


def main() -> None:
    spec = importlib.util.spec_from_file_location("robustness_matrix", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = pd.read_csv(
        ROOT / "experiments" / "exp001_dataset_acquisition" / "asset_manifest.csv"
    )
    holdout = set(manifest.asset_id) - ORIGINAL_ASSETS
    if len(holdout) < 6:
        raise RuntimeError(f"Expected six holdout assets, found {len(holdout)}")
    module.OUT = ROOT / "experiments" / "exp008_external_holdout"
    module.INPUTS = ROOT / "data" / "processed" / "exp008_holdout_inputs"
    module.LOG = ROOT / "logs" / "external_holdout.log"
    module.RESOLUTIONS = (128, 192, 256, 320, 384, 448, 512, 640)
    module.CONDITIONS = (
        "clean",
        "blur_low",
        "blur_high",
        "jpeg_70",
        "jpeg_35",
        "occlusion_15",
        "occlusion_30",
    )
    module.ASSET_FILTER = holdout
    module.main()


if __name__ == "__main__":
    main()
