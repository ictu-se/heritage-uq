"""Recreate the four-asset development inputs with the original deterministic protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "scripts" / "007_overnight_real_data_matrix.py"
ORIGINAL_ASSETS = {
    "SI_fc2bca6b-1f44-4d2c-a1b9-713a33ee0596",
    "SI_1f1ec6af-193f-4969-9185-dc5dd587239d",
    "SI_b1a04b98-c12f-4123-8093-af32c20ce5f3",
    "SI_451bcb49-01f9-4030-9d89-6f976b64572b",
}


def main() -> None:
    spec = importlib.util.spec_from_file_location("restore_inputs", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ASSET_FILTER = ORIGINAL_ASSETS
    module.OUT.mkdir(parents=True, exist_ok=True)
    rows = module.prepare_inputs()
    if len(rows) != 112:
        raise RuntimeError(f"Expected 112 development inputs, found {len(rows)}")


if __name__ == "__main__":
    main()
