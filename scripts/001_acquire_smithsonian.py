"""Acquire a small, provenance-complete Smithsonian CC0 3D pilot set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "smithsonian"
EXP = ROOT / "experiments" / "exp001_dataset_acquisition"
API = "https://3d-api.si.edu/api/v1.0/content/file/search"
QUERIES = [
    "pottery",
    "vase",
    "jar",
    "bowl",
    "sculpture",
    "mask",
    "fossil",
    "mandible",
    "crab",
    "whale",
    "space shuttle",
    "apollo",
    "skull",
    "helmet",
    "statue",
    "coin",
    "artifact",
    "insect",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate(query: str) -> dict | None:
    response = requests.get(API, params={"q": query}, timeout=60)
    response.raise_for_status()
    rows = response.json().get("rows", [])
    downloadable = [
        row["content"]
        for row in rows
        if row.get("content", {}).get("usage") == "Download3D"
        and row.get("content", {}).get("model_type") == "glb"
    ]
    return downloadable[0] if downloadable else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for query in QUERIES:
        item = candidate(query)
        if not item or item["model_url"] in seen:
            continue
        seen.add(item["model_url"])
        asset_id = item["model_url"].split(":")[-1]
        path = OUT / f"SI_{asset_id}.glb"
        if not path.exists() or path.stat().st_size == 0:
            temporary = path.with_suffix(".glb.part")
            with requests.get(item["uri"], stream=True, timeout=180) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Empty download for {item['uri']}")
            temporary.replace(path)
        records.append(
            {
                "asset_id": f"SI_{asset_id}",
                "query": query,
                "model_url": item["model_url"],
                "download_url": item["uri"],
                "source_api": API,
                "source_institution": "Smithsonian Institution",
                "rights": "Smithsonian Open Access / CC0-designated content",
                "quality": item.get("quality", ""),
                "file_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "local_path": path.relative_to(ROOT).as_posix(),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(EXP / "asset_manifest.csv", index=False)
    summary = {
        "requested_queries": len(QUERIES),
        "unique_assets": len(manifest),
        "total_bytes": int(manifest.file_bytes.sum()) if len(manifest) else 0,
        "claim_boundary": "Dataset acquisition only; no reconstruction-quality claim.",
    }
    (EXP / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
