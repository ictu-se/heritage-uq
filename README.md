# 07 - Uncertainty-Aware 3D Heritage Reconstruction

Reproducible experiments for uncertainty-aware geometric quality assessment of single-image 3D cultural-heritage reconstruction.

## Completed real-data campaign

The audited factorial campaign contains **2,240 TripoSR reconstructions** from 10 Smithsonian Open Access 3D assets, plus a 16-run pilot (2,256 generated reconstructions in total). Development and category-holdout sets are disjoint. Each asset uses four rendered views, seven controlled input conditions, and eight extraction resolutions. The independent CH3D-Reco analysis covers 45 models from five sites and 182,023 published interaction-log rows.

Completed experiments:

- `exp001`: Smithsonian acquisition and cryptographic provenance
- `exp002`: controlled mesh degradation
- `exp003`: CH3D-Reco MOS calibration with site-disjoint validation
- `exp004`: initial zero-shot multiview reconstruction
- `exp005`/`exp008`: development and external-holdout robustness matrices
- `exp006`/`exp009`: reconstruction uncertainty and perturbation sensitivity
- `exp007`: unsupported-surface localization and colored PLY heatmaps
- `exp010`: development-to-category-holdout probability calibration

Compact audited results are stored in `experiments/campaign_summary.json` and the numbered experiment directories. Source Smithsonian assets are acquired through `scripts/001_acquire_smithsonian.py`; generated images, meshes, model weights, and logs are intentionally excluded because they are reproducible and exceed ordinary Git hosting limits.

## Reproduction

Create a Python 3.12 environment and install the pinned dependencies with `python -m pip install -r requirements.txt`. Run the numbered scripts in ascending order; each script resolves paths relative to the repository root. TripoSR requires its upstream CUDA installation and checkpoint, whose revision and digest are recorded by the experiment provenance scripts. Run `python scripts/014_finalize_campaign.py` to repeat the integrity audit and `python -m pytest -q` for regression tests.

## Claim boundary

These experiments measure geometric robustness, instability, unsupported surface, and objective calibration. They do not establish cultural authenticity and introduce no new human-subject evaluation.
