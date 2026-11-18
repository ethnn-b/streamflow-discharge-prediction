# Streamflow Discharge Imputation (Physically Informed MissForest (PIMF))

This repository contains the code, data and result files for **PIMF (Physically Informed MissForest)**, an iterative Random Forest imputer for daily streamflow records. PIMF reads river-network information through features rather than predictor weights: a weighted basin discharge (`A`), a travel-time routed upstream discharge (`R`, lag = round(distance / celerity) with a default celerity of 100 km/day), anchor features at the nearest observed readings on either side of each gap (`N`), and seasonal terms (`S`). Where no connectivity matrix is available it runs as `{A, N, S}`.

The same codebase is used for:
- the experiments reported in the paper (Mahanadi, Kaveri and US networks), and
- deployment as a Hugging Face Space via `app.py` (`SmokedChicken123/streamflow-imputer`).

## Quick Setup

1. Clone the repository.
2. Install dependencies:

```bash
uv sync
```

3. Run the space app locally:

```bash
uv run python app.py
```

Note: `requirements.txt` is a separate, pinned dependency list used only by the Hugging Face Space build (Spaces install from `requirements.txt`, not `pyproject.toml`). Keep it in sync manually if you change core dependencies.

## What You Will Find

- `app.py`: Gradio app for drag-and-drop CSV imputation with PIMF (every uploaded station is imputed; routing is active at stations that have contributors in the uploaded matrix).
- `src/imputers/missforest.py`: Physically Informed MissForest (PIMF) implementation.
- `src/utils/`: Data loading, spatial matrices, and metrics.
- `rerun_expts/`: the experiments, tables and figures of the paper (see "Reproducing the paper").
- `src/experiments/`: earlier exploratory scripts (windowed, chained-refit benchmarks, per-station plots). They are kept for reference and do **not** produce any number in the paper.
- `tests/`: self-checks, run directly, e.g. `python tests/test_missforest.py` and `python tests/test_anchor_placebo.py` (the latter asserts that the library default is PIMF).

## Demo-Friendly Data Layout

Ready-to-use CSVs for Hugging Face demo/testing, and the canonical source data used by the experiment scripts above (they read from these paths by default):

- `mahanadi_data/`
  - `mahanadi_discharge_data.csv`
  - `mahanadi_station_coordinates.csv`
  - `mahanadi_connectivity_matrix.csv`

- `kaveri_data/`
  - `cauv_discharge.csv`
  - `lat_long_cauv.csv`

- `us_data/` (top-10 stations used in US benchmark connectivity experiments)
  - `us_top10_discharge_data_demo.csv`
  - `us_top10_station_coordinates.csv`
  - `us_top10_connectivity_matrix.csv`
  - `us23_discharge_2013_2024.csv`, `us23_station_coordinates.csv` -- all 23 sites over the study window, for `benchmark_us_data.load_us_data()`

Note: `us_top10_discharge_data_demo.csv` intentionally contains missing values so it can be used directly for imputation demos.

Note: the per-site USGS downloads (`us_data/US_Data/`, 51 MB, 47 files) are not tracked.
`load_us_data()` uses them when present and otherwise falls back to
`us23_discharge_2013_2024.csv`, which is byte-for-byte the frame it builds from them.
Both paths mask the USGS `-999999` no-data sentinel; two such cells exist at site
01010000 on 2024-11-29/30, and treating them as discharge is what produced the
anomalous US figures in earlier drafts of the paper.

## Reproducing the paper

Every number in the paper comes from `rerun_expts/`. One cell is one (arm, seed, gap length) imputation, saved as a JSON file; the drivers skip cells that already exist, so a run can be resumed. The stored cells are in the `res_*` folders.

```bash
# Main results, factorial ablation, cross-network comparison (transductive protocol, 5 seeds x 4 gap lengths)
python rerun_expts/drive_ablation.py mahanadi kaveri us --protocol transductive --out res_main
# Wave-celerity sweep (Section 4.4) and blend sweep (Section 3.9)
python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms R,AR --celerities 25,50,75,150,200,350 --gaps 7,100 --out res_celerity
python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms AR,ARLanc --alphas 0,0.25,0.5,0.75,1 --out res_alpha
# Column-weighting invariance (Section 3.3.4)
python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms wcol --out res_invariance
python rerun_expts/check_invariance.py
# Headline contrasts and tables
python rerun_expts/headline_stats.py
python rerun_expts/make_tables.py
# Analyses that need the imputed series (supplementary tables S.T.4-S.T.7): see rerun_expts/README.md
python rerun_expts/series_analysis.py
python rerun_expts/make_figures.py --root $PWD
```

`rerun_expts/README.md` documents the protocol, the scored set (the gauges that take part in the connectivity graph), the metrics and each result folder. The arm `ARLanc` is PIMF; `base` is the seasonal-only configuration `{S}`; `vanilla` is MissForest on the discharge columns alone.

## Branches

- `main` — canonical branch. Contains the research codebase (`src/`, `rerun_expts/`), the Hugging Face Space app (`app.py`) and the demo data layout above.
- `space-demo` — currently mirrors `main`; used as the active working branch for space/demo-data changes before they land on `main`.

