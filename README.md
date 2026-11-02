# Streamflow Discharge Imputation (Physically Informed MissForest (PIMF))

This repository contains code and data for streamflow discharge imputation using an ordered, distance-weighted MissForest framework with optional hydrological connectivity and temporal features.

The same codebase is used for:
- research benchmarks (Mahanadi, Kaveri, and US cross-regional experiments), and
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

- `app.py`: Gradio app for drag-and-drop CSV imputation.
- `src/imputers/missforest.py`: Physically Informed MissForest (PIMF) implementation.
- `src/utils/`: Data loading, spatial matrices, and metrics.
- `src/experiments/`: Benchmark scripts, grouped by study area:
  - Mahanadi basin: `benchmark_1980_1990.py` (natural vs. ordered MissForest), `station_comparison.py` (per-station hydrographs), `contribution_ablation.py`/`contribution_ablation_sweep.py` (with/without hydrological contribution data), `gap_init_comparison.py`/`gap_init_comparison_sweep.py` (gap-initialization strategies)
  - Kaveri basin: `kaveri_benchmark.py`, `kaveri_station_comparison.py`
  - US cross-regional: `us_benchmark_full.py`, which imports shared helpers from `benchmark_us_data.py`. Note this script uses its own protocol (four fixed three-year blocks, 7-day gaps, chained refits) and is **not** the script behind the paper's US table -- see "Reproducing the paper" below.
- Result folders (e.g. `station_comparison_*/`, `kaveri_station_comparison_*/`, `us_benchmark_full_*/`, `contribution_ablation_results_*/`): saved figures/CSVs from the scripts above.
- `contribution_ablation_sweep_combined.csv`, `gap_init_comparison_sweep_combined.csv`/`_means.csv`: multi-seed robustness sweep results aggregated across runs (produced by the `*_sweep.py` scripts).

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

The experiments behind the paper's tables live in `rerun_expts/`, not in
`src/experiments/`. `src/experiments/benchmark_1980_1990.py` implements the *windowed*
(chained, refit-per-block) schedule and a single gap seed, so it reproduces the
"Column-weighted MF (windowed)" arm -- not PIMF, which the paper fits once over the
whole gapped record, and not the three-seed means every table reports.

```bash
python rerun_expts/drive.py      # Mahanadi: all arms + the celerity sweep -> res/
python rerun_expts/drive2.py     # Kaveri, Mahanadi PIMF-L, linear          -> res2/
python rerun_expts/drive3.py     # US                                       -> res3/
python rerun_expts/summarise.py  # seed-mean tables, paired tests, sweeps
```

Each driver skips cells whose JSON already exists, so a run can be resumed.
`rerun_expts/README.md` maps each arm to the table it belongs to and records which
paper claims the reruns confirmed, corrected or could not check.

Not reproducible from this repository: the factorial feature ablation and the
time-shift control (paper Tables 5 and 6) and the per-station decomposition
(Table 8). The scripts that produced them were never committed.

## Branches

- `main` — canonical branch. Contains the full research codebase (`src/`, experiment scripts, benchmark results), the Hugging Face Space app (`app.py`) and the demo data layout above.
- `space-demo` — currently mirrors `main`; used as the active working branch for space/demo-data changes before they land on `main`.

