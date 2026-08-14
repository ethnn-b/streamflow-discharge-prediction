# Streamflow Discharge Imputation (Ordered Distance-Weighted MissForest)

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
- `src/imputers/missforest.py`: Ordered distance-weighted MissForest implementation.
- `src/utils/`: Data loading, spatial matrices, and metrics.
- `src/experiments/`: Benchmark scripts, grouped by study area:
  - Mahanadi basin: `benchmark_1980_1990.py` (natural vs. ordered MissForest), `station_comparison.py` (per-station hydrographs), `contribution_ablation.py`/`contribution_ablation_sweep.py` (with/without hydrological contribution data), `gap_init_comparison.py`/`gap_init_comparison_sweep.py` (gap-initialization strategies)
  - Kaveri basin: `kaveri_benchmark.py`, `kaveri_station_comparison.py`
  - US cross-regional: `us_benchmark_full.py` (current, reproducible version); `benchmark_us_data.py`/`benchmark_us_connectivity.py` (earlier, non-reproducible-RNG versions, kept only because `us_benchmark_full.py` still imports shared helpers from `benchmark_us_data.py`)
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

Note: `us_top10_discharge_data_demo.csv` intentionally contains missing values so it can be used directly for imputation demos.

## Branches

- `main` — canonical branch. Contains the full research codebase (`src/`, experiment scripts, benchmark results), the Hugging Face Space app (`app.py`), the demo data layout above, and the paper source (`paper_latex_v2/`).
- `space-demo` — currently mirrors `main`; used as the active working branch for space/demo-data changes before they land on `main`.

