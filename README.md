# Streamflow Discharge Imputation (Ordered Distance-Weighted MissForest)

This repository contains code and data for streamflow discharge imputation using an ordered, distance-weighted MissForest framework with optional hydrological connectivity and temporal features.

The same codebase is used for:
- research benchmarks (Mahanadi + US experiments), and
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

## What You Will Find

- `app.py`: Gradio app for drag-and-drop CSV imputation.
- `src/imputers/missforest.py`: Ordered distance-weighted MissForest implementation.
- `src/experiments/`: Benchmark scripts used to generate experiment outputs.
- `src/utils/`: Data loading, spatial matrices, and metrics.
- `benchmark_*` folders: Saved benchmark result CSVs.

## Demo-Friendly Data Layout

Ready-to-use CSVs for Hugging Face demo/testing:

- `mahanadi_data/`
  - `mahanadi_discharge_data.csv`
  - `mahanadi_station_coordinates.csv`
  - `mahanadi_connectivity_matrix.csv`

- `us_data/` (top-10 stations used in US benchmark connectivity experiments)
  - `us_top10_discharge_data_demo.csv`
  - `us_top10_station_coordinates.csv`
  - `us_top10_connectivity_matrix.csv`

Note: `us_top10_discharge_data_demo.csv` intentionally contains missing values so it can be used directly for imputation demos.

## Branches

- `main` — canonical branch. Contains the full research codebase (`src/`, experiment scripts, benchmark results), the Hugging Face Space app (`app.py`), the demo data layout above, and the paper source (`paper_latex_v2/`).
- `space-demo` — currently mirrors `main`; used as the active working branch for space/demo-data changes before they land on `main`.
- `archive/main-legacy` — frozen snapshot of the original, pre-restructure prototype (flat file layout, no `src/` package, no paper). Kept for reference only; superseded by `main`.

Previously this repo had several other branches (`final`, `v2026`, and a handful of `import/*` branches from an automated import tool) that were either fully contained in `main`'s history or unrelated one-off snapshots with no unique value. These were deleted to keep the branch list meaningful.
