"""
Reproduces the Kaveri basin streamflow-imputation evaluation (paper Table 3).

Unlike the Mahanadi analysis, no hydrological connectivity (contributor) matrix
is available for the Kaveri network, so this configuration relies solely on
geodesic-distance weighting plus seasonal features -- the same "no contributor"
ablation used in the original Mahanadi contribution study (train_no_contributor_model),
applied here as the only feasible PIMF configuration for this basin. All 12
gauging stations are evaluated (no receiver-station restriction, since that
concept requires a connectivity matrix this basin does not have). Averaged
over multiple random seeds for robustness, matching the Mahanadi methodology.
"""
import os
import warnings
from datetime import datetime

import pandas as pd

from ..utils.data import (
    load_and_preprocess_data,
    add_temporal_features,
    find_best_data_window,
    create_contiguous_segment_gaps_by_percent
)
from ..utils.spatial import build_distance_matrix
from .benchmark_1980_1990 import run_ordered_missforest_chaining, evaluate_imputation_performance


def run_kaveri_benchmark(discharge_path='kaveri_data/cauv_discharge.csv',
                          lat_long_path='kaveri_data/lat_long_cauv.csv',
                          window_years=3,
                          target_gap_percentage=10.0,
                          search_start='1980-01-01',
                          search_end='1990-12-31',
                          seeds=(42, 7, 123),
                          run_tag=None):
    gap_lengths = [3, 7, 30, 100]

    tag = run_tag if run_tag is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"kaveri_benchmark_results_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Kaveri Basin Benchmark ({search_start} to {search_end}, distance-only PIMF)")
    print("=" * 60)

    df_original_all, _, df_coords, _, _ = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path=None)
    df_with_features = add_temporal_features(df_original_all)
    df_full_original = df_with_features.loc[search_start:search_end].copy()

    all_cols = df_full_original.columns.tolist()
    temporal_features = [c for c in all_cols if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
    all_stations = sorted([c for c in all_cols if c not in temporal_features])
    print(f"Using {len(all_stations)} Kaveri stations: {all_stations}")

    distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
    zero_connectivity = pd.DataFrame(0.0, index=all_stations, columns=all_stations)

    seed_window_days = (window_years * 365) + (window_years // 4)
    seed_start, seed_end = find_best_data_window(df_full_original, all_stations, search_start, search_end, seed_window_days)

    rows = []
    for gap in gap_lengths:
        print(f"\n{'=' * 20} Evaluating {gap}-day gaps {'=' * 20}")
        df_gapped = create_contiguous_segment_gaps_by_percent(
            df_full_original, all_stations, gap, target_gap_percentage)

        for seed in seeds:
            print(f"\n--- Kaveri distance-only PIMF ({gap}-day gap, seed={seed}) ---")
            df_imp = run_ordered_missforest_chaining(
                df_gapped, seed_start, seed_end, window_years, all_stations, temporal_features,
                distance_matrix, zero_connectivity, ordering='most_full_first', random_state=seed)
            metrics, _, _ = evaluate_imputation_performance(df_full_original, df_gapped, df_imp, all_stations)
            print(f"seed={seed} -> RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
                  f"NSE={metrics['NSE']:.4f} KGE={metrics['KGE']:.4f}")
            rows.append({'Gap_Length': f'{gap}_day_gap', 'random_state': seed, **metrics})

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'kaveri_benchmark_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    means = results_df.groupby('Gap_Length')[['RMSE', 'MAE', 'NSE', 'KGE']].mean()
    means = means.reindex([f'{g}_day_gap' for g in gap_lengths])
    means_path = os.path.join(out_dir, 'kaveri_benchmark_means.csv')
    means.to_csv(means_path)
    print("\n" + "=" * 60)
    print("SEED-AVERAGED RESULTS (Table 3 candidate)")
    print(means.round(4).to_string())
    print("=" * 60)
    print(f"Saved means to {means_path}")
    return results_df, means


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_kaveri_benchmark()
