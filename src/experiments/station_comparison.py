"""
Regenerates the qualitative-validation hydrograph/scatter figures (Figures 6-7)
for a specific station, reusing the same Ordered PIMF chaining run used for
Table 4. Used to compare candidate stations (e.g., Andhiyarkhore vs Basantpur)
for the 1980-1990 Mahanadi qualitative validation section.
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
from ..utils.spatial import build_distance_matrix, build_connectivity_matrix
from .benchmark_1980_1990 import (
    run_ordered_missforest_chaining,
    plot_selected_gaps_subplots,
    plot_overall_results_subplots,
)


def generate_station_figures(target_stations,
                              discharge_path='discharge_data_cleaned.csv',
                              lat_long_path='lat_long_discharge.csv',
                              contrib_path='mahanadi_contribs.csv',
                              window_years=3,
                              target_gap_percentage=10.0,
                              search_start='1980-01-01',
                              search_end='1990-12-31',
                              random_state=42):
    gap_lengths = [3, 7, 30, 100]
    out_dir = f"station_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    df_original_all, df_contrib, df_coords, _, station_to_vcode = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path)
    df_with_features = add_temporal_features(df_original_all)
    df_full_original = df_with_features.loc[search_start:search_end].copy()

    all_cols = df_full_original.columns.tolist()
    discharge_cols = [c for c in all_cols if not c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
    temporal_features = [c for c in all_cols if c not in discharge_cols]
    all_stations = sorted(discharge_cols)

    distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode).loc[all_stations, all_stations]

    seed_window_days = (window_years * 365) + (window_years // 4)
    seed_start, seed_end = find_best_data_window(df_full_original, discharge_cols, search_start, search_end, seed_window_days)

    data_list = []
    for gap in gap_lengths:
        print(f"\n{'=' * 20} {gap}-day gap: Ordered PIMF {'=' * 20}")
        df_eval_gapped = create_contiguous_segment_gaps_by_percent(df_full_original, discharge_cols, gap, target_gap_percentage)
        df_imp_full = run_ordered_missforest_chaining(
            df_eval_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features,
            distance_matrix, connectivity_matrix, ordering='most_full_first', random_state=random_state)
        data_list.append((gap, df_eval_gapped, df_imp_full))

    for station in target_stations:
        print(f"\n>>> Generating figures for station: {station}")
        plot_selected_gaps_subplots(df_full_original, data_list, discharge_cols, out_dir,
                                     f"Ordered_MissForest_{station}", target_station=station)
        plot_overall_results_subplots(df_full_original, data_list, discharge_cols, out_dir,
                                       f"Ordered_MissForest_{station}", target_station=station)

    print(f"\nSaved comparison figures to {out_dir}")
    return out_dir


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_station_figures(['andhiyarkhore', 'basantpur'])
