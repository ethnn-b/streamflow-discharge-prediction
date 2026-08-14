"""
Reproduces the "Impact of Gap Initialization on Imputation Performance" comparison
(paper Section 4.4): compares Column Mean, Seasonal Mean, and Historical Mean
initialization strategies for the full PIMF (distance + connectivity) configuration,
restricted to downstream receiver stations that natively have less than 30% missing
data, over the 1985-1989 window on the Mahanadi basin.

Produces:
  - gap_init_comparison_results.csv: RMSE/MAE/NSE/KGE for each gap length x initialization method.
  - figure5_rmse.png / figure5_nse.png / figure5_kge.png: grouped bar charts, one per metric.
"""
import os
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.data import (
    load_and_preprocess_data,
    add_temporal_features,
    find_best_data_window,
    create_contiguous_segment_gaps_by_percent
)
from ..utils.spatial import build_distance_matrix, build_connectivity_matrix
from .benchmark_1980_1990 import run_ordered_missforest_chaining, evaluate_imputation_performance
from .contribution_ablation import MAHANADI_TABLE1_STATIONS

INIT_METHODS = ['column_mean', 'seasonal_mean', 'historical_mean']
INIT_LABELS = {'column_mean': 'Column Mean', 'seasonal_mean': 'Seasonal Mean', 'historical_mean': 'Historical Mean'}


def plot_init_comparison_bar(gap_lengths, values_by_method, ylabel, title, out_path):
    x = np.arange(len(gap_lengths))
    width = 0.25
    colors = {'column_mean': '#ED7D31', 'seasonal_mean': '#A9A9A9', 'historical_mean': '#5B9BD5'}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, method in enumerate(INIT_METHODS):
        offset = (i - 1) * width
        ax.bar(x + offset, values_by_method[method], width, label=INIT_LABELS[method],
               color=colors[method], edgecolor='black', linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}-day gap' for g in gap_lengths], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')

    all_vals = [v for vals in values_by_method.values() for v in vals]
    if ylabel == 'RMSE (cumec)':
        y_min, y_max = 0.0, max(all_vals) * 1.15
    else:
        y_min = max(0.0, min(all_vals) - 0.1)
        y_max = min(1.0, max(all_vals) + 0.05)
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='#D9D9D9')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#BFBFBF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_gap_init_comparison(discharge_path='discharge_data_cleaned.csv',
                             lat_long_path='lat_long_discharge.csv',
                             contrib_path='mahanadi_contribs.csv',
                             window_years=3,
                             target_gap_percentage=10.0,
                             missing_threshold=30.0,
                             ordering='most_full_first',
                             random_state=42,
                             make_plots=True,
                             run_tag=None):
    search_start = '1980-01-01'
    search_end = '1990-12-31'
    gap_lengths = [3, 7, 30, 100]

    tag = run_tag if run_tag is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"gap_init_comparison_results_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Gap Initialization Comparison (Mahanadi, 1980-1990)")
    print("=" * 60)

    df_original_all, df_contrib, df_coords, _, station_to_vcode = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path)
    df_with_features = add_temporal_features(df_original_all)
    df_full_original = df_with_features.loc[search_start:search_end].copy()

    all_cols = df_full_original.columns.tolist()
    temporal_features = [c for c in all_cols if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]

    all_stations = sorted([s for s in MAHANADI_TABLE1_STATIONS if s in all_cols])
    df_full_original = df_full_original[all_stations + temporal_features]

    distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode).loc[all_stations, all_stations]

    receivers = connectivity_matrix.index[(connectivity_matrix.sum(axis=1) > 0)].tolist()
    missing_pct = df_full_original[all_stations].isna().mean() * 100
    qualifying_stations = [s for s in receivers if missing_pct[s] < missing_threshold]
    print(f"Downstream receiver stations: {receivers}")
    print(f"Qualifying stations (<{missing_threshold}% native missing, 1985-1989): {qualifying_stations}")
    if not qualifying_stations:
        raise ValueError("No qualifying receiver stations found; cannot run comparison.")

    seed_window_days = (window_years * 365) + (window_years // 4)
    seed_start, seed_end = find_best_data_window(df_full_original, all_stations, search_start, search_end, seed_window_days)

    results = {}
    for gap in gap_lengths:
        print(f"\n{'=' * 20} Evaluating {gap}-day gaps {'=' * 20}")
        df_gapped = create_contiguous_segment_gaps_by_percent(
            df_full_original, qualifying_stations, gap, target_gap_percentage)

        results[gap] = {}
        for init_method in INIT_METHODS:
            print(f"\n--- {INIT_LABELS[init_method]} initialization ({gap}-day gap) ---")
            df_imp = run_ordered_missforest_chaining(
                df_gapped, seed_start, seed_end, window_years, all_stations, temporal_features,
                distance_matrix, connectivity_matrix, ordering=ordering, random_state=random_state,
                initialization_method=init_method)
            metrics, _, _ = evaluate_imputation_performance(df_full_original, df_gapped, df_imp, qualifying_stations)
            print(f"{INIT_LABELS[init_method]:16s} -> RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
                  f"NSE={metrics['NSE']:.4f} KGE={metrics['KGE']:.4f}")
            results[gap][init_method] = metrics

    rows = []
    for gap, d in results.items():
        for method, m in d.items():
            rows.append({'Gap_Length': f'{gap}_day_gap', 'Initialization': method, **m})
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'gap_init_comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    if make_plots:
        for metric, ylabel in [('RMSE', 'RMSE (cumec)'), ('NSE', 'NSE'), ('KGE', 'KGE')]:
            values_by_method = {m: [results[g][m][metric] for g in gap_lengths] for m in INIT_METHODS}
            plot_init_comparison_bar(gap_lengths, values_by_method, ylabel,
                                      f'{metric} for Different Gap Initialization Strategies',
                                      os.path.join(out_dir, f'figure5_{metric.lower()}.png'))
        print(f"Saved figures to {out_dir}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print(results_df.round(4).to_string())
    print("=" * 60)
    return results_df


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_gap_init_comparison()
