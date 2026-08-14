"""
Reproduces the "Effect of Hydrological Contribution" ablation (paper Section 4.1):
compares the full PIMF weighting (distance + connectivity) against a distance-only
("without contribution") configuration, restricted to downstream receiver stations
(stations with at least one upstream contributor per the connectivity matrix) that
natively have less than 30% missing data over 1980-1990, on the Mahanadi basin's
16 canonical gauging stations (Table 1 of the manuscript).

Produces:
  - contribution_ablation_results.csv: RMSE/MAE/NSE/KGE for each gap length, with vs without.
  - figure3_kge.png / figure4_nse.png: grouped bar charts matching the manuscript's
    existing Figure 3 / Figure 4 styling.
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

# The 16 gauging stations documented in Table 1 of the manuscript. The raw
# discharge_data_cleaned.csv contains additional stations not part of the
# reported Mahanadi network; we restrict to Table 1's stations for fidelity
# to the paper's stated study area.
MAHANADI_TABLE1_STATIONS = [
    'andhiyarkhore', 'bamnidhi', 'baronda', 'basantpur', 'ghatora', 'jondhra',
    'kelo', 'kotni', 'kurubhata', 'paramanpur', 'patharidih', 'rajim',
    'rampur', 'seorinarayan', 'simga', 'sundergarh'
]


def plot_contribution_bar(gap_lengths, without_vals, with_vals, ylabel, title, out_path):
    x = np.arange(len(gap_lengths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, without_vals, width, label='Without Contribution Data',
           color='#5B9BD5', edgecolor='black', linewidth=1.0)
    ax.bar(x + width / 2, with_vals, width, label='With Contribution Data',
           color='#5B9BD5', edgecolor='black', linewidth=1.0, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}-day gap' for g in gap_lengths], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')

    all_vals = list(without_vals) + list(with_vals)
    y_min = max(0.0, min(all_vals) - 0.1)
    y_max = min(1.0, max(all_vals) + 0.05)
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, linestyle='-', linewidth=0.7, color='#D9D9D9')
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor('#BFBFBF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_contribution_ablation(discharge_path='discharge_data_cleaned.csv',
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
    out_dir = f"contribution_ablation_results_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Contribution Ablation (Mahanadi, 1980-1990)")
    print("=" * 60)

    df_original_all, df_contrib, df_coords, _, station_to_vcode = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path)
    df_with_features = add_temporal_features(df_original_all)
    df_full_original = df_with_features.loc[search_start:search_end].copy()

    all_cols = df_full_original.columns.tolist()
    temporal_features = [c for c in all_cols if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]

    all_stations = sorted([s for s in MAHANADI_TABLE1_STATIONS if s in all_cols])
    print(f"Using {len(all_stations)} canonical Mahanadi (Table 1) stations: {all_stations}")
    df_full_original = df_full_original[all_stations + temporal_features]

    distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode).loc[all_stations, all_stations]
    zero_connectivity = pd.DataFrame(0.0, index=all_stations, columns=all_stations)

    receivers = connectivity_matrix.index[(connectivity_matrix.sum(axis=1) > 0)].tolist()
    missing_pct = df_full_original[all_stations].isna().mean() * 100
    qualifying_stations = [s for s in receivers if missing_pct[s] < missing_threshold]
    print(f"Downstream receiver stations: {receivers}")
    print(f"Qualifying stations (<{missing_threshold}% native missing, 1980-1990): {qualifying_stations}")
    if not qualifying_stations:
        raise ValueError("No qualifying receiver stations found; cannot run ablation.")

    seed_window_days = (window_years * 365) + (window_years // 4)
    seed_start, seed_end = find_best_data_window(df_full_original, all_stations, search_start, search_end, seed_window_days)

    results = {}
    for gap in gap_lengths:
        print(f"\n{'=' * 20} Evaluating {gap}-day gaps {'=' * 20}")
        df_gapped = create_contiguous_segment_gaps_by_percent(
            df_full_original, qualifying_stations, gap, target_gap_percentage)

        print(f"\n--- WITH hydrological contribution ({gap}-day gap) ---")
        df_imp_with = run_ordered_missforest_chaining(
            df_gapped, seed_start, seed_end, window_years, all_stations, temporal_features,
            distance_matrix, connectivity_matrix, ordering=ordering, random_state=random_state)
        metrics_with, _, _ = evaluate_imputation_performance(df_full_original, df_gapped, df_imp_with, qualifying_stations)
        print(f"WITH contribution    -> RMSE={metrics_with['RMSE']:.4f} MAE={metrics_with['MAE']:.4f} "
              f"NSE={metrics_with['NSE']:.4f} KGE={metrics_with['KGE']:.4f}")

        print(f"\n--- WITHOUT hydrological contribution ({gap}-day gap) ---")
        df_imp_without = run_ordered_missforest_chaining(
            df_gapped, seed_start, seed_end, window_years, all_stations, temporal_features,
            distance_matrix, zero_connectivity, ordering=ordering, random_state=random_state)
        metrics_without, _, _ = evaluate_imputation_performance(df_full_original, df_gapped, df_imp_without, qualifying_stations)
        print(f"WITHOUT contribution -> RMSE={metrics_without['RMSE']:.4f} MAE={metrics_without['MAE']:.4f} "
              f"NSE={metrics_without['NSE']:.4f} KGE={metrics_without['KGE']:.4f}")

        results[gap] = {'with': metrics_with, 'without': metrics_without}

    rows = []
    for gap, d in results.items():
        for variant, m in d.items():
            rows.append({'Gap_Length': f'{gap}_day_gap', 'Variant': variant, **m})
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'contribution_ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    if make_plots:
        kge_with = [results[g]['with']['KGE'] for g in gap_lengths]
        kge_without = [results[g]['without']['KGE'] for g in gap_lengths]
        nse_with = [results[g]['with']['NSE'] for g in gap_lengths]
        nse_without = [results[g]['without']['NSE'] for g in gap_lengths]

        plot_contribution_bar(gap_lengths, kge_without, kge_with, 'KGE',
                               'Effect of Contribution Data on KGE',
                               os.path.join(out_dir, 'figure3_kge.png'))
        plot_contribution_bar(gap_lengths, nse_without, nse_with, 'NSE',
                               'Effect of Contribution Data on NSE',
                               os.path.join(out_dir, 'figure4_nse.png'))

        print(f"Saved figures to {out_dir}")
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print(results_df.round(4).to_string())
    print("=" * 60)
    return results_df


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_contribution_ablation()
