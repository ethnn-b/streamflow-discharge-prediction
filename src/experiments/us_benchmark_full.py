"""
Reproducible, seeded regeneration of the US cross-regional validation results
(Section 4.5): distance-only benchmark across all 23 stations and
distance+connectivity benchmark across the top-10 stations, both over four
3-year blocks (2013-2024). Also produces the qualitative hydrograph/scatter
figures (Figure-6/7 style, with the four blocks standing in for the four
gap-length panels used elsewhere) for two candidate stations, so a good
example can be picked for the main paper and one more for the supplementary.

Unlike the original benchmark_us_data.py / benchmark_us_connectivity.py
scripts, gap placement here uses a seeded local RNG instead of the global
`random` module, making the reported numbers exactly reproducible.
"""
import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from src.utils.data import add_temporal_features
from src.utils.spatial import build_distance_matrix
from src.utils.metrics import evaluate_metrics
from src.imputers.missforest import MissForestImputer
from src.experiments.benchmark_us_data import load_us_data, evaluate_imputation_performance

BLOCKS = [
    ('2013-01-01', '2015-12-31'),
    ('2016-01-01', '2018-12-31'),
    ('2019-01-01', '2021-12-31'),
    ('2022-01-01', '2024-12-31'),
]

TOP_10_STATIONS = [
    '01076500', '01078000', '01064500', '01048000', '01022500',
    '01054200', '01052500', '01010000', '01055000', '01047000'
]


def create_gaps_seeded(df, columns, gap_length, min_percent, rng):
    df_gapped = df.copy()
    n_days = len(df)
    target_gaps_days = np.ceil(n_days * (min_percent / 100.0))
    n_gaps = int(np.ceil(target_gaps_days / gap_length))

    for col in columns:
        valid_indices = np.where(df[col].notna())[0]
        if len(valid_indices) < n_gaps * gap_length:
            continue
        gaps_placed = 0
        attempts = 0
        max_attempts = n_gaps * 10
        col_idx = df_gapped.columns.get_loc(col)
        while gaps_placed < n_gaps and attempts < max_attempts:
            attempts += 1
            start_idx = rng.randint(0, n_days - gap_length)
            end_idx = start_idx + gap_length
            if df_gapped.iloc[start_idx:end_idx, col_idx].notna().all():
                df_gapped.iloc[start_idx:end_idx, col_idx] = np.nan
                gaps_placed += 1
    return df_gapped


def build_custom_connectivity(stations):
    connectivity = pd.DataFrame(0.0, index=stations, columns=stations)
    if '01078000' in connectivity.index and '01076500' in connectivity.columns:
        connectivity.loc['01078000', '01076500'] = 1.0
    if '01048000' in connectivity.index and '01047000' in connectivity.columns:
        connectivity.loc['01048000', '01047000'] = 1.0
    if '01055000' in connectivity.index and '01054200' in connectivity.columns:
        connectivity.loc['01055000', '01054200'] = 1.0
    return connectivity


def _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state):
    return MissForestImputer(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=random_state,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features,
        initialization_method='historical_mean',
        ordering_method='none'
    )


def run_distance_only(df_with_features, df_coords, discharge_cols, temporal_features, rng, random_state=42):
    distance_matrix = build_distance_matrix(df_coords, discharge_cols)
    connectivity_matrix = pd.DataFrame()

    all_results = {}
    current_model = None
    for start_date, end_date in BLOCKS:
        df_block_original = df_with_features.loc[start_date:end_date]
        df_block_gapped = create_gaps_seeded(df_block_original, discharge_cols, 7, 10.0, rng)

        if current_model is None:
            model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            model.fit(df_block_gapped)
            df_block_imputed = model.transform(df_block_gapped)
            current_model = model
        else:
            df_block_imputed = current_model.transform(df_block_gapped)
            new_model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            new_model.fit(df_block_imputed)
            current_model = new_model

        metrics, _, _ = evaluate_imputation_performance(df_block_original, df_block_gapped, df_block_imputed, discharge_cols)
        block_label = f"{start_date}_to_{end_date}"
        all_results[block_label] = metrics
        print(f"[distance-only, 23 stations] {block_label}: {metrics}")

    return pd.DataFrame(all_results).T


def run_connectivity(df_with_features, df_coords, discharge_cols, temporal_features, rng, random_state=42):
    distance_matrix = build_distance_matrix(df_coords, discharge_cols)
    connectivity_matrix = build_custom_connectivity(discharge_cols)

    all_results = {}
    per_station_results = {}
    data_list = []
    current_model = None
    for start_date, end_date in BLOCKS:
        df_block_original = df_with_features.loc[start_date:end_date]
        df_block_gapped = create_gaps_seeded(df_block_original, discharge_cols, 7, 10.0, rng)

        if current_model is None:
            model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            model.fit(df_block_gapped)
            df_block_imputed = model.transform(df_block_gapped)
            current_model = model
        else:
            df_block_imputed = current_model.transform(df_block_gapped)
            new_model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            new_model.fit(df_block_imputed)
            current_model = new_model

        metrics, _, _ = evaluate_imputation_performance(df_block_original, df_block_gapped, df_block_imputed, discharge_cols)
        block_label = f"{start_date}_to_{end_date}"
        all_results[block_label] = metrics
        print(f"[distance+connectivity, top-10] {block_label}: {metrics}")

        station_metrics = {}
        for station in discharge_cols:
            gap_mask = df_block_gapped[station].isnull() & df_block_original[station].notnull()
            if gap_mask.sum() > 0:
                st_metrics = evaluate_metrics(
                    df_block_original.loc[gap_mask, station].values,
                    df_block_imputed.loc[gap_mask, station].values,
                )
                station_metrics[station] = st_metrics['KGE']
        per_station_results[block_label] = station_metrics

        data_list.append((block_label, df_block_original.copy(), df_block_gapped.copy(), df_block_imputed.copy()))

    return pd.DataFrame(all_results).T, pd.DataFrame(per_station_results), data_list


def plot_us_block_subplots(data_list, discharge_cols, target_station, out_dir, method_name):
    print(f"    Plotting zoomed block subplots for {method_name} / {target_station}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns

    n_blocks = len(data_list)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

    fig, axes = plt.subplots(nrows=n_blocks, ncols=1, figsize=(12, 4 * n_blocks), squeeze=False)

    for i, (block_label, df_original, df_gapped, df_imputed) in enumerate(data_list):
        ax = axes[i, 0]
        block_years = block_label.replace('_to_', '_')[:4] + '-' + df_original.index.max().strftime('%Y')

        gap_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        is_gap = gap_mask.astype(int)
        gap_starts = is_gap[(is_gap == 1) & (is_gap.shift(1, fill_value=0) == 0)].index
        gap_ends = is_gap[(is_gap == 1) & (is_gap.shift(-1, fill_value=0) == 0)].index

        gap_candidates = []
        for start_idx, end_idx in zip(gap_starts, gap_ends):
            orig_vals = df_original.loc[start_idx:end_idx, target_station]
            imp_vals = df_imputed.loc[start_idx:end_idx, target_station]
            if len(orig_vals) >= 3:
                variance = orig_vals.var()
                mse = ((orig_vals - imp_vals) ** 2).mean() if pd.notna(variance) else None
                gap_candidates.append((start_idx, end_idx, variance, mse))

        if gap_candidates:
            variances = [g[2] for g in gap_candidates]
            var_threshold = np.percentile(variances, 75)
            high_var_gaps = [g for g in gap_candidates if g[2] >= var_threshold] or gap_candidates
            best_gap = min(high_var_gaps, key=lambda g: g[3] if g[3] is not None else float('inf'))

            start_idx, end_idx = best_gap[0], best_gap[1]
            plot_start = max(start_idx - pd.Timedelta(days=10), df_original.index.min())
            plot_end = min(end_idx + pd.Timedelta(days=10), df_original.index.max())

            ax.plot(df_original.loc[plot_start:plot_end].index, df_original.loc[plot_start:plot_end, target_station],
                    label='Original', color='#5B9BD5', linewidth=2.5, alpha=0.8)
            ax.plot(df_imputed.loc[start_idx:end_idx].index, df_imputed.loc[start_idx:end_idx, target_station],
                    label='Imputed', color='#FFC000', linewidth=2.5, linestyle='--')
            ax.axvspan(start_idx, end_idx, color='gray', alpha=0.15, label='Gap Region')

            block_title = block_label.split('_to_')
            ax.set_title(f'Block: {block_title[0][:4]}-{block_title[1][:4]}', pad=5)
            ax.set_ylabel('Discharge (m$^3$/s)')
            ax.text(-0.06, 1.05, f'({chr(97 + i)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='bottom', ha='right')

            ax.yaxis.grid(True, linestyle='-', linewidth=1, color='#D9D9D9')
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, left=True, bottom=False, top=True, right=True)
            ax.spines['bottom'].set_color('#D9D9D9')
            if i == 0:
                ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plot_filename = os.path.join(out_dir, f"zoomed_subplots_{method_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved to {plot_filename}")


def plot_us_overall_subplots(data_list, discharge_cols, target_station, out_dir, method_name):
    print(f"    Plotting overall block subplots for {method_name} / {target_station}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns

    n_blocks = len(data_list)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

    fig, axes = plt.subplots(nrows=n_blocks, ncols=2, figsize=(20, 5 * n_blocks), squeeze=False)

    for i, (block_label, df_original, df_gapped, df_imputed) in enumerate(data_list):
        ax_hydro = axes[i, 0]
        ax_scatter = axes[i, 1]
        block_title = block_label.split('_to_')
        block_short = f'{block_title[0][:4]}-{block_title[1][:4]}'

        all_true_vals, all_pred_vals = [], []
        for station in discharge_cols:
            gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
            if gap_mask.sum() > 0:
                all_true_vals.extend(df_original.loc[gap_mask, station].values)
                all_pred_vals.extend(df_imputed.loc[gap_mask, station].values)

        true_vals_global = pd.Series(all_true_vals)
        pred_vals_global = pd.Series(all_pred_vals)

        if not true_vals_global.empty:
            ax_scatter.scatter(true_vals_global, pred_vals_global, alpha=0.3, edgecolors='k', label=f'Imputed ({block_short})')
            min_val = min(true_vals_global.min(), pred_vals_global.min())
            max_val = max(true_vals_global.max(), pred_vals_global.max())

            ss_tot = np.sum((true_vals_global - np.mean(true_vals_global)) ** 2)
            ss_res = np.sum((true_vals_global - pred_vals_global) ** 2)
            nse_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label=f'1:1 Line ($NSE={nse_val:.3f}$)')
            ax_scatter.set_title(f'Scatter ({block_short})')
            ax_scatter.set_xlabel('Observed Discharge (m$^3$/s)')
            ax_scatter.set_ylabel('Predicted Discharge (m$^3$/s)')
            ax_scatter.grid(True, linestyle='-', linewidth=1, color='#D9D9D9')
            ax_scatter.legend(loc='upper left')

        target_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        pred_vals_station = df_imputed.loc[target_mask, target_station]

        ax_hydro.plot(df_original.index, df_original[target_station], label='Original Data', color='#5B9BD5', linewidth=1.5, alpha=0.9)
        if not pred_vals_station.empty:
            ax_hydro.scatter(pred_vals_station.index, pred_vals_station.values, color='red', label=f'Imputed ({block_short})', zorder=5)

        ax_hydro.set_title(f'Hydrograph ({block_short})')
        ax_hydro.set_ylabel('Discharge (m$^3$/s)')
        ax_hydro.text(-0.12, 1.05, f'({chr(97 + i)})', transform=ax_hydro.transAxes,
                      fontsize=14, fontweight='bold', va='bottom', ha='right')

        ax_hydro.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#D9D9D9')
        ax_hydro.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#D9D9D9')
        ax_hydro.set_axisbelow(True)
        sns.despine(ax=ax_hydro, left=True, bottom=False, top=True, right=True)
        ax_hydro.spines['bottom'].set_color('#D9D9D9')
        if i == 0:
            ax_hydro.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    overall_filename = os.path.join(out_dir, f"overall_subplots_{method_name.replace(' ', '_')}.png")
    plt.savefig(overall_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved to {overall_filename}")


def run_full(candidate_stations=('01054200', '01064500', '01048000', '01076500'), random_state=42):
    out_dir = f"us_benchmark_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading US Data...")
    df_combined, df_coords = load_us_data()
    df_with_features = add_temporal_features(df_combined)
    all_discharge_cols = df_combined.columns.tolist()
    all_temporal_features = [c for c in df_with_features.columns if c not in all_discharge_cols]

    rng_distance = random.Random(random_state)
    print("\n" + "=" * 60)
    print("Distance-only benchmark (23 stations)")
    print("=" * 60)
    distance_results = run_distance_only(df_with_features, df_coords, all_discharge_cols, all_temporal_features, rng_distance, random_state)
    distance_results.to_csv(os.path.join(out_dir, "us_distance_only_results.csv"))
    print(distance_results.round(4).to_string())

    existing_top_10 = [s for s in TOP_10_STATIONS if s in all_discharge_cols]
    df_top10 = df_combined[existing_top_10]
    df_top10_features = add_temporal_features(df_top10)
    top10_temporal_features = [c for c in df_top10_features.columns if c not in existing_top_10]

    rng_conn = random.Random(random_state)
    print("\n" + "=" * 60)
    print("Distance+connectivity benchmark (top-10 stations)")
    print("=" * 60)
    conn_results, per_station_results, data_list = run_connectivity(
        df_top10_features, df_coords, existing_top_10, top10_temporal_features, rng_conn, random_state)
    conn_results.to_csv(os.path.join(out_dir, "us_connectivity_results.csv"))
    per_station_results.to_csv(os.path.join(out_dir, "us_connectivity_per_station_kge.csv"))
    print(conn_results.round(4).to_string())
    print("\nPer-station KGE:")
    print(per_station_results.round(4).to_string())

    print("\nMean per-station KGE across blocks:")
    station_means = per_station_results.mean(axis=1).sort_values(ascending=False)
    print(station_means.round(4).to_string())

    for station in candidate_stations:
        if station not in existing_top_10:
            continue
        plot_us_block_subplots(data_list, existing_top_10, station, out_dir, f"US_PIMF_{station}")
        plot_us_overall_subplots(data_list, existing_top_10, station, out_dir, f"US_PIMF_{station}")

    print(f"\nAll outputs saved to {out_dir}")
    return out_dir


if __name__ == '__main__' and os.environ.get('US_BENCH_STAGE') != 'top10_control':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_full()


def run_distance_only_subset(df_with_features, df_coords, discharge_cols, temporal_features, rng, random_state=42):
    """Distance-only control restricted to the same station subset used for the
    connectivity experiment, enabling a fair like-for-like comparison."""
    distance_matrix = build_distance_matrix(df_coords, discharge_cols)
    connectivity_matrix = pd.DataFrame()

    all_results = {}
    current_model = None
    for start_date, end_date in BLOCKS:
        df_block_original = df_with_features.loc[start_date:end_date]
        df_block_gapped = create_gaps_seeded(df_block_original, discharge_cols, 7, 10.0, rng)

        if current_model is None:
            model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            model.fit(df_block_gapped)
            df_block_imputed = model.transform(df_block_gapped)
            current_model = model
        else:
            df_block_imputed = current_model.transform(df_block_gapped)
            new_model = _make_model(distance_matrix, connectivity_matrix, temporal_features, random_state)
            new_model.fit(df_block_imputed)
            current_model = new_model

        metrics, _, _ = evaluate_imputation_performance(df_block_original, df_block_gapped, df_block_imputed, discharge_cols)
        block_label = f"{start_date}_to_{end_date}"
        all_results[block_label] = metrics
        print(f"[distance-only, top-10 subset] {block_label}: {metrics}")

    return pd.DataFrame(all_results).T


if __name__ == '__main__' and os.environ.get('US_BENCH_STAGE') == 'top10_control':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("Loading US Data...")
        df_combined, df_coords = load_us_data()
        existing_top_10 = [s for s in TOP_10_STATIONS if s in df_combined.columns]
        df_top10 = df_combined[existing_top_10]
        df_top10_features = add_temporal_features(df_top10)
        top10_temporal_features = [c for c in df_top10_features.columns if c not in existing_top_10]

        rng_control = random.Random(42)
        control_results = run_distance_only_subset(df_top10_features, df_coords, existing_top_10, top10_temporal_features, rng_control, 42)
        out_dir = "us_benchmark_full_top10_control"
        os.makedirs(out_dir, exist_ok=True)
        control_results.to_csv(os.path.join(out_dir, "us_distance_only_top10_subset_results.csv"))
        print(control_results.round(4).to_string())
