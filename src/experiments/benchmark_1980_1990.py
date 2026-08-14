import os
import random
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
from ..utils.metrics import evaluate_metrics

from ..imputers.baselines import historical_mean_imputation, linear_interpolation_imputation
from ..imputers.kalman import kalman_imputation
from ..imputers.knn import knn_imputation
from ..imputers.missforest import MissForestImputer


def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    y_true_eval, y_pred_eval = [], []
    for station in discharge_cols:
        if station not in df_imputed.columns or station not in df_gapped.columns or station not in df_original.columns:
            continue
        
        gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
        if gap_mask.sum() > 0:
            predicted_vals = df_imputed.loc[gap_mask.index[gap_mask], station].values
            true_vals = df_original.loc[gap_mask.index[gap_mask], station].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)

    if y_true_eval:
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics, y_true_eval, y_pred_eval
    else:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}, [], []


def run_ordered_missforest_chaining(df_full_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features, distance_matrix, connectivity_matrix, ordering='most_full_first', random_state=42, initialization_method='historical_mean'):
    print(f"  Starting 3-year *chaining* ordered MissForest process (ordering: {ordering}, init: {initialization_method})...")

    def create_model():
        return MissForestImputer(
            distance_matrix=distance_matrix,
            connectivity=connectivity_matrix,
            max_iter=10,
            n_estimators=100,
            random_state=random_state,
            distance_weighting_type='inverse',
            temporal_feature_columns=temporal_features,
            initialization_method=initialization_method,
            ordering_method=ordering
        )

    print(f"  Training initial model on seed: {seed_start.date()} to {seed_end.date()}")
    df_seed_gapped = df_full_gapped.loc[seed_start:seed_end].copy()
    
    initial_model = create_model()
    initial_model.fit(df_seed_gapped)
    df_seed_imputed = initial_model.transform(df_seed_gapped)
    
    all_imputed_blocks = [df_seed_imputed]
    
    full_start = df_full_gapped.index.min()
    full_end = df_full_gapped.index.max()
    window_timedelta = pd.Timedelta(days=(window_years * 365) + (window_years // 4))

    print("\n  Chaining backwards from seed...")
    current_training_model = initial_model
    current_start = seed_start
    
    while current_start > full_start:
        prev_end = current_start - pd.Timedelta(days=1)
        prev_start = max(prev_end - window_timedelta, full_start)
        df_prev_gapped = df_full_gapped.loc[prev_start:prev_end]
        if df_prev_gapped.empty: break
            
        print(f"    Imputing backwards block: {prev_start.date()} to {prev_end.date()}")
        df_prev_imputed = current_training_model.transform(df_prev_gapped)
        all_imputed_blocks.append(df_prev_imputed)
        
        print(f"    Training new model on: {prev_start.date()} to {prev_end.date()}")
        new_model = create_model()
        new_model.fit(df_prev_imputed)
        current_training_model = new_model
        current_start = prev_start

    print("\n  Chaining forwards from seed...")
    current_training_model = initial_model 
    current_end = seed_end
    
    while current_end < full_end:
        next_start = current_end + pd.Timedelta(days=1)
        next_end = min(next_start + window_timedelta, full_end)
        df_next_gapped = df_full_gapped.loc[next_start:next_end]
        if df_next_gapped.empty: break
            
        print(f"    Imputing forwards block: {next_start.date()} to {next_end.date()}")
        df_next_imputed = current_training_model.transform(df_next_gapped)
        all_imputed_blocks.append(df_next_imputed)

        print(f"    Training new model on: {next_start.date()} to {next_end.date()}")
        new_model = create_model()
        new_model.fit(df_next_imputed)
        current_training_model = new_model
        current_end = next_end
            
    print("\n  ✓ Rolling/Chaining imputation complete.")
    df_final_imputed = pd.concat(all_imputed_blocks)
    return df_final_imputed.sort_index()


def plot_selected_gaps_subplots(df_original, data_list, discharge_cols, out_dir, method_name, target_station=None):
    print(f"    Plotting best gaps subplots for {method_name}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns
    import os
    import numpy as np
    import pandas as pd

    if target_station is None:
        completeness = df_original[discharge_cols].notna().sum()
        target_station = completeness.idxmax()

    n_gaps = len(data_list)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
    
    fig, axes = plt.subplots(nrows=n_gaps, ncols=1, figsize=(12, 4 * n_gaps), squeeze=False)
    
    for i, (gap_length, df_gapped, df_imputed) in enumerate(data_list):
        ax = axes[i, 0]
        gap_candidates = []
        gap_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        is_gap = gap_mask.astype(int)
        gap_starts = is_gap[(is_gap == 1) & (is_gap.shift(1, fill_value=0) == 0)].index
        gap_ends = is_gap[(is_gap == 1) & (is_gap.shift(-1, fill_value=0) == 0)].index

        for start_idx, end_idx in zip(gap_starts, gap_ends):
            orig_vals = df_original.loc[start_idx:end_idx, target_station]
            imp_vals = df_imputed.loc[start_idx:end_idx, target_station]
            if len(orig_vals) >= 3:
                variance = orig_vals.var()
                mse = ((orig_vals - imp_vals) ** 2).mean() if pd.notna(variance) else None
                # Calculate correlation if possible
                corr = orig_vals.corr(imp_vals) if orig_vals.std() > 0 and imp_vals.std() > 0 else None
                gap_candidates.append((start_idx, end_idx, variance, mse, corr))

        if gap_candidates:
            if gap_length == 3:
                # Hybrid: maximize (normalized corr) and minimize (normalized mse)
                valid = [g for g in gap_candidates if g[4] is not None and g[3] is not None]
                if valid:
                    corrs = np.array([g[4] for g in valid])
                    mses = np.array([g[3] for g in valid])
                    # Normalize: corr (0-1, 1 best), mse (0-1, 0 best)
                    norm_corrs = (corrs - corrs.min()) / (corrs.max() - corrs.min() + 1e-8)
                    norm_mses = (mses - mses.min()) / (mses.max() - mses.min() + 1e-8)
                    alpha = 0.5
                    scores = alpha * norm_corrs + (1 - alpha) * (1 - norm_mses)
                    best_idx = np.argmax(scores)
                    best_gap = valid[best_idx]
                else:
                    # fallback: use lowest mse
                    best_gap = min(gap_candidates, key=lambda g: g[3] if g[3] is not None else float('inf'))
            else:
                variances = [g[2] for g in gap_candidates]
                var_threshold = np.percentile(variances, 75)
                high_var_gaps = [g for g in gap_candidates if g[2] >= var_threshold]
                if not high_var_gaps:
                    high_var_gaps = gap_candidates
                best_gap = min(high_var_gaps, key=lambda g: g[3] if g[3] is not None else float('inf'))

            station, start_idx, end_idx = target_station, best_gap[0], best_gap[1]
            context_days = min(15, max(5, gap_length // 2))
            plot_start = max(start_idx - pd.Timedelta(days=context_days), df_original.index.min())
            plot_end = min(end_idx + pd.Timedelta(days=context_days), df_original.index.max())

            ax.plot(df_original.loc[plot_start:plot_end].index, df_original.loc[plot_start:plot_end, station],
                    label='Original', color='#5B9BD5', linewidth=2.5, alpha=0.8)
            ax.plot(df_imputed.loc[start_idx:end_idx].index, df_imputed.loc[start_idx:end_idx, station],
                    label=f'Imputed', color='#FFC000', linewidth=2.5, linestyle='--')
            ax.axvspan(start_idx, end_idx, color='gray', alpha=0.15, label='Gap Region')

            ax.set_title(f'Gap: {gap_length} days', pad=5)
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
    print(f"    Saved zoomed subplots to {plot_filename}")


def plot_overall_results_subplots(df_original, data_list, discharge_cols, out_dir, method_name, target_station=None):
    print(f"    Plotting overall subplots for {method_name}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns
    import numpy as np
    import pandas as pd
    import os

    if target_station is None:
        completeness = df_original[discharge_cols].notna().sum()
        target_station = completeness.idxmax()
    n_gaps = len(data_list)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
    
    # 2 columns: left is Hydrograph, right is R^2 scatter
    fig, axes = plt.subplots(nrows=n_gaps, ncols=2, figsize=(20, 5 * n_gaps), squeeze=False)
    
    for i, (gap_length, df_gapped, df_imputed) in enumerate(data_list):
        ax_hydro = axes[i, 0]
        ax_scatter = axes[i, 1]
        
        # --- Scatter Plot (Global) ---
        all_true_vals = []
        all_pred_vals = []
        for station in discharge_cols:
            gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
            if gap_mask.sum() > 0:
                all_true_vals.extend(df_original.loc[gap_mask, station].values)
                all_pred_vals.extend(df_imputed.loc[gap_mask, station].values)
                
        true_vals_global = pd.Series(all_true_vals)
        pred_vals_global = pd.Series(all_pred_vals)
        
        if not true_vals_global.empty:
            ax_scatter.scatter(true_vals_global, pred_vals_global, alpha=0.3, edgecolors='k', label=f'Imputed (Gaps: {gap_length}d)')
            min_val = min(true_vals_global.min(), pred_vals_global.min())
            max_val = max(true_vals_global.max(), pred_vals_global.max())
            
            if len(true_vals_global) > 1:
                ss_tot = np.sum((true_vals_global - np.mean(true_vals_global))**2)
                ss_res = np.sum((true_vals_global - pred_vals_global)**2)
                nse_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                nse_val = np.nan

            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label=f'1:1 Line ($NSE={nse_val:.3f}$)')
            ax_scatter.set_title(f'Scatter ({gap_length}d)')
            ax_scatter.set_xlabel('Observed Discharge (m$^3$/s)')
            ax_scatter.set_ylabel('Predicted Discharge (m$^3$/s)')
            ax_scatter.grid(True, linestyle='-', linewidth=1, color='#D9D9D9')
            ax_scatter.legend(loc='upper left')

        # --- Hydrograph (Target Station) ---
        target_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        pred_vals_station = df_imputed.loc[target_mask, target_station]

        ax_hydro.plot(df_original.index, df_original[target_station], label='Original Data', color='#5B9BD5', linewidth=1.5, alpha=0.9)
        if not pred_vals_station.empty:
            ax_hydro.scatter(pred_vals_station.index, pred_vals_station.values, color='red', label=f'Imputed ({gap_length}d)', zorder=5)

        ax_hydro.set_title(f'Hydrograph ({gap_length}d)')
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
    print(f"    Saved overall subplots to {overall_filename}")


def run_benchmark(discharge_path='discharge_data_cleaned.csv',
                  lat_long_path='lat_long_discharge.csv',
                  contrib_path='mahanadi_contribs.csv',
                  test_mode=False,
                  window_years=3):
                  
    search_start = '1980-01-01'
    search_end = '1990-12-31'
    target_gap_percentage = 10.0
    gap_lengths = [30] if test_mode else [3, 7, 30, 100]

    out_dir = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    print("="*60)
    print(f"Running Refactored Benchmark (1980-1990)")
    print(f"Test mode: {test_mode}")
    print("="*60)

    df_original_all, df_contrib, df_coords, _, station_to_vcode = load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    df_with_features = add_temporal_features(df_original_all)
    df_full_original = df_with_features.loc[search_start:search_end].copy()

    all_cols = df_full_original.columns.tolist()
    discharge_cols = [c for c in all_cols if not c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
    temporal_features = [c for c in all_cols if c not in discharge_cols]

    if test_mode:
        discharge_cols = discharge_cols[:5]
        df_full_original = df_full_original[discharge_cols + temporal_features]

    all_stations = sorted(discharge_cols)
    distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode).loc[all_stations, all_stations]

    seed_window_days = (window_years * 365) + (window_years // 4)
    seed_start, seed_end = find_best_data_window(df_full_original, discharge_cols, search_start, search_end, seed_window_days)
    
    df_seed_gapped = df_full_original.loc[seed_start:seed_end].copy()
    eval_mask = ~df_full_original.index.isin(df_seed_gapped.index)
    df_eval_original = df_full_original.loc[eval_mask].copy()

    print(f"Seed block: {seed_start.date()} to {seed_end.date()}")
    all_results = {}
    model_plot_data = {'Ordered_MissForest': [], 'Natural_MissForest': []}

    for gap in gap_lengths:
        print(f"\nEvaluating {gap}-day gaps...")
        df_eval_gapped = create_contiguous_segment_gaps_by_percent(df_eval_original, discharge_cols, gap, target_gap_percentage)
        df_full_gapped = pd.concat([df_seed_gapped, df_eval_gapped]).sort_index()

        gap_res = {}

        # 1. Historical Mean
        try:
            print("\nMethod 1: Historical Mean")
            df_imp = historical_mean_imputation(df_eval_gapped, discharge_cols, training_data=df_full_gapped)
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Historical_Mean'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        # 2. Linear Interpolation
        try:
            print("\nMethod 2: Linear Interpolation")
            df_imp = linear_interpolation_imputation(df_eval_gapped, discharge_cols)
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Linear_Interpolation'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        # 3. Kalman Filters
        try:
            print("\nMethod 3: Kalman Filters")
            df_imp = kalman_imputation(df_seed_gapped, df_eval_gapped, discharge_cols)
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Kalman_Filters'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        # 4. KNN
        try:
            print("\nMethod 4: KNN (k=5)")
            df_imp = knn_imputation(df_seed_gapped, df_eval_gapped, df_full_original.columns.tolist(), k=5)
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['KNN'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        # 5. Vanilla MissForest
        try:
            print("\nMethod 5: Vanilla MissForest")
            model = MissForestImputer(distance_matrix=None, connectivity=None)
            model.fit(df_seed_gapped)
            df_imp = model.transform(df_eval_gapped)
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Vanilla_MissForest'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        # 6. Ordered MissForest
        try:
            print("\nMethod 6: Ordered MissForest (Chaining)")
            df_imp_full = run_ordered_missforest_chaining(df_full_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features, distance_matrix, connectivity_matrix, ordering='most_full_first')
            df_imp = df_imp_full.loc[df_eval_original.index]
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Ordered_MissForest'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
            model_plot_data['Ordered_MissForest'].append((gap, df_full_gapped, df_imp_full))
        except Exception as e: print(e)

        # 7. Natural MissForest
        try:
            print("\nMethod 7: Natural MissForest (Chaining)")
            df_imp_full = run_ordered_missforest_chaining(df_full_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features, distance_matrix, connectivity_matrix, ordering='none')
            df_imp = df_imp_full.loc[df_eval_original.index]
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Natural_MissForest'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
            model_plot_data['Natural_MissForest'].append((gap, df_full_gapped, df_imp_full))
        except Exception as e: print(e)

        all_results[f"{gap}_day_gap"] = gap_res

    # Plot Subplots at the end
    for method_name, data_list in model_plot_data.items():
        if not data_list: continue
        try:
            plot_selected_gaps_subplots(df_full_original, data_list, discharge_cols, out_dir, method_name)
            plot_overall_results_subplots(df_full_original, data_list, discharge_cols, out_dir, method_name)
        except Exception as e: 
            print(f"Failed to plot subplots for {method_name}: {e}")

    results_df = pd.DataFrame.from_dict({
        (g, m): v for g, d in all_results.items() for m, v in d.items()
    }, orient='index')
    results_df.index.names = ['Gap_Length', 'Method']

    csv_out = os.path.join(out_dir, "benchmark_1980_1990_results.csv")
    results_df.to_csv(csv_out)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print(results_df.round(4).to_string())
    print(f"Saved to {csv_out}")
    print("="*60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--window-years', type=int, default=3, help='Window size in years for training blocks')
    args = parser.parse_args()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_benchmark(test_mode=args.test_mode, window_years=args.window_years)
