import os
import random
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.data import (
    load_and_preprocess_data,
    add_temporal_features,
    find_best_data_window,
    create_contiguous_segment_gaps_by_percent
)
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix
from src.utils.metrics import evaluate_metrics

from src.imputers.baselines import historical_mean_imputation, linear_interpolation_imputation
from src.imputers.kalman import kalman_imputation
from src.imputers.knn import knn_imputation
from src.imputers.missforest import MissForestImputer


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


def run_ordered_missforest_chaining(df_full_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features, distance_matrix, connectivity_matrix):
    print("  Starting 3-year *chaining* ordered MissForest process...")
    
    def create_model():
        return MissForestImputer(
            distance_matrix=distance_matrix,
            connectivity=connectivity_matrix,
            max_iter=10,
            n_estimators=100,
            random_state=42,
            distance_weighting_type='inverse',
            temporal_feature_columns=temporal_features,
            initialization_method='historical_mean',
            ordering_method='most_full_first'
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
            df_imp_full = run_ordered_missforest_chaining(df_full_gapped, seed_start, seed_end, window_years, discharge_cols, temporal_features, distance_matrix, connectivity_matrix)
            df_imp = df_imp_full.loc[df_eval_original.index]
            metrics, _, _ = evaluate_imputation_performance(df_eval_original, df_eval_gapped, df_imp, discharge_cols)
            gap_res['Ordered_MissForest'] = metrics
            print(f"KGE: {metrics['KGE']:.4f}")
        except Exception as e: print(e)

        all_results[f"{gap}_day_gap"] = gap_res

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
