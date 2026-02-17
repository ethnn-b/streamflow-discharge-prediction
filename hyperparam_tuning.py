import pandas as pd
import numpy as np
import time
import os
import warnings
from custom_missforest import CustomMissForest
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    create_contiguous_segment_gaps,
    evaluate_metrics
)

# --- CONFIGURATION ---
DISCHARGE_PATH = 'discharge_data_cleaned.csv'
LAT_LONG_PATH = 'lat_long_discharge.csv'
CONTRIB_PATH = 'mahanadi_contribs.csv'
OUTPUT_DIR = 'hyperparam_tuning_results'

# Data Split Dates
TRAIN_START = '1986-01-01'
TRAIN_END = '1990-12-31'
TEST_START = '1991-01-01' # Overlap slightly or start fresh
TEST_END = '1995-12-31'

# Experiment Settings
GAP_LENGTH_DAYS = 1         # Size of artificial gaps to create for testing
TARGET_GAP_PERCENT = 10      # Percentage of test data to remove
RANDOM_SEED = 42

# 1. Iteration Experiment Settings
ITERATION_VALUES = [1, 3, 5, 10, 15, 20]
FIXED_ESTIMATORS_FOR_ITER = 50 # Keep trees lower to speed up iteration test

# 2. Regressor Experiment Settings
ESTIMATOR_VALUES = [10, 50, 100, 200, 300]
FIXED_ITER_FOR_EST = 5         # Keep iterations fixed for this test

def prepare_data():
    """Loads, splits, and gaps the data."""
    print("--- 1. Loading Data ---")
    df_all, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(DISCHARGE_PATH, LAT_LONG_PATH, CONTRIB_PATH)
    
    if df_all is None:
        raise ValueError("Data loading failed.")

    df_with_features = add_temporal_features(df_all)
    
    # Define Column Sets
    all_cols = df_with_features.columns.tolist()
    discharge_cols = [c for c in all_cols if not (c.startswith('day_') or c.startswith('month_') or c.startswith('week_'))]
    temporal_cols = [c for c in all_cols if c not in discharge_cols]
    
    # Build Matrices
    dist_matrix = build_distance_matrix(df_coords, discharge_cols)
    conn_matrix = build_connectivity_matrix(df_contrib, discharge_cols, station_to_vcode)

    # Split Data
    print(f"--- 2. Splitting Data ---")
    print(f"    Training (Seed): {TRAIN_START} to {TRAIN_END}")
    print(f"    Testing (Eval):  {TEST_START} to {TEST_END}")
    
    df_train = df_with_features.loc[TRAIN_START:TRAIN_END].copy()
    df_test_original = df_with_features.loc[TEST_START:TEST_END].copy()
    
    # Create Gaps in Test Data
    print(f"--- 3. Creating Artificial Gaps in Test Data ({GAP_LENGTH_DAYS} days, ~{TARGET_GAP_PERCENT}%) ---")
    
    # Calculate num_intervals roughly based on target percent
    data_len = len(df_test_original)
    target_points = data_len * (TARGET_GAP_PERCENT / 100.0)
    num_intervals = int(target_points / GAP_LENGTH_DAYS)
    if num_intervals < 1: num_intervals = 1
    
    # Use the utility to create gaps
    gap_results = create_contiguous_segment_gaps(
        df_test_original, 
        discharge_cols, 
        gap_lengths=[GAP_LENGTH_DAYS], 
        num_intervals_per_column=num_intervals
    )
    
    df_test_gapped = gap_results[GAP_LENGTH_DAYS]['gapped_data']
    
    return {
        'train': df_train,
        'test_orig': df_test_original,
        'test_gapped': df_test_gapped,
        'discharge_cols': discharge_cols,
        'temporal_cols': temporal_cols,
        'dist_matrix': dist_matrix,
        'conn_matrix': conn_matrix
    }

def run_experiment_iterations(data):
    """Varies max_iter while keeping n_estimators constant."""
    results = []
    print("\n" + "="*60)
    print(f"RUNNING EXPERIMENT 1: OPTIMIZING ITERATIONS")
    print(f"Fixed n_estimators: {FIXED_ESTIMATORS_FOR_ITER}")
    print("="*60)

    for max_iter in ITERATION_VALUES:
        print(f"\nTesting max_iter = {max_iter}...")
        
        start_time = time.time()
        
        # Initialize
        model = CustomMissForest(
            distance_matrix=data['dist_matrix'],
            connectivity=data['conn_matrix'],
            max_iter=max_iter,
            n_estimators=FIXED_ESTIMATORS_FOR_ITER, # Fixed
            random_state=RANDOM_SEED,
            temporal_feature_columns=data['temporal_cols'],
            initialization_method='historical_mean'
        )
        
        # Fit on Training Data
        model.fit(data['train'])
        
        # Transform Test Data
        df_imputed = model.transform(data['test_gapped'])
        
        elapsed_time = time.time() - start_time
        
        # Evaluate
        # We only evaluate on the specific gaps we created
        y_true_all, y_pred_all = [], []
        
        for col in data['discharge_cols']:
            if col not in df_imputed.columns: continue
            
            # Find mask where we created gaps (NaN in gapped, Not NaN in original)
            mask = data['test_gapped'][col].isna() & data['test_orig'][col].notna()
            
            if mask.sum() > 0:
                y_true_all.extend(data['test_orig'].loc[mask, col].values)
                y_pred_all.extend(df_imputed.loc[mask, col].values)

        metrics = evaluate_metrics(np.array(y_true_all), np.array(y_pred_all))
        
        # Record Results
        res_row = {
            'max_iter': max_iter,
            'time_sec': round(elapsed_time, 2),
            'KGE': metrics['KGE'],
            'NSE': metrics['NSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        }
        results.append(res_row)
        print(f"  -> Time: {elapsed_time:.2f}s | KGE: {metrics['KGE']:.4f} | NSE: {metrics['NSE']:.4f}")

    return pd.DataFrame(results)

def run_experiment_regressors(data):
    """Varies n_estimators while keeping max_iter constant."""
    results = []
    print("\n" + "="*60)
    print(f"RUNNING EXPERIMENT 2: OPTIMIZING REGRESSORS (TREES)")
    print(f"Fixed max_iter: {FIXED_ITER_FOR_EST}")
    print("="*60)

    for n_est in ESTIMATOR_VALUES:
        print(f"\nTesting n_estimators = {n_est}...")
        
        start_time = time.time()
        
        # Initialize
        model = CustomMissForest(
            distance_matrix=data['dist_matrix'],
            connectivity=data['conn_matrix'],
            max_iter=FIXED_ITER_FOR_EST, # Fixed
            n_estimators=n_est,
            random_state=RANDOM_SEED,
            temporal_feature_columns=data['temporal_cols'],
            initialization_method='historical_mean'
        )
        
        # Fit on Training Data
        model.fit(data['train'])
        
        # Transform Test Data
        df_imputed = model.transform(data['test_gapped'])
        
        elapsed_time = time.time() - start_time
        
        # Evaluate
        y_true_all, y_pred_all = [], []
        for col in data['discharge_cols']:
            if col not in df_imputed.columns: continue
            mask = data['test_gapped'][col].isna() & data['test_orig'][col].notna()
            if mask.sum() > 0:
                y_true_all.extend(data['test_orig'].loc[mask, col].values)
                y_pred_all.extend(df_imputed.loc[mask, col].values)

        metrics = evaluate_metrics(np.array(y_true_all), np.array(y_pred_all))
        
        # Record Results
        res_row = {
            'n_estimators': n_est,
            'time_sec': round(elapsed_time, 2),
            'KGE': metrics['KGE'],
            'NSE': metrics['NSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        }
        results.append(res_row)
        print(f"  -> Time: {elapsed_time:.2f}s | KGE: {metrics['KGE']:.4f} | NSE: {metrics['NSE']:.4f}")

    return pd.DataFrame(results)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare Data
    try:
        data = prepare_data()
    except Exception as e:
        print(f"Error preparing data: {e}")
        return

    # 2. Run Iteration Experiment
    df_iter_results = run_experiment_iterations(data)
    iter_csv_path = os.path.join(OUTPUT_DIR, 'experiment_1_iterations.csv')
    df_iter_results.to_csv(iter_csv_path, index=False)
    
    # 3. Run Regressor Experiment
    df_est_results = run_experiment_regressors(data)
    est_csv_path = os.path.join(OUTPUT_DIR, 'experiment_2_regressors.csv')
    df_est_results.to_csv(est_csv_path, index=False)

    # 4. Analysis & Summary
    print("\n" + "="*60)
    print("FINAL ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n--- Experiment 1 Results (Iterations) saved to {iter_csv_path} ---")
    print(df_iter_results.to_string(index=False))
    
    # Find optimal iteration (Tradeoff between Time and KGE)
    # Simple heuristic: Stop if KGE improvement < 0.001 per extra second, or just max KGE
    best_iter_row = df_iter_results.loc[df_iter_results['KGE'].idxmax()]
    print(f"\n[Recommendation] Best Max Iterations based on KGE: {int(best_iter_row['max_iter'])}")
    
    print(f"\n--- Experiment 2 Results (Regressors) saved to {est_csv_path} ---")
    print(df_est_results.to_string(index=False))
    
    # Find optimal estimators
    best_est_row = df_est_results.loc[df_est_results['KGE'].idxmax()]
    print(f"\n[Recommendation] Best N_Estimators based on KGE: {int(best_est_row['n_estimators'])}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()