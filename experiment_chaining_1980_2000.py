# experiment_chaining_1980_2000.py
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Local imports
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    find_best_data_window,
    evaluate_metrics
)
from ordered_missforest import OrderedMissForest

# ==========================================
# CONFIGURATION
# ==========================================
START_YEAR = 1980
END_YEAR = 2000
SEED_WINDOW_YEARS = 3
GAP_PERCENTAGE = 10.0  # 10% of data will be removed as single random points
OUTPUT_DIR = "experiment_results_chaining"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def introduce_random_single_point_gaps(df, discharge_cols, pct=10.0, random_seed=42):
    """
    Randomly sets pct% of valid values in discharge columns to NaN.
    Returns: df_gapped, df_mask (True where artificial gap created)
    """
    np.random.seed(random_seed)
    df_gapped = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=discharge_cols)
    
    print(f"  Introducing {pct}% random individual gaps...")
    
    total_gaps = 0
    for col in discharge_cols:
        # valid indices are where data exists
        valid_indices = df[col].dropna().index
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            continue
            
        n_gaps = int(n_valid * (pct / 100.0))
        if n_gaps == 0: 
            continue
            
        # Randomly choose indices to drop
        gap_indices = np.random.choice(valid_indices, n_gaps, replace=False)
        
        df_gapped.loc[gap_indices, col] = np.nan
        mask.loc[gap_indices, col] = True
        total_gaps += n_gaps
        
    print(f"  Total gaps created: {total_gaps}")
    return df_gapped, mask

def run_flexible_chaining(
    df_full_gapped,
    seed_start,
    seed_end,
    discharge_cols,
    temporal_cols,
    dist_matrix,
    conn_matrix,
    ordering_method
):
    """
    Runs the chaining imputation (3-year rolling window) using OrderedMissForest.
    """
    print(f"\n--- Running Chaining (Ordering: {ordering_method}) ---")
    
    # 1. Factory for model creation
    def create_model():
        return OrderedMissForest(
            distance_matrix=dist_matrix,
            connectivity=conn_matrix,
            n_estimators=50, # Lower estimators for speed in experiment
            max_iter=5,
            temporal_feature_columns=temporal_cols,
            ordering_method=ordering_method,
            initialization_method='historical_mean'
        )

    # 2. Train Initial Seed
    print(f"  Training Seed: {seed_start.date()} to {seed_end.date()}")
    df_seed = df_full_gapped.loc[seed_start:seed_end].copy()
    
    model = create_model()
    model.fit(df_seed)
    df_seed_imp = model.transform(df_seed)
    
    imputed_blocks = [df_seed_imp]
    
    # 3. Chaining Setup
    window_days = (SEED_WINDOW_YEARS * 365)
    window_delta = pd.Timedelta(days=window_days)
    full_start = df_full_gapped.index.min()
    full_end = df_full_gapped.index.max()
    
    current_model = model

    # 4. Chain Backward
    print("  Chaining Backward...")
    curr_start = seed_start
    while curr_start > full_start:
        prev_end = curr_start - pd.Timedelta(days=1)
        prev_start = max(prev_end - window_delta, full_start)
        
        block_df = df_full_gapped.loc[prev_start:prev_end]
        if block_df.empty: break
            
        # Impute
        block_imp = current_model.transform(block_df)
        imputed_blocks.append(block_imp)
        
        # Retrain on newly imputed block
        # print(f"    Retraining on block: {prev_start.date()} to {prev_end.date()}")
        new_model = create_model()
        new_model.fit(block_imp)
        current_model = new_model
        
        curr_start = prev_start

    # 5. Chain Forward
    print("  Chaining Forward...")
    # Reset model to seed model for forward pass? 
    # Usually standard chaining continues updating. We'll restart from seed model for fairness.
    current_model = model 
    curr_end = seed_end
    while curr_end < full_end:
        next_start = curr_end + pd.Timedelta(days=1)
        next_end = min(next_start + window_delta, full_end)
        
        block_df = df_full_gapped.loc[next_start:next_end]
        if block_df.empty: break
            
        # Impute
        block_imp = current_model.transform(block_df)
        imputed_blocks.append(block_imp)
        
        # Retrain
        # print(f"    Retraining on block: {next_start.date()} to {next_end.date()}")
        new_model = create_model()
        new_model.fit(block_imp)
        current_model = new_model
        
        curr_end = next_end

    # Combine
    df_final = pd.concat(imputed_blocks).sort_index()
    # Handle overlaps if any (though logic above tries to avoid overlap)
    df_final = df_final[~df_final.index.duplicated(keep='first')]
    
    return df_final

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Data...")
    df, df_contrib, df_coords, _, s_to_v = load_and_preprocess_data(
        'discharge_data_cleaned.csv', 'lat_long_discharge.csv', 'mahanadi_contribs.csv'
    )
    if df is None: return

    df = add_temporal_features(df)
    
    # Filter 1980-2000
    start_dt = f"{START_YEAR}-01-01"
    end_dt = f"{END_YEAR}-12-31"
    df_period = df.loc[start_dt:end_dt].copy()
    
    print(f"Data Range: {start_dt} to {end_dt}")
    print(f"Rows: {len(df_period)}")

    # Columns
    discharge_cols = [c for c in df_period.columns if c in df_coords.index]
    temporal_cols = [c for c in df_period.columns if c not in discharge_cols]
    
    # Matrices
    dist_mat = build_distance_matrix(df_coords, discharge_cols)
    conn_mat = build_connectivity_matrix(df_contrib, discharge_cols, s_to_v)

    # 2. Find Seed Window (Best 3 Years)
    print("\nFinding Seed Window...")
    seed_days = (SEED_WINDOW_YEARS * 365) + 20
    seed_start, seed_end = find_best_data_window(
        df_period, discharge_cols, start_dt, end_dt, seed_days
    )
    print(f"Seed Window: {seed_start.date()} to {seed_end.date()}")

    # 3. Create Evaluation Dataset (Ground Truth vs Gapped)
    # We create gaps on the WHOLE period, but we will valid mask later
    df_truth = df_period.copy()
    
    # Create random individual gaps
    df_gapped, gap_mask = introduce_random_single_point_gaps(
        df_period, discharge_cols, pct=GAP_PERCENTAGE
    )
    
    # Ensure SEED window in df_gapped has minimal/no ARTIFICIAL gaps for training stability?
    # Usually we want the seed to be high quality.
    # Ideally, we revert the seed window in df_gapped to original (minus real missing)
    # to give the models a fair start.
    print("Restoring seed window data to original state (removing artificial gaps for seed)...")
    df_gapped.loc[seed_start:seed_end, discharge_cols] = df_truth.loc[seed_start:seed_end, discharge_cols]
    gap_mask.loc[seed_start:seed_end, discharge_cols] = False # Don't evaluate on seed

    # 4. Run Experiment
    results = {}
    
    # -- A. Standard (Unordered) --
    start_time = datetime.now()
    df_imp_std = run_flexible_chaining(
        df_gapped, seed_start, seed_end, discharge_cols, temporal_cols, 
        dist_mat, conn_mat, ordering_method='none'
    )
    
    # Evaluate only on the artificial gaps
    y_true_std = df_truth.values[gap_mask.values]
    y_pred_std = df_imp_std.loc[df_truth.index, discharge_cols].values[gap_mask.values]
    
    metrics_std = evaluate_metrics(y_true_std, y_pred_std)
    metrics_std['Runtime'] = (datetime.now() - start_time).seconds
    results['Standard (Unordered)'] = metrics_std
    print(f"Standard Results: NSE={metrics_std['NSE']:.3f}, KGE={metrics_std['KGE']:.3f}")

    # -- B. Ordered (Most Full First) --
    start_time = datetime.now()
    df_imp_ord = run_flexible_chaining(
        df_gapped, seed_start, seed_end, discharge_cols, temporal_cols, 
        dist_mat, conn_mat, ordering_method='most_full_first'
    )
    
    y_true_ord = df_truth.values[gap_mask.values]
    y_pred_ord = df_imp_ord.loc[df_truth.index, discharge_cols].values[gap_mask.values]
    
    metrics_ord = evaluate_metrics(y_true_ord, y_pred_ord)
    metrics_ord['Runtime'] = (datetime.now() - start_time).seconds
    results['Ordered (Most Full First)'] = metrics_ord
    print(f"Ordered Results:  NSE={metrics_ord['NSE']:.3f}, KGE={metrics_ord['KGE']:.3f}")

    # 5. Save & Display
    res_df = pd.DataFrame(results).T
    print("\n" + "="*40)
    print("FINAL COMPARISON (1980-2000 Chaining)")
    print("="*40)
    print(res_df)
    
    csv_path = os.path.join(OUTPUT_DIR, f"results_chaining_ordered_vs_std_{START_YEAR}_{END_YEAR}.csv")
    res_df.to_csv(csv_path)
    print(f"\nSaved to {csv_path}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()