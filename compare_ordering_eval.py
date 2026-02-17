import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    evaluate_metrics
)
from tiny_eval import create_contiguous_segment_gaps_by_percent, evaluate_imputation_performance
from ordered_missforest import OrderedMissForest

def run_ordering_comparison(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    start_date='1980-01-01',
    end_date='1985-12-31', # Shorter period for faster testing
    gap_length=30,          # Length of artificial gaps
    target_gap_pct=10.0     # % of data to remove for testing
):
    print("="*60)
    print("COMPARISON: Standard vs. Ordered (Most Full First) MissForest")
    print("="*60)

    # 1. Load Data
    df_discharge, df_contrib, df_coords, _, station_to_vcode = load_and_preprocess_data(
        discharge_path, lat_long_path, contrib_path
    )
    
    if df_discharge is None: return

    # Add temporal features
    df_with_features = add_temporal_features(df_discharge)
    
    # Filter to time period
    df_period = df_with_features.loc[start_date:end_date].copy()
    if df_period.empty:
        print("Error: No data in date range.")
        return

    # Identify columns
    all_cols = df_period.columns.tolist()
    discharge_cols = [c for c in all_cols if c in df_coords.index] # Valid stations only
    temporal_cols = [c for c in all_cols if c not in discharge_cols]
    
    # Build matrices
    dist_matrix = build_distance_matrix(df_coords, discharge_cols)
    conn_matrix = build_connectivity_matrix(df_contrib, discharge_cols, station_to_vcode)

    # 2. Create Artificial Gaps (Test Set)
    print(f"\nCreating {target_gap_pct}% artificial gaps (Length: {gap_length} days)...")
    df_gapped = create_contiguous_segment_gaps_by_percent(
        df_period, 
        discharge_cols, 
        gap_length=gap_length, 
        target_gap_percentage=target_gap_pct
    )
    
    # Verify missingness
    orig_nans = df_period[discharge_cols].isna().sum().sum()
    new_nans = df_gapped[discharge_cols].isna().sum().sum()
    print(f"Original NaNs: {orig_nans} | New NaNs (with artificial gaps): {new_nans}")

    # --- 3. Run Standard MissForest (Unordered/Alphabetical) ---
    print("\n" + "-"*40)
    print("RUNNING MODEL 1: Standard (Unordered)")
    print("-"*-40)
    
    model_standard = OrderedMissForest(
        distance_matrix=dist_matrix,
        connectivity=conn_matrix,
        max_iter=5, # Reduced for speed
        n_estimators=50,
        temporal_feature_columns=temporal_cols,
        initialization_method='historical_mean',
        ordering_method='none' # Standard behavior
    )
    
    model_standard.fit(df_gapped)
    # Note: fit() updates the internal data, but we usually call transform or inspect internal X
    # For evaluation, we look at the final state of the training data (since we imputed the training gaps)
    # In MissForest, fit() performs imputation on the training set.
    
    # We need to extract the final imputed dataframe from the object or transform again
    # Since fit() in CustomMissForest usually iterates on X and updates it, 
    # but doesn't store X as a public attribute, we will use transform() on the same data
    # OR modify the class to return the result. 
    # Standard MissForest logic in transform re-runs the iter loop. 
    # Let's just use transform() for consistency.
    df_imputed_standard = model_standard.transform(df_gapped)
    
    metrics_standard, _, _ = evaluate_imputation_performance(
        df_period, df_gapped, df_imputed_standard, discharge_cols
    )
    print(f"Standard Results -> NSE: {metrics_standard['NSE']:.4f}, KGE: {metrics_standard['KGE']:.4f}")

    # --- 4. Run Ordered MissForest (Most Full First) ---
    print("\n" + "-"*40)
    print("RUNNING MODEL 2: Ordered (Most Full First)")
    print("-"*-40)
    
    model_ordered = OrderedMissForest(
        distance_matrix=dist_matrix,
        connectivity=conn_matrix,
        max_iter=10,
        n_estimators=50,
        temporal_feature_columns=temporal_cols,
        initialization_method='historical_mean',
        ordering_method='most_full_first' # <--- THE KEY CHANGE
    )
    
    model_ordered.fit(df_gapped)
    df_imputed_ordered = model_ordered.transform(df_gapped)
    
    metrics_ordered, _, _ = evaluate_imputation_performance(
        df_period, df_gapped, df_imputed_ordered, discharge_cols
    )
    print(f"Ordered Results  -> NSE: {metrics_ordered['NSE']:.4f}, KGE: {metrics_ordered['KGE']:.4f}")

    # --- 5. Compare Results ---
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame([metrics_standard, metrics_ordered], 
                              index=['Standard (Unordered)', 'Ordered (Most Full First)'])
    
    print(results_df.round(4).to_string())
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"ordering_comparison_{ts}.csv")
    print(f"\nResults saved to ordering_comparison_{ts}.csv")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_ordering_comparison()