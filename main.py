# rolling_imputation.py
#
# This script implements a "seed-and-fill" imputation strategy
# focused *only* on the 1980-1990 period.

import pandas as pd
import numpy as np
import os
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    build_distance_matrix,
    build_connectivity_matrix,
    find_best_data_window 
)
from custom_missforest import CustomMissForest # Use the custom one
import warnings

warnings.filterwarnings('ignore')

def run_focused_imputation(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir="imputation_1980s_output",
    target_start_date='1980-01-01',
    target_end_date='1990-12-31',
    seed_window_days=1826 # Approx 5 years (this is now flexible)
):
    """
    Performs a focused "seed-and-fill" imputation on a target date range.
    
    1. Loads all data.
    2. Isolates the target period (e.g., 1980-1990).
    3. Finds the "best" N-day seed block *within* that period.
    4. Imputes the seed block to make it high-quality.
    5. Trains a final model on this high-quality seed.
    6. Uses this one model to impute the *entire* target period.
    7. Saves the final imputed target block.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Focused Imputation for {target_start_date} to {target_end_date} ---")

    # 1. Load all data (needed for matrices and full context)
    print("\n--- 1. Loading and preparing data ---")
    df_original, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    if df_original is None:
        print("Data loading failed. Exiting.")
        return
    
    df_full_features = add_temporal_features(df_original)
    
    all_cols = df_full_features.columns.tolist()
    discharge_cols = [col for col in all_cols if not (
                        col.startswith('day_of_year_') or 
                        col.startswith('month_') or 
                        col.startswith('week_of_year_'))]
    temporal_features = [col for col in all_cols if col not in discharge_cols]
    
    print(f"Full dataset loaded: {df_full_features.index.min().date()} to {df_full_features.index.max().date()}")

    # 2. Build helper matrices (using all stations)
    print("\n--- 2. Building helper matrices ---")
    all_stations = sorted(discharge_cols) 
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)

    # 3. Isolate Target Block and Find Seed Window
    print(f"\n--- 3. Finding {seed_window_days}-day seed window in {target_start_date} to {target_end_date} ---")
    
    # Get the original, gapped data for the target period
    df_target_block_original = df_full_features.loc[target_start_date:target_end_date].copy()
    if df_target_block_original.empty:
        print(f"No data found in the target range {target_start_date} to {target_end_date}. Exiting.")
        return

    try:
        # Find the best window *within* the target period
        seed_start, seed_end = find_best_data_window(
            df_target_block_original, # Search only within the target block
            discharge_cols, 
            target_start_date, 
            target_end_date, 
            window_size_days=seed_window_days
        )
    except Exception as e:
        print(f"Could not find seed window: {e}. Exiting.")
        return
    
    # This is our starting seed block
    df_seed_block = df_target_block_original.loc[seed_start:seed_end].copy()
    print(f"Found seed block: {seed_start.date()} to {seed_end.date()}")
    
    # 4. Impute the Seed Block to create a high-quality training set
    print("\n--- 4. Imputing seed block ---")
    model_seed = CustomMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        temporal_feature_columns=temporal_features,
        initialization_method='historical_mean' # Use a robust init
    )
    
    # Fit and transform the seed block
    model_seed.fit(df_seed_block)
    df_seed_imputed = model_seed.transform(df_seed_block)
    print("✓ Seed block imputed.")

    # 5. Train Final Model on the imputed seed block
    print("\n--- 5. Training final model on imputed seed block ---")
    model_final = CustomMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        temporal_feature_columns=temporal_features,
        initialization_method='historical_mean' # Use a robust init
    )
    
    # Fit the final model on the *clean* seed data
    model_final.fit(df_seed_imputed)
    print("✓ Final model trained.")

    # 6. Impute the ENTIRE Target Block (1980-1990)
    print(f"\n--- 6. Imputing entire target block ({target_start_date} to {target_end_date}) ---")
    
    # Use the final model to transform the *original* gapped target block
    df_final_imputed = model_final.transform(df_target_block_original)
    print("✓ Entire target block imputed.")

    # 7. Save final dataset
    output_path = os.path.join(output_dir, f"imputed_{target_start_date}_to_{target_end_date}.csv")
    df_final_imputed.to_csv(output_path)
    
    print("\n" + "="*50)
    print("FOCUSED IMPUTATION COMPLETE")
    print(f"Final imputed dataset saved to: {output_path}")
    print(f"Data covers: {df_final_imputed.index.min().date()} to {df_final_imputed.index.max().date()}")
    print("="*50)
    
    return df_final_imputed

if __name__ == '__main__':
    run_focused_imputation(
        target_start_date='1980-01-01',
        target_end_date='1990-12-31',
        seed_window_days=1826 # Approx 5 years
    )

