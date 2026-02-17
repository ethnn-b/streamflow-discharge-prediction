# evaluation_1980s.py
#
# This script performs a focused "seed-and-fill" evaluation *within*
# the 1980-1990 period.
#
# NEW LOGIC (as of 2025-11-08):
# 1. Load 1980-1990 data. Find 3-year "seed" block.
# 2. Define "eval" block as everything *except* the seed.
# 3. Loop through gap lengths: [3, 7, 30, 100] days.
# 4. Inside loop:
#    a. Create a fresh `df_eval_gapped` with a 10% target for the current gap length.
#    b. Run all 9 benchmarks (Chaining, Expanding, Seed-Only, MICE, KNN, etc.)
#       using this new `df_eval_gapped`.
#    c. Store metrics for this gap length.
# 5. After loop, create a final summary table comparing all methods across all gap lengths.
#

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import warnings
from datetime import datetime

# Import utility functions
from simplified_utils import (
    load_and_preprocess_data,
    add_temporal_features,
    # create_contiguous_segment_gaps is NOT imported, we define a new one
    build_distance_matrix,
    build_connectivity_matrix,
    historical_mean_imputation,
    find_best_data_window, # Use the 5-arg version
    evaluate_metrics # The base metric calculator
)
# Import your custom model
from custom_missforest import CustomMissForest
# Import the base imputer to test "vanilla" mode
from missforest_imputer import ModifiedMissForest 

# Import sklearn models
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# Import statsmodels for Kalman filter
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    print("WARNING: statsmodels not found. Kalman (SARIMAX) benchmark will be skipped.")
    SARIMAX = None

# ---
# LOCAL HELPER FUNCTIONS (for this script)
# ---

def create_contiguous_segment_gaps_by_percent(df_data, discharge_cols, gap_length, target_gap_percentage, random_seed=42):
    """
    Creates contiguous gaps in discharge data for evaluation, based on a
    target percentage of missing data per column.
    
    Args:
        df_data: The DataFrame to introduce gaps into.
        discharge_cols: List of columns to create gaps in.
        gap_length: The length of each individual gap (e.g., 3, 7, 30).
        target_gap_percentage: The target % of data to turn into gaps for each column.
        random_seed: Seed for reproducibility.
        
    Returns:
        df_gapped: DataFrame with new artificial gaps.
    """
    np.random.seed(random_seed)
    df_gapped = df_data.copy()
    data_length = len(df_gapped)
    
    if gap_length <= 0:
        return df_gapped
        
    # Calculate how many intervals are needed *per column* for the target percentage
    target_points_per_col = data_length * (target_gap_percentage / 100.0)
    num_intervals_per_column = int(np.floor(target_points_per_col / gap_length))
    
    if num_intervals_per_column == 0 and target_gap_percentage > 0:
        num_intervals_per_column = 1 # Ensure at least one gap if percent > 0

    print(f"  Creating gaps: length={gap_length}, target={target_gap_percentage}% -> {num_intervals_per_column} intervals per column.")

    for target_column in discharge_cols:
        if target_column not in df_gapped.columns:
            continue
            
        if gap_length >= data_length:
            df_gapped[target_column] = np.nan
        else:
            possible_starts = np.arange(data_length - gap_length + 1)
            np.random.shuffle(possible_starts)

            gaps_to_apply = []
            for start_candidate in possible_starts:
                end_candidate = start_candidate + gap_length
                
                # Check for overlap
                is_overlapping = False
                for existing_start, existing_end in gaps_to_apply:
                    if not (end_candidate <= existing_start or start_candidate >= existing_end):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    gaps_to_apply.append((start_candidate, end_candidate))
                    if len(gaps_to_apply) == num_intervals_per_column:
                        break
            
            # Apply gaps
            for start_idx, end_idx in gaps_to_apply:
                df_gapped.iloc[start_idx:end_idx, df_gapped.columns.get_loc(target_column)] = np.nan
    
    return df_gapped


def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    """
    Evaluate imputation performance by comparing true values with imputed values
    at the locations where artificial gaps were created.
    
    Returns: (metrics_dict, y_true_list, y_pred_list)
    """
    y_true_eval, y_pred_eval = [], []
    for station in discharge_cols:
        if station not in df_imputed.columns or station not in df_gapped.columns or station not in df_original.columns:
            continue
        
        # Find where the artificial gaps were created
        # We only evaluate on gaps in df_gapped that have data in df_original
        gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
        
        if gap_mask.sum() > 0:
            # Get the predicted values from our model at these specific gap locations
            # We must index df_imputed by the gap_mask's index
            # Use .loc to ensure index alignment
            predicted_vals = df_imputed.loc[gap_mask.index[gap_mask], station].values
            
            # Get the ground truth values from the original, non-gapped test data
            true_vals = df_original.loc[gap_mask.index[gap_mask], station].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)

    if y_true_eval:
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics, y_true_eval, y_pred_eval
    else:
        # No artificial gaps were created or evaluable
        print("Warning: No evaluable artificial gaps found.")
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}, [], []

def plot_imputation_results(df_original, df_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length):
    """Generates and saves plots to visualize imputation results for a specific method."""
    print(f"  Generating visualizations for {method_name} (Gap: {gap_length}d)...")
    
    # Create a subdirectory for the gap length, then for the method
    method_plot_dir = os.path.join(output_dir, f"gap_{gap_length}d", f"plots_{method_name}")
    os.makedirs(method_plot_dir, exist_ok=True)
    
    # 1. Scatter plot of True vs. Predicted (for artificial gaps)
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', label=f'Imputed Values (Gaps: {gap_length}d)')
        if y_true and y_pred:
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line (Perfect Fit)')
        plt.title(f'True vs. Predicted (Gaps: {gap_length}d) - {method_name}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.legend()
        scatter_path = os.path.join(method_plot_dir, f"{method_name}_true_vs_predicted_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
    except Exception as e:
        print(f"  WARNING: Could not generate scatter plot for {method_name}. Error: {e}")

    # 2. Time series plots for a few stations
    try:
        stations_with_gaps = df_gapped.columns[df_gapped.isnull().any()].tolist()
        stations_in_original = [s for s in stations_with_gaps if s in df_original.columns]
        
        if not stations_in_original:
            print("  WARNING: No stations with gaps found for time series plots.")
            return
            
        stations_to_plot = random.sample(stations_in_original, min(len(stations_in_original), 3))
        
        for station in stations_to_plot:
            if station not in df_imputed.columns or station not in df_original.columns:
                continue
            
            plt.figure(figsize=(15, 6))
            plt.plot(df_original.index, df_original[station], color='cornflowerblue', label='Original Data (Ground Truth)', zorder=1)
            gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
            
            if gap_mask.sum() > 0:
                plt.scatter(
                    df_imputed.index[gap_mask], 
                    df_imputed.loc[gap_mask.index[gap_mask], station], 
                    color='red', 
                    marker='o',
                    s=50,
                    label=f'Imputed Values ({method_name})', 
                    zorder=2
                )
            
            plt.title(f'Imputation Results for Station: {station} (Gaps: {gap_length}d)')
            plt.xlabel('Date')
            plt.ylabel('Discharge')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plot_path = os.path.join(method_plot_dir, f"{station}_{method_name}_imputation_plot.png")
            plt.savefig(plot_path)
            plt.close()
        print(f"  ✓ Time series plots saved for {method_name}.")
    except Exception as e:
        print(f"  WARNING: Could not generate time series plots for {method_name}. Error: {e}")


# ---
# BENCHMARK MODEL FUNCTIONS (local)
# ---

def scale_and_impute_sklearn(imputer, df_seed_gapped, df_eval_gapped, all_cols):
    """
    Helper function to correctly scale and impute data for sklearn models.
    Handles shape mismatches from all-NA columns.
    """
    print("  Scaling data...")
    df_seed = df_seed_gapped[all_cols].copy()
    df_eval = df_eval_gapped[all_cols].copy()
    
    seed_mask = df_seed.isnull()
    eval_mask = df_eval.isnull()

    col_means_seed = df_seed.mean()
    seed_cols_all_na = col_means_seed[col_means_seed.isna()].index
    col_means_seed[seed_cols_all_na] = 0.0 
    
    df_seed_filled = df_seed.fillna(col_means_seed)
    df_eval_filled = df_eval.fillna(col_means_seed) 
    
    eval_cols_still_na = df_eval_filled.columns[df_eval_filled.isnull().any()]
    if not eval_cols_still_na.empty:
         print(f"  Warning: Eval set columns still have NaNs after fill: {eval_cols_still_na.tolist()}. Filling with 0.")
         df_eval_filled[eval_cols_still_na] = df_eval_filled[eval_cols_still_na].fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(df_seed_filled)

    df_seed_scaled_values = scaler.transform(df_seed_filled)
    df_eval_scaled_values = scaler.transform(df_eval_filled)
    
    df_seed_scaled = pd.DataFrame(df_seed_scaled_values, columns=all_cols, index=df_seed.index)
    df_eval_scaled = pd.DataFrame(df_eval_scaled_values, columns=all_cols, index=df_eval.index)
    
    df_seed_scaled_with_nans = df_seed_scaled.mask(seed_mask)
    df_eval_scaled_with_nans = df_eval_scaled.mask(eval_mask)

    seed_all_nan_after_mask = df_seed_scaled_with_nans.columns[df_seed_scaled_with_nans.isnull().all()]
    eval_all_nan_after_mask = df_eval_scaled_with_nans.columns[df_eval_scaled_with_nans.isnull().all()]
    
    if not seed_all_nan_after_mask.empty:
        df_seed_scaled_with_nans[seed_all_nan_after_mask] = df_seed_scaled[seed_all_nan_after_mask]
        
    if not eval_all_nan_after_mask.empty:
        df_eval_scaled_with_nans[eval_all_nan_after_mask] = df_eval_scaled[eval_all_nan_after_mask]

    print(f"  Fitting imputer ({imputer.__class__.__name__}) on scaled seed block...")
    imputer.fit(df_seed_scaled_with_nans)
    
    print("  Transforming scaled evaluation block...")
    imputed_eval_scaled_values = imputer.transform(df_eval_scaled_with_nans)

    print("  Inverse transforming scaled data...")
    if imputed_eval_scaled_values.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Imputer output shape ({imputed_eval_scaled_values.shape[1]}) does not match scaler input shape ({scaler.n_features_in_})")
        
    imputed_eval_original_scale = scaler.inverse_transform(imputed_eval_scaled_values)
    
    df_imputed = pd.DataFrame(imputed_eval_original_scale, 
                              columns=all_cols, 
                              index=df_eval.index)
                              
    return df_imputed


def linear_interpolation_imputation(df_gapped, discharge_cols):
    """Imputes missing values using time-based linear interpolation."""
    print(f"\n--- Running Linear Interpolation ---")
    df_imputed = df_gapped.copy()
    
    for col in discharge_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].interpolate(method='time')
            df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')
            
    print("✓ Linear interpolation complete.")
    return df_imputed

def kalman_imputation(df_seed_gapped, df_eval_gapped, discharge_cols):
    """Imputes missing values using SARIMAX with a Kalman filter."""
    if SARIMAX is None:
        print("  Skipping Kalman (SARIMAX): statsmodels not installed.")
        return df_eval_gapped.copy()

    print(f"\n--- Running Kalman (SARIMAX) Imputation ---")
    
    df_combined_gapped = pd.concat([df_seed_gapped, df_eval_gapped])
    df_imputed = df_combined_gapped.copy()

    for col in discharge_cols:
        print(f"  Kalman: Processing {col}...")
        
        col_mean = df_combined_gapped[col].mean()
        if pd.isna(col_mean):
            col_mean = 0.0

        try:
            model = SARIMAX(
                df_combined_gapped[col],
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            imputed_series = res.predict(start=df_combined_gapped.index[0], end=df_combined_gapped.index[-1])
            df_imputed[col] = df_imputed[col].fillna(imputed_series)

        except Exception as e:
            print(f"  WARNING: SARIMAX failed for {col}: {e}. Falling back to column mean.")
            df_imputed[col] = df_imputed[col].fillna(col_mean)

    return df_imputed.loc[df_eval_gapped.index]

# ---
# BENCHMARK FUNCTION 1: "CHAINING" (Train A -> Impute B; Train B -> Impute C)
# ---

def run_chaining_imputation_benchmark(
    df_full_gapped, 
    seed_start_date, 
    seed_end_date, 
    window_years,
    discharge_cols, 
    temporal_features, 
    distance_matrix, 
    connectivity_matrix
):
    """
    Performs the "chaining" window imputation process.
    """
    print("  Starting 3-year *chaining* imputation process...")
    
    def create_model():
        return CustomMissForest(
            distance_matrix=distance_matrix,
            connectivity=connectivity_matrix,
            max_iter=10,
            n_estimators=100,
            random_state=42,
            distance_weighting_type='inverse',
            temporal_feature_columns=temporal_features,
            initialization_method='historical_mean'
        )

    print(f"  Training initial model on seed: {seed_start_date.date()} to {seed_end_date.date()}")
    df_seed_gapped = df_full_gapped.loc[seed_start_date:seed_end_date].copy()
    
    initial_model = create_model()
    initial_model.fit(df_seed_gapped)
    df_seed_imputed = initial_model.transform(df_seed_gapped)
    
    all_imputed_blocks = [df_seed_imputed]
    
    full_start_date = df_full_gapped.index.min()
    full_end_date = df_full_gapped.index.max()
    window_size_days = (window_years * 365) + (window_years // 4)
    window_timedelta = pd.Timedelta(days=window_size_days)

    print("\n  Chaining backwards from seed...")
    current_training_model = initial_model
    current_block_start_date = seed_start_date
    
    while current_block_start_date > full_start_date:
        prev_block_end = current_block_start_date - pd.Timedelta(days=1)
        prev_block_start = max(prev_block_end - window_timedelta, full_start_date)
        
        df_prev_gapped = df_full_gapped.loc[prev_block_start:prev_block_end]
        if df_prev_gapped.empty:
            break
            
        print(f"    Imputing backwards block: {prev_block_start.date()} to {prev_block_end.date()}")
        df_prev_imputed = current_training_model.transform(df_prev_gapped)
        all_imputed_blocks.append(df_prev_imputed)
        
        print(f"    Training new model *only* on: {prev_block_start.date()} to {prev_block_end.date()}")
        new_model = create_model()
        new_model.fit(df_prev_imputed)
        
        current_training_model = new_model
        current_block_start_date = prev_block_start

    print("\n  Chaining forwards from seed...")
    current_training_model = initial_model 
    current_block_end_date = seed_end_date
    
    while current_block_end_date < full_end_date:
        next_block_start = current_block_end_date + pd.Timedelta(days=1)
        next_block_end = min(next_block_start + window_timedelta, full_end_date)
        
        df_next_gapped = df_full_gapped.loc[next_block_start:next_block_end]
        if df_next_gapped.empty:
            break
            
        print(f"    Imputing forwards block: {next_block_start.date()} to {next_block_end.date()}")
        df_next_imputed = current_training_model.transform(df_next_gapped)
        all_imputed_blocks.append(df_next_imputed)

        print(f"    Training new model *only* on: {next_block_start.date()} to {next_block_end.date()}")
        new_model = create_model()
        new_model.fit(df_next_imputed)
        
        current_training_model = new_model
        current_block_end_date = next_block_end
            
    print("\n  ✓ Rolling/Chaining imputation complete.")
    df_final_imputed = pd.concat(all_imputed_blocks)
    return df_final_imputed.sort_index()


# ---
# BENCHMARK FUNCTION 2: "EXPANDING" (Train A -> Impute B; Train A+B -> Impute C)
# ---

def run_expanding_imputation_benchmark(
    df_full_gapped, 
    seed_start_date, 
    seed_end_date, 
    window_years,
    discharge_cols, 
    temporal_features, 
    distance_matrix, 
    connectivity_matrix
):
    """
    Performs the "expanding" window imputation process.
    """
    print("  Starting 3-year *expanding* imputation process...")
    
    model = CustomMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features,
        initialization_method='historical_mean'
    )

    print(f"  Training initial model on seed: {seed_start_date.date()} to {seed_end_date.date()}")
    df_seed_gapped = df_full_gapped.loc[seed_start_date:seed_end_date].copy()
    
    model.fit(df_seed_gapped)
    df_seed_imputed = model.transform(df_seed_gapped)
    
    all_imputed_blocks = [df_seed_imputed]
    df_known_data_pool = [df_seed_imputed] 
    
    full_start_date = df_full_gapped.index.min()
    full_end_date = df_full_gapped.index.max()
    window_size_days = (window_years * 365) + (window_years // 4)
    window_timedelta = pd.Timedelta(days=window_size_days)
    
    current_start_date = seed_start_date
    current_end_date = seed_end_date
    
    iteration = 0
    while current_start_date > full_start_date or current_end_date < full_end_date:
        iteration += 1
        print(f"  Expanding Imputation Iteration {iteration}...")
        
        df_training_pool = pd.concat(df_known_data_pool).sort_index()
        print(f"    Retraining model on {len(df_training_pool)} rows ({df_training_pool.index.min().date()} to {df_training_pool.index.max().date()})")
        model.fit(df_training_pool)

        new_data_added = False
        
        if current_start_date > full_start_date:
            prev_block_end = current_start_date - pd.Timedelta(days=1)
            prev_block_start = max(prev_block_end - window_timedelta, full_start_date)
            
            df_prev_gapped = df_full_gapped.loc[prev_block_start:prev_block_end]
            
            if not df_prev_gapped.empty:
                print(f"    Imputing backwards block: {prev_block_start.date()} to {prev_block_end.date()}")
                df_prev_imputed = model.transform(df_prev_gapped)
                
                all_imputed_blocks.append(df_prev_imputed)
                df_known_data_pool.append(df_prev_imputed)
                
                current_start_date = prev_block_start
                new_data_added = True

        if current_end_date < full_end_date:
            next_block_start = current_end_date + pd.Timedelta(days=1)
            next_block_end = min(next_block_start + window_timedelta, full_end_date)

            df_next_gapped = df_full_gapped.loc[next_block_start:next_block_end]

            if not df_next_gapped.empty:
                print(f"    Imputing forwards block: {next_block_start.date()} to {next_block_end.date()}")
                df_next_imputed = model.transform(df_next_gapped)
                
                all_imputed_blocks.append(df_next_imputed)
                df_known_data_pool.append(df_next_imputed)

                current_end_date = next_block_end
                new_data_added = True

        if not new_data_added:
            print("  No new data was added in either direction. Stopping.")
            break
            
    print("\n  ✓ Expanding imputation complete.")
    df_final_imputed = pd.concat(all_imputed_blocks)
    return df_final_imputed.sort_index()


# ---
# MAIN EVALUATION SCRIPT
# ---

def run_focused_evaluation(
    discharge_path='discharge_data_cleaned.csv',
    lat_long_path='lat_long_discharge.csv',
    contrib_path='mahanadi_contribs.csv',
    output_dir_base="evaluation_1980s_output",
    search_start_date='1980-01-01',
    search_end_date='1990-12-31',
    seed_window_years=3
):
    """
    Main function to run the "seed-and-fill" evaluation.
    """
    
    # --- 0. Setup ---
    start_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{output_dir_base}_{start_time_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # NEW: Define gap lengths and target percentage
    gap_lengths_to_test = [3, 7, 30, 100]
    target_gap_percentage = 10.0
    
    print("="*60)
    print(f"Starting Focused Evaluation (1980-1990)")
    print(f"Comparing all methods across gap lengths: {gap_lengths_to_test}")
    print(f"Using 3-Year Seed and {target_gap_percentage}% target gaps")
    print(f"Output Directory: {output_dir}")
    print("="*60)
    
    # --- 1. Load and Filter Data ---
    print("\n--- 1. Loading and preparing all data ---")
    df_original_all, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(discharge_path, lat_long_path, contrib_path)
    if df_original_all is None:
        print("Data loading failed. Exiting.")
        return
        
    df_with_features = add_temporal_features(df_original_all)
    
    print(f"Filtering data to {search_start_date} - {search_end_date} period...")
    df_full_period_original = df_with_features.loc[search_start_date:search_end_date].copy()
    
    if df_full_period_original.empty:
        print(f"No data found in the specified {search_start_date} to {search_end_date} period. Exiting.")
        return
            
    print(f"Loaded and filtered data from {df_full_period_original.index.min().date()} to {df_full_period_original.index.max().date()}")

    # --- 2. Define Columns and Build Matrices ---
    print("\n--- 2. Defining columns and building matrices ---")
    all_cols_in_data = df_full_period_original.columns.tolist()
    discharge_cols = [col for col in all_cols_in_data if not (
                        col.startswith('day_of_year_') or 
                        col.startswith('month_') or 
                        col.startswith('week_of_year_'))]
    temporal_features = [col for col in all_cols_in_data if col not in discharge_cols]
    
    print(f"Found {len(discharge_cols)} discharge columns (stations).")
    print(f"Found {len(temporal_features)} temporal features.")

    all_stations = sorted(discharge_cols) 
    distance_matrix = build_distance_matrix(df_coords, all_stations)
    distance_matrix = distance_matrix.loc[all_stations, all_stations]
    connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode)
    connectivity_matrix = connectivity_matrix.loc[all_stations, all_stations]

    # --- 3. Find Seed & Evaluation Blocks (Done ONCE) ---
    print(f"\n--- 3. Finding {seed_window_years}-year seed block ---")
    seed_window_days = (seed_window_years * 365) + (seed_window_years // 4)
    
    try:
        seed_start_date, seed_end_date = find_best_data_window(
            df=df_full_period_original,
            discharge_cols=discharge_cols,
            start_date_str=search_start_date,
            end_date_str=search_end_date,
            window_size_days=seed_window_days
        )
    except ValueError as e:
        print(f"ERROR: Could not find seed window: {e}")
        return

    # This is the "training" data for the *other* benchmarks
    df_seed_gapped = df_full_period_original.loc[seed_start_date:seed_end_date].copy()
    
    # The "evaluation" data is everything *outside* this seed block
    eval_mask = ~df_full_period_original.index.isin(df_seed_gapped.index)
    # This is the ground truth for the evaluation period
    df_eval_original = df_full_period_original.loc[eval_mask].copy()
    
    print(f"Seed block (for training): {df_seed_gapped.index.min().date()} to {df_seed_gapped.index.max().date()} ({len(df_seed_gapped)} rows)")
    print(f"Eval block (for testing):  {len(df_eval_original)} rows (all data *except* seed block)")
    if df_eval_original.empty:
        print("ERROR: Evaluation block is empty. Cannot proceed.")
        return


    # --- 4. START EVALUATION LOOP (Looping over Gaps) ---
    print("\n" + "="*60)
    print("--- STARTING EVALUATION LOOP FOR ALL GAP LENGTHS ---")
    print("="*60)
    
    all_results = {} # This will be a nested dictionary

    for gap_length in gap_lengths_to_test:
        
        print(f"\n{'-'*20} EVALUATING FOR {gap_length}-DAY GAPS (Target: {target_gap_percentage}%) {'-'*20}")
        
        # This dict will hold results for this gap length
        gap_results_for_this_length = {}

        # --- 4a. Create Artificial Gaps ---
        print(f"--- 4a. Creating artificial {gap_length}-day gaps ---")
        
        df_eval_gapped = create_contiguous_segment_gaps_by_percent(
            df_eval_original, # The original, gappy eval data
            discharge_cols, 
            gap_length=gap_length,
            target_gap_percentage=target_gap_percentage
        )
        
        print(f"  Total artificial gaps created: {df_eval_gapped[discharge_cols].isnull().sum().sum() - df_eval_original[discharge_cols].isnull().sum().sum()}")
        
        print("  Combining seed and gapped eval blocks for imputers...")
        df_full_period_with_artificial_gaps = pd.concat([df_seed_gapped, df_eval_gapped]).sort_index()


        # --- 4b. Run All Benchmarks (for this gap length) ---
        print(f"\n--- 4b. Running All Imputation Benchmarks (Gap: {gap_length}d) ---")
        
        # --- Method 1: CustomMissForest (3-Year Chaining) ---
        print("\n--- Testing CustomMissForest (3-Year Chaining) ---")
        try:
            df_imputed_full_period = run_chaining_imputation_benchmark(
                df_full_gapped=df_full_period_with_artificial_gaps, 
                seed_start_date=seed_start_date,
                seed_end_date=seed_end_date,
                window_years=seed_window_years,
                discharge_cols=discharge_cols,
                temporal_features=temporal_features,
                distance_matrix=distance_matrix,
                connectivity_matrix=connectivity_matrix
            )
            print("  Extracting evaluation period from chaining imputation result...")
            df_imputed = df_imputed_full_period.loc[df_eval_original.index]
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'CustomMissForest_3Yr_Chaining'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"CustomMissForest Chaining failed: {e}")
            gap_results_for_this_length['CustomMissForest_3Yr_Chaining'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 2: CustomMissForest (3-Year Expanding) ---
        print("\n--- Testing CustomMissForest (3-Year Expanding) ---")
        try:
            df_imputed_full_period = run_expanding_imputation_benchmark(
                df_full_gapped=df_full_period_with_artificial_gaps, 
                seed_start_date=seed_start_date,
                seed_end_date=seed_end_date,
                window_years=seed_window_years,
                discharge_cols=discharge_cols,
                temporal_features=temporal_features,
                distance_matrix=distance_matrix,
                connectivity_matrix=connectivity_matrix
            )
            print("  Extracting evaluation period from expanding imputation result...")
            df_imputed = df_imputed_full_period.loc[df_eval_original.index]
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'CustomMissForest_3Yr_Expanding'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"CustomMissForest Expanding failed: {e}")
            gap_results_for_this_length['CustomMissForest_3Yr_Expanding'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 3: CustomMissForest (3-Year Seed-Only) ---
        print("\n--- Testing CustomMissForest (3-Year Seed-Only) ---")
        try:
            model_custom_seed = CustomMissForest(
                distance_matrix=distance_matrix,
                connectivity=connectivity_matrix,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=temporal_features,
                initialization_method='historical_mean'
            )
            print(f"  Training Custom MissForest on {seed_window_years}-year seed block...")
            model_custom_seed.fit(df_seed_gapped)
            print("  Imputing evaluation block...")
            df_imputed = model_custom_seed.transform(df_eval_gapped)
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'CustomMissForest_3Yr_SeedOnly'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"Custom MissForest (Seed-Only) failed: {e}")
            gap_results_for_this_length['CustomMissForest_3Yr_SeedOnly'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


        # --- Method 4: Benchmark_Vanilla_MissForest (No weighting, Seed-Only) ---
        print("\n--- Testing Benchmark_Vanilla_MissForest (No Weighting) ---")
        try:
            model_vanilla = ModifiedMissForest(
                distance_matrix=None,
                connectivity=None,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                temporal_feature_columns=temporal_features
            )
            print(f"  Training Vanilla MissForest on {seed_window_years}-year seed block...")
            model_vanilla.fit(df_seed_gapped)
            print("  Imputing evaluation block...")
            df_imputed = model_vanilla.transform(df_eval_gapped)
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Benchmark_Vanilla_MissForest'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"Vanilla MissForest failed: {e}")
            gap_results_for_this_length['Benchmark_Vanilla_MissForest'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}


        # --- Method 5: Benchmark_MICE_RandomForest ---
        print("\n--- Testing Benchmark_MICE_RandomForest ---")
        try:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42), 
                max_iter=10, 
                random_state=42
            )
            df_imputed = scale_and_impute_sklearn(imputer, df_seed_gapped, df_eval_gapped, all_cols_in_data)
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Benchmark_MICE_RandomForest'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"MICE (RF) failed: {e}")
            gap_results_for_this_length['Benchmark_MICE_RandomForest'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 6: Benchmark_KNN_k5 ---
        print("\n--- Testing Benchmark_KNN_k5 ---")
        try:
            imputer = KNNImputer(n_neighbors=5)
            df_imputed = scale_and_impute_sklearn(imputer, df_seed_gapped, df_eval_gapped, all_cols_in_data)
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Benchmark_KNN_k5'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"KNN failed: {e}")
            gap_results_for_this_length['Benchmark_KNN_k5'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 7: Benchmark_Kalman_SARIMAX ---
        print("\n--- Testing Benchmark_Kalman_SARIMAX ---")
        try:
            df_imputed = kalman_imputation(df_seed_gapped, df_eval_gapped, discharge_cols)
            
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Benchmark_Kalman_SARIMAX'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)

        except Exception as e:
            print(f"Kalman (SARIMAX) failed: {e}")
            gap_results_for_this_length['Benchmark_Kalman_SARIMAX'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 8: Benchmark_Linear_Interp ---
        print("\n--- Testing Benchmark_Linear_Interp ---")
        try:
            df_imputed = linear_interpolation_imputation(df_eval_gapped, discharge_cols)
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Benchmark_Linear_Interp'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)
        except Exception as e:
            print(f"Linear Interpolation failed: {e}")
            gap_results_for_this_length['Benchmark_Linear_Interp'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

        # --- Method 9: Baseline_Historical_Mean ---
        print("\n--- Testing Baseline_Historical_Mean ---")
        try:
            print("--- Running Historical Mean Imputation ---")
            df_imputed = historical_mean_imputation(df_eval_gapped, discharge_cols)
            metrics, y_true, y_pred = evaluate_imputation_performance(
                df_eval_original, df_eval_gapped, df_imputed, discharge_cols
            )
            method_name = 'Baseline_Historical_Mean'
            gap_results_for_this_length[method_name] = metrics
            print(f"✓ {method_name} - KGE: {metrics['KGE']:.4f}")
            plot_imputation_results(df_eval_original, df_eval_gapped, df_imputed, y_true, y_pred, output_dir, method_name, gap_length)
        except Exception as e:
            print(f"Historical Mean failed: {e}")
            gap_results_for_this_length['Baseline_Historical_Mean'] = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        
        # --- End of Benchmark Loop (for this gap length) ---
        
        # Add all results for this gap length to the main results dict
        all_results[f"{gap_length}-day_gap"] = gap_results_for_this_length

    # --- 6. Save comprehensive results ---
    print("\n" + "="*60)
    print("FOCUSED EVALUATION COMPLETE")
    print("="*60)
    
    # --- NEW: Create a MultiIndex DataFrame for easier analysis ---
    try:
        results_df = pd.DataFrame.from_dict({
            (gap_scenario, method): metrics
            for gap_scenario, methods in all_results.items()
            for method, metrics in methods.items()
        }, orient='index')
        
        results_df.index.names = ['GapScenario', 'Method']
        results_df = results_df.sort_index()
        
    except Exception as e:
        print(f"Could not create MultiIndex DataFrame, falling back to simple dict dump. Error: {e}")
        results_df = pd.DataFrame.from_dict(all_results, orient='index')

    
    results_csv = os.path.join(output_dir, f"evaluation_results_{search_start_date}_to_{search_end_date}.csv")
    results_df.to_csv(results_csv)
    
    print("\n--- Final Results Summary ---")
    print(results_df.round(4).to_string()) # .to_string() ensures it prints well
    print(f"\nResults saved to: {results_csv}")
    
    return results_df

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_focused_evaluation(
            search_start_date='1980-01-01',
            search_end_date='1990-12-31',
            seed_window_years=3
        )