# simplified_utils.py - Essential utility functions for burst imputation
import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic

def clean_station_name(name):
    """Standardizes station names for consistent matching."""
    name_correction_map = {
        "parmanpur": "paramanpur",
        "bamnidih": "bamnidhi",
    }
    
    name = str(name).strip().lower()
    name = re.sub(r"[''\"`]", "", name)
    name = re.sub(r"\s+", "", name)
    
    return name_correction_map.get(name, name)

def parse_lat_lon(coord_str, is_latitude=True):
    """Parses coordinate strings into decimal degrees."""
    if isinstance(coord_str, (int, float)):
        val = float(coord_str)
        if (is_latitude and not -90 <= val <= 90) or (not is_latitude and not -180 <= val <= 180):
            raise ValueError(f"Coordinate {val} is out of bounds.")
        return val

    s = str(coord_str).strip()
    s_spaced = re.sub(r"[o°'\"`'']", " ", s)
    numbers = [float(n) for n in re.findall(r'(-?\d+\.?\d*)', s_spaced)]
    direction = re.search(r'([NSEWnsew])', s)
    direction_char = direction.group(1).upper() if direction else None

    if not numbers:
        raise ValueError(f"No numerical parts found in coordinate string: '{coord_str}'")

    deg, min_, sec = numbers[0], 0.0, 0.0
    if len(numbers) > 1: min_ = numbers[1]
    if len(numbers) > 2: sec = numbers[2]
        
    decimal_deg_abs = abs(deg) + min_/60 + sec/3600
    final_decimal_deg = decimal_deg_abs
    
    if direction_char:
        if (is_latitude and direction_char == 'S') or (not is_latitude and direction_char == 'W'):
            final_decimal_deg *= -1
    elif deg < 0:
        final_decimal_deg *= -1

    if (is_latitude and not -90 <= final_decimal_deg <= 90) or \
       (not is_latitude and not -180 <= final_decimal_deg <= 180):
        raise ValueError(f"Parsed coordinate {final_decimal_deg} is out of bounds.")
        
    return final_decimal_deg

def load_and_preprocess_data(discharge_path, lat_long_path, contrib_path=None):
    """Load and preprocess all data sources with consistent station matching."""
    print("--- Loading and Preprocessing Data ---")
    
    # Load discharge data
    try:
        df_discharge = pd.read_csv(discharge_path, index_col=0, parse_dates=True, dayfirst=True)
        df_discharge.index.name = 'date'
        
        # Identify discharge vs temporal columns
        all_cols = df_discharge.columns.tolist()
        discharge_cols = [col for col in all_cols if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))]
        temporal_cols = [col for col in all_cols if col not in discharge_cols]

        # Clean discharge column names
        df_discharge_stations = df_discharge[discharge_cols].copy()
        df_discharge_stations.columns = df_discharge_stations.columns.map(clean_station_name)
        
        # Recombine with temporal features
        if temporal_cols:
            df_temporal = df_discharge[temporal_cols].copy()
            df_discharge = pd.concat([df_discharge_stations, df_temporal], axis=1)
        else:
            df_discharge = df_discharge_stations

        # Convert to numeric
        for col in discharge_cols:
            cleaned_name = clean_station_name(col)
            if cleaned_name in df_discharge.columns:
                df_discharge[cleaned_name] = pd.to_numeric(df_discharge[cleaned_name], errors='coerce')

    except Exception as e:
        print(f"ERROR: Could not load discharge data: {e}")
        return None, None, None, None, None

    # Load coordinate data
    try:
        df_coords = pd.read_csv(lat_long_path)
        column_rename_map = {'Latitude (N)': 'Latitude', 'Longitude (E)': 'Longitude', 'Name of site': 'Station'}
        df_coords.rename(columns={k: v for k, v in column_rename_map.items() if k in df_coords.columns}, inplace=True)
        
        if 'Station' in df_coords.columns:
            df_coords.set_index('Station', inplace=True)
        else:
            raise ValueError("'Name of site' column not found in coordinate file.")
        
        df_coords.index = df_coords.index.map(clean_station_name)
        
    except Exception as e:
        print(f"ERROR: Could not load coordinate data: {e}")
        return None, None, None, None, None

    # Find canonical stations (present in both discharge and coordinate data)
    discharge_stations = set([col for col in df_discharge.columns 
                            if not (col.startswith('day_of_year_') or col.startswith('month_') or col.startswith('week_of_year_'))])
    coords_stations = set(df_coords.index)
    canonical_stations = sorted(list(discharge_stations.intersection(coords_stations)))

    print(f"Using {len(canonical_stations)} stations found in both files.")

    # Filter data to canonical stations
    cols_to_keep = canonical_stations + temporal_cols
    df_discharge = df_discharge[cols_to_keep]
    df_coords = df_coords.loc[canonical_stations]

    # Parse coordinates
    df_coords['Latitude'] = df_coords['Latitude'].apply(lambda x: parse_lat_lon(x, is_latitude=True))
    df_coords['Longitude'] = df_coords['Longitude'].apply(lambda x: parse_lat_lon(x, is_latitude=False))

    # Load contributor data (optional)
    df_contrib = None
    if contrib_path:
        try:
            df_contrib_matrix = pd.read_csv(contrib_path, index_col=0)
            v_code_to_name = {v_code: clean_station_name(name) for v_code, name in df_contrib_matrix['Name of site'].items()}
            matrix_v_codes = [col for col in df_contrib_matrix.columns if col.startswith('V')]
            
            contrib_pairs = []
            for station_v_code, row in df_contrib_matrix.iterrows():
                station_name = v_code_to_name.get(station_v_code)
                if not station_name: continue
                for contributor_v_code in matrix_v_codes:
                    if row[contributor_v_code] == 1:
                        contributor_name = v_code_to_name.get(contributor_v_code)
                        if contributor_name:
                            contrib_pairs.append({'station': station_name, 'contributor': contributor_name})
            
            if contrib_pairs:
                df_contrib = pd.DataFrame(contrib_pairs)
                df_contrib = df_contrib[df_contrib['station'].isin(canonical_stations) & 
                                      df_contrib['contributor'].isin(canonical_stations)]
                print(f"Loaded contributor matrix with {len(df_contrib)} relationships.")
            else:
                df_contrib = None
        except Exception as e:
            print(f"Warning: Could not load contributor data: {e}")
            df_contrib = None
            
    # Create station mappings
    vcode_to_station, station_to_vcode = {}, {}
    if 'v_code' in df_coords.columns:
        coords_unique = df_coords[~df_coords.index.duplicated(keep='first')]
        vcode_to_station = coords_unique['v_code'].to_dict()
        station_to_vcode = {v: k for k, v in vcode_to_station.items()}
    
    print("Data loading complete.")
    return df_discharge, df_contrib, df_coords, vcode_to_station, station_to_vcode

def add_temporal_features(df):
    """Add cyclical temporal features (sin/cos of day of year)."""
    print("Adding temporal features...")
    df_temp = df.copy()
    df_temp.index = pd.to_datetime(df_temp.index)
    day_of_year = df_temp.index.dayofyear
    df_temp['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 366.0)
    df_temp['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 366.0)
    return df_temp

def build_distance_matrix(df_coords, discharge_cols):
    """Build distance matrix between all stations."""
    print("Building distance matrix...")
    stations = discharge_cols
    distance_matrix = pd.DataFrame(np.inf, index=stations, columns=stations)
    
    for station_i in stations:
        for station_j in stations:
            if station_i == station_j:
                distance_matrix.loc[station_i, station_j] = 0
                continue
            if station_i in df_coords.index and station_j in df_coords.index:
                lat1, lon1 = df_coords.loc[station_i, ['Latitude', 'Longitude']]
                lat2, lon2 = df_coords.loc[station_j, ['Latitude', 'Longitude']]
                dist = geodesic((lat1, lon1), (lat2, lon2)).km
                distance_matrix.loc[station_i, station_j] = dist
    return distance_matrix

def build_connectivity_matrix(df_contrib, discharge_cols, station_name_to_vcode):
    """Build directed connectivity matrix from contributor data."""
    print("Building connectivity matrix...")
    stations = discharge_cols
    connectivity_matrix = pd.DataFrame(0.0, index=stations, columns=stations)
    
    if df_contrib is None or df_contrib.empty:
        return connectivity_matrix
        
    for _, row in df_contrib.iterrows():
        station = row.get('station')
        contributor = row.get('contributor')
        if station in connectivity_matrix.index and contributor in connectivity_matrix.columns:
            connectivity_matrix.loc[station, contributor] = 1.0
            
    return connectivity_matrix

def create_contiguous_segment_gaps(df_data, discharge_cols, gap_lengths, random_seed=42, num_intervals_per_column=5):
    """Create contiguous gaps in discharge data for evaluation."""
    np.random.seed(random_seed)
    results = {}

    for gap_length in gap_lengths:
        if gap_length <= 0:
            continue

        df_gapped = df_data.copy()
        true_values = {}
        mask = {}

        for target_column in discharge_cols:
            data_length = len(df_gapped)
            
            if gap_length >= data_length:
                true_values[target_column] = df_data[target_column].copy().values
                df_gapped[target_column] = np.nan
                mask[target_column] = np.ones(data_length, dtype=bool)
            else:
                num_intervals = min(num_intervals_per_column, data_length // gap_length)
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
                        if len(gaps_to_apply) == num_intervals:
                            break
                
                # Apply gaps
                col_true_values = np.array([])
                col_mask = np.zeros(data_length, dtype=bool)

                for start_idx, end_idx in gaps_to_apply:
                    col_true_values = np.concatenate([col_true_values, df_data.loc[df_data.index[start_idx:end_idx], target_column].copy().values])
                    col_mask[start_idx:end_idx] = True
                    df_gapped.loc[df_gapped.index[start_idx:end_idx], target_column] = np.nan
                
                true_values[target_column] = col_true_values
                mask[target_column] = col_mask
        
        results[gap_length] = {
            'gapped_data': df_gapped,
            'true_values': true_values, 
            'mask': mask
        }
    return results

def create_single_point_gaps(df_data, discharge_cols, num_gaps_per_column, random_seed=42):
    """Create single point gaps in discharge data for evaluation."""
    np.random.seed(random_seed)
    results = {}

    for num_gaps_key, num_gaps_value in num_gaps_per_column.items():
        df_gapped = df_data.copy()
        true_values = {}
        mask = {}

        for target_column in discharge_cols:
            data_length = len(df_gapped)
            if data_length == 0: continue

            # Calculate number of gaps as percentage of data length
            n_gaps = int(data_length * (num_gaps_value / 365.0))
            if n_gaps == 0 and num_gaps_value > 0: n_gaps = 1

            if n_gaps >= data_length:
                true_values[target_column] = df_data[target_column].copy().values
                df_gapped[target_column] = np.nan
                mask[target_column] = np.ones(data_length, dtype=bool)
            else:
                # Select random indices for gaps
                gap_indices = np.random.choice(data_length, n_gaps, replace=False)
                
                # Store true values and create mask
                true_values[target_column] = df_data.loc[df_data.index[gap_indices], target_column].copy().values
                col_mask = np.zeros(data_length, dtype=bool)
                col_mask[gap_indices] = True
                mask[target_column] = col_mask

                # Introduce NaN values
                df_gapped.loc[df_gapped.index[gap_indices], target_column] = np.nan
        
        results[num_gaps_key] = {
            'gapped_data': df_gapped,
            'true_values': true_values,
            'mask': mask
        }
    return results

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics (RMSE, MAE, R2, NSE, KGE)."""
    # Filter out NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    ss_total = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    ss_residual = np.sum((y_true_clean - y_pred_clean)**2)
    
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
    nse = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    # Calculate Kling Gupta Efficiency (KGE)
    kge = calculate_kge(y_true_clean, y_pred_clean)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse, 'KGE': kge}

def calculate_kge(y_true, y_pred):
    """Calculate Kling Gupta Efficiency (KGE).
    
    KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)
    
    Where:
    - r = correlation coefficient
    - beta = mean(y_pred) / mean(y_true) (bias ratio)
    - gamma = std(y_pred) / std(y_true) (variability ratio)
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Correlation coefficient
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        r = 0
    
    # Bias ratio (beta)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if mean_true == 0:
        beta = np.nan if mean_pred != 0 else 1.0
    else:
        beta = mean_pred / mean_true
    
    # Variability ratio (gamma)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true == 0:
        gamma = np.nan if std_pred != 0 else 1.0
    else:
        gamma = std_pred / std_true
    
    # Handle NaN values in beta or gamma
    if np.isnan(beta) or np.isnan(gamma):
        return np.nan
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return kge

def historical_mean_imputation(df_data, discharge_cols, min_years_for_mean=2):
    """
    Impute missing values using historical mean for each day of year.
    For day X, use the average value of day X across all prior years.
    
    Args:
        df_data: DataFrame with datetime index
        discharge_cols: List of discharge column names to impute
        min_years_for_mean: Minimum number of years needed to calculate historical mean
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col not in df_data.columns:
            continue
            
        print(f"Applying historical mean imputation to {col}...")
        
        # Calculate column mean as ultimate fallback
        column_mean = df_data[col].mean()
        if pd.isna(column_mean):
            column_mean = 0.0  # Default value for completely empty columns
        
        # Get day of year for each date
        day_of_year = df_data.index.dayofyear
        
        for i, (date, value) in enumerate(df_data[col].items()):
            if pd.isna(value):
                current_day = day_of_year[i]
                current_year = date.year
                
                # Find historical values for the same day of year from prior years
                historical_data = df_data[
                    (df_data.index.dayofyear == current_day) & 
                    (df_data.index.year < current_year)
                ][col]
                
                # Filter out NaN values from historical data
                historical_values = historical_data.dropna()
                
                if len(historical_values) >= min_years_for_mean:
                    # Use mean of historical values for this day of year
                    imputed_value = historical_values.mean()
                    df_imputed.loc[date, col] = imputed_value
                else:
                    # Fallback to column mean if insufficient historical data
                    df_imputed.loc[date, col] = column_mean
        
        # Ensure ALL missing values are filled (additional safety check)
        remaining_nans = df_imputed[col].isnull().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values still remain in {col}, filling with column mean")
            df_imputed[col] = df_imputed[col].fillna(column_mean)
    
    return df_imputed

def seasonal_mean_imputation(df_data, discharge_cols, window_days=15):
    """
    Impute missing values using seasonal mean within a window around the missing day.
    
    Args:
        df_data: DataFrame with datetime index
        discharge_cols: List of discharge column names to impute
        window_days: Number of days before and after to include in seasonal mean
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col not in df_data.columns:
            continue
            
        print(f"Applying seasonal mean imputation to {col}...")
        
        # Calculate column mean as ultimate fallback
        column_mean = df_data[col].mean()
        if pd.isna(column_mean):
            column_mean = 0.0  # Default value for completely empty columns
        
        for i, (date, value) in enumerate(df_data[col].items()):
            if pd.isna(value):
                # Create window around the missing date
                start_date = date - pd.Timedelta(days=window_days)
                end_date = date + pd.Timedelta(days=window_days)
                
                # Get data within the window, excluding the missing date itself
                window_data = df_data[
                    (df_data.index >= start_date) & 
                    (df_data.index <= end_date) & 
                    (df_data.index != date)
                ][col]
                
                # Filter out NaN values
                window_values = window_data.dropna()
                
                if len(window_values) > 0:
                    # Use mean of values within the seasonal window
                    imputed_value = window_values.mean()
                    df_imputed.loc[date, col] = imputed_value
                else:
                    # Fallback to column mean if no valid window data
                    df_imputed.loc[date, col] = column_mean
        
        # Ensure ALL missing values are filled (additional safety check)
        remaining_nans = df_imputed[col].isnull().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values still remain in {col}, filling with column mean")
            df_imputed[col] = df_imputed[col].fillna(column_mean)
    
    return df_imputed

def simple_column_mean_imputation(df_data, discharge_cols):
    """
    Simple imputation using column means (baseline method).
    
    Args:
        df_data: DataFrame with datetime index
        discharge_cols: List of discharge column names to impute
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col in df_data.columns:
            column_mean = df_data[col].mean()
            
            # If column mean is NaN (all values are missing), use a default value
            if pd.isna(column_mean):
                print(f"Warning: Column {col} has all NaN values, using default value 0")
                df_imputed[col] = df_imputed[col].fillna(0.0)
            else:
                df_imputed[col] = df_imputed[col].fillna(column_mean)
    
    return df_imputed

def initialize_for_missforest(df_data, discharge_cols, initialization_method='column_mean', min_years_for_mean=2):
    """
    Initialize missing values using different methods for MissForest.
    
    Args:
        df_data: DataFrame with datetime index
        discharge_cols: List of discharge column names to initialize
        initialization_method: Method to use ('column_mean', 'historical_mean', 'seasonal_mean')
        min_years_for_mean: Minimum years for historical mean calculation
    
    Returns:
        DataFrame with initialized values (guaranteed no NaN values)
    """
    if initialization_method == 'column_mean':
        df_initialized = simple_column_mean_imputation(df_data, discharge_cols)
    elif initialization_method == 'historical_mean':
        df_initialized = historical_mean_imputation(df_data, discharge_cols, min_years_for_mean)
    elif initialization_method == 'seasonal_mean':
        df_initialized = seasonal_mean_imputation(df_data, discharge_cols, window_days=15)
    else:
        raise ValueError(f"Unknown initialization method: {initialization_method}")
    
    # Final safety check: ensure absolutely no NaN values remain
    for col in discharge_cols:
        if col in df_initialized.columns:
            remaining_nans = df_initialized[col].isnull().sum()
            if remaining_nans > 0:
                print(f"Final safety check: {remaining_nans} NaN values found in {col}, filling with column mean")
                column_mean = df_data[col].mean()
                if pd.isna(column_mean):
                    column_mean = 0.0  # Default value for completely empty columns
                df_initialized[col] = df_initialized[col].fillna(column_mean)
    
    # Verify no NaN values remain
    total_nans = df_initialized[discharge_cols].isnull().sum().sum()
    if total_nans > 0:
        print(f"ERROR: {total_nans} NaN values still remain after initialization!")
        # Emergency fallback to simple column mean for all columns
        for col in discharge_cols:
            if col in df_initialized.columns:
                column_mean = df_data[col].mean()
                if pd.isna(column_mean):
                    column_mean = 0.0  # Default value for completely empty columns
                df_initialized[col] = df_initialized[col].fillna(column_mean)
    
    print(f"Initialization complete. Total NaN values remaining: {df_initialized[discharge_cols].isnull().sum().sum()}")
    return df_initialized

def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    """
    Evaluate imputation performance by comparing true values with imputed values
    at the locations where artificial gaps were created.
    """
    y_true_eval, y_pred_eval = [], []

    for station in discharge_cols:
        if station not in df_imputed.columns:
            continue
            
        # Find where the artificial gaps were created
        gap_mask = df_gapped[station].isnull()
        
        if gap_mask.sum() > 0:
            # Get the predicted values from our model at these specific gap locations
            predicted_vals = df_imputed[station][gap_mask].values
            
            # Get the ground truth values from the original, non-gapped test data
            true_vals = df_original[station][gap_mask].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)

    if y_true_eval:
        # evaluate_metrics is already defined in this file
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics
    else:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

def find_best_data_window(df, discharge_cols, start_date_str, end_date_str, window_size_days):
    """
    Finds the N-day window with the most non-NaN values within a given date range.

    Args:
        df: DataFrame with datetime index.
        discharge_cols: List of columns to check for completeness.
        start_date_str: Start of the search range (e.g., '1980-01-01').
        end_date_str: End of the search range (e.g., '1990-12-31').
        window_size_days: The size of the window to find (e.g., 1826 for 5 years).

    Returns:
        (best_start_date, best_end_date) as pd.Timestamps
    """
    print(f"Searching for best {window_size_days}-day window in {start_date_str} to {end_date_str}...")
    df_range = df.loc[start_date_str:end_date_str].copy()
    
    if df_range.empty:
        raise ValueError(f"No data found in the specified range {start_date_str} to {end_date_str}.")
        
    window_size_str = f"{window_size_days}D"
    
    # Calculate rolling completeness
    # Sum non-NaN values per row, then get a rolling sum of this completeness score
    completeness_series = df_range[discharge_cols].notna().sum(axis=1)
    rolling_completeness = completeness_series.rolling(window=window_size_str, min_periods=window_size_days).sum()
    
    if rolling_completeness.dropna().empty:
        raise ValueError(f"No complete windows of size {window_size_days} days found.")
        
    # Find the end date of the best window
    best_end_date = rolling_completeness.idxmax()
    best_start_date = best_end_date - pd.Timedelta(days=window_size_days - 1)
    
    print(f"Found best window: {best_start_date.date()} to {best_end_date.date()} "
          f"with {rolling_completeness.max()} data points.")
          
    return best_start_date, best_end_date