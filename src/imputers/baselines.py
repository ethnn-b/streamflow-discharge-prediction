import pandas as pd

def simple_column_mean_imputation(df_data, discharge_cols):
    """
    Simple imputation using column means (baseline method).
    """
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col in df_data.columns:
            column_mean = df_data[col].mean()
            if pd.isna(column_mean):
                print(f"Warning: Column {col} has all NaN values, using default value 0")
                df_imputed[col] = df_imputed[col].fillna(0.0)
            else:
                df_imputed[col] = df_imputed[col].fillna(column_mean)
                
    return df_imputed

def historical_mean_imputation(df_data, discharge_cols, min_years_for_mean=2, training_data=None):
    """
    Impute missing values using historical mean for each day of year.
    For day X, use the average value of day X across all prior years.
    If testing on gapped eval data, training_data can be provided to compute means.
    """
    df_imputed = df_data.copy()
    source_df = training_data if training_data is not None else df_data
    
    print("\n--- Running Baseline_Historical_Mean_Imputation ---")
    
    for col in discharge_cols:
        if col not in df_data.columns:
            continue
            
        print(f"Applying historical mean imputation to {col}...")
        
        # Calculate column mean as ultimate fallback
        column_mean = source_df[col].mean()
        if pd.isna(column_mean):
            column_mean = 0.0  # Default value for completely empty columns
        
        day_of_year_source = source_df.index.dayofyear
        
        for i, (date, value) in enumerate(df_data[col].items()):
            if pd.isna(value):
                current_day = date.dayofyear
                current_year = date.year
                
                # Find historical values for the same day of year from ALL other years
                historical_data = source_df[
                    (source_df.index.dayofyear == current_day) & 
                    (source_df.index.year != current_year)
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

def linear_interpolation_imputation(df_data, discharge_cols):
    """Imputes missing values using time-based linear interpolation."""
    print(f"\n--- Running Benchmark_Linear_Interp ---")
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].interpolate(method='time')
            df_imputed[col] = df_imputed[col].bfill().ffill()
            
    print("✓ Linear interpolation complete.")
    return df_imputed


def seasonal_mean_imputation(df_data, discharge_cols, window_days=15):
    """
    Impute missing values using seasonal mean within a window around the missing day.
    """
    df_imputed = df_data.copy()
    
    for col in discharge_cols:
        if col not in df_data.columns:
            continue
            
        print(f"Applying seasonal mean imputation to {col}...")
        
        column_mean = df_data[col].mean()
        if pd.isna(column_mean):
            column_mean = 0.0
        
        for i, (date, value) in enumerate(df_data[col].items()):
            if pd.isna(value):
                start_date = date - pd.Timedelta(days=window_days)
                end_date = date + pd.Timedelta(days=window_days)
                
                window_data = df_data[
                    (df_data.index >= start_date) & 
                    (df_data.index <= end_date) & 
                    (df_data.index != date)
                ][col]
                
                window_values = window_data.dropna()
                
                if len(window_values) > 0:
                    imputed_value = window_values.mean()
                    df_imputed.loc[date, col] = imputed_value
                else:
                    df_imputed.loc[date, col] = column_mean
        
        remaining_nans = df_imputed[col].isnull().sum()
        if remaining_nans > 0:
            df_imputed[col] = df_imputed[col].fillna(column_mean)
    
    return df_imputed


def initialize_for_missforest(df_data, discharge_cols, initialization_method='column_mean', min_years_for_mean=2, training_data=None):
    """
    Initialize missing values using different methods for MissForest.
    """
    if initialization_method == 'column_mean':
        df_initialized = simple_column_mean_imputation(df_data, discharge_cols)
    elif initialization_method == 'historical_mean':
        df_initialized = historical_mean_imputation(df_data, discharge_cols, min_years_for_mean, training_data=training_data)
    elif initialization_method == 'seasonal_mean':
        df_initialized = seasonal_mean_imputation(df_data, discharge_cols, window_days=15)
    else:
        raise ValueError(f"Unknown initialization method: {initialization_method}")
    
    for col in discharge_cols:
        if col in df_initialized.columns:
            remaining_nans = df_initialized[col].isnull().sum()
            if remaining_nans > 0:
                print(f"Final safety check: {remaining_nans} NaN values found in {col}, filling with column mean")
                column_mean = df_data[col].mean()
                if pd.isna(column_mean):
                    column_mean = 0.0
                df_initialized[col] = df_initialized[col].fillna(column_mean)
    
    return df_initialized

