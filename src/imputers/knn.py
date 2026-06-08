import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def scale_and_impute_sklearn(imputer, df_seed_gapped, df_eval_gapped, all_cols):
    """
    Helper function to correctly scale and impute data for sklearn models.
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
         print(f"  Warning: Eval set columns still have NaNs after fill. Filling with 0.")
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

    if imputed_eval_scaled_values.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Imputer output shape ({imputed_eval_scaled_values.shape[1]}) does not match scaler input shape ({scaler.n_features_in_})")
        
    imputed_eval_original_scale = scaler.inverse_transform(imputed_eval_scaled_values)
    
    df_imputed = pd.DataFrame(imputed_eval_original_scale, 
                              columns=all_cols, 
                              index=df_eval.index)
                              
    return df_imputed

def knn_imputation(df_seed_gapped, df_eval_gapped, all_cols_in_data, k=5):
    """Imputes missing values using KNNImputer."""
    print(f"\n--- Running Benchmark_KNN_k{k} ---")
    imputer = KNNImputer(n_neighbors=k)
    df_imputed = scale_and_impute_sklearn(imputer, df_seed_gapped, df_eval_gapped, all_cols_in_data)
    return df_imputed
