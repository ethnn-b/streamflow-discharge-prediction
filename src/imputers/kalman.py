import pandas as pd
import numpy as np

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    print("WARNING: statsmodels not found. Kalman (SARIMAX) benchmark will be skipped.")
    SARIMAX = None

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
