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
    
    # Disjoint by contract: overlapping frames would put each shared date in twice and
    # the returned frame would carry duplicate rows for it.
    assert df_seed_gapped.index.intersection(df_eval_gapped.index).empty, \
        'kalman_imputation: the seed and eval frames overlap'
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
            # The SMOOTHED state, not predict()'s one-step-ahead filtered one. A filter
            # only sees the past, so inside a gap it decays from the last observation
            # towards the mean and never uses the reading on the far side: on Mahanadi
            # 30-day gaps that cost 527 -> 423 RMSE and -61% -> -23% PBIAS.
            imputed_series = res.get_prediction(
                start=df_combined_gapped.index[0], end=df_combined_gapped.index[-1],
                information_set='smoothed').predicted_mean

            # enforce_stationarity=False lets the fit return finite but explosive
            # coefficients, so predict() can emit values of order 1e37 without
            # raising anything the except branch would catch.
            # shortcut: a fixed 1e3x multiple of the observed maximum, loose enough
            # never to reject a usable fit; revisit if a gauge legitimately needs it.
            observed_max = df_combined_gapped[col].abs().max()
            diverged = not np.isfinite(imputed_series).all() or (
                pd.notna(observed_max) and observed_max > 0
                and np.nanmax(np.abs(imputed_series)) > 1e3 * observed_max)
            if diverged:
                raise ValueError(
                    f"SARIMAX diverged: |prediction| up to "
                    f"{np.nanmax(np.abs(imputed_series)):.3g} against an observed "
                    f"maximum of {observed_max:.3g}")

            df_imputed[col] = df_imputed[col].fillna(imputed_series)

        except Exception as e:
            print(f"  WARNING: SARIMAX failed for {col}: {e}. Falling back to column mean.")
            df_imputed[col] = df_imputed[col].fillna(col_mean)

    return df_imputed.loc[df_eval_gapped.index]
