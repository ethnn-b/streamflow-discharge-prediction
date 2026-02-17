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
OUTPUT_FILE = 'verification_results_final.csv'

# Same dates as before
TRAIN_START, TRAIN_END = '1985-01-01', '1990-12-31'
TEST_START, TEST_END = '1990-01-01', '1995-12-31'
GAP_LENGTH = 7
TARGET_GAP_PCT = 10

# Configurations to verify
CONFIGS_TO_TEST = {
    "Baseline_Light": {"max_iter": 5, "n_estimators": 50},   # Fast
    "Winner_Combined": {"max_iter": 10, "n_estimators": 200} # The theoretical optimum
}

# Run multiple seeds to ensure stability
SEEDS = [42, 123, 2024, 777, 99]

def get_data():
    """Reuses loading logic."""
    print("Loading data...")
    df_all, df_contrib, df_coords, _, station_to_vcode = \
        load_and_preprocess_data(DISCHARGE_PATH, LAT_LONG_PATH, CONTRIB_PATH)
    
    df_feat = add_temporal_features(df_all)
    all_cols = df_feat.columns.tolist()
    discharge_cols = [c for c in all_cols if "day_" not in c and "month_" not in c]
    temporal_cols = [c for c in all_cols if c not in discharge_cols]
    
    dist = build_distance_matrix(df_coords, discharge_cols)
    conn = build_connectivity_matrix(df_contrib, discharge_cols, station_to_vcode)
    
    df_train = df_feat.loc[TRAIN_START:TRAIN_END].copy()
    df_test_orig = df_feat.loc[TEST_START:TEST_END].copy()
    
    # Create gaps once per seed inside the loop? 
    # Better: Create ONE set of gaps per seed to ensure fairness across models
    return df_train, df_test_orig, discharge_cols, temporal_cols, dist, conn

def main():
    df_train, df_test_orig, dis_cols, temp_cols, dist, conn = get_data()
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Testing {len(CONFIGS_TO_TEST)} Configs across {len(SEEDS)} Seeds")
    print(f"{'='*60}")

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        
        # 1. Create gaps for this seed (Fair comparison: both models get same gaps)
        gap_res = create_contiguous_segment_gaps(
            df_test_orig, dis_cols, [GAP_LENGTH], 
            num_intervals_per_column=int(len(df_test_orig)*0.10/GAP_LENGTH), 
            random_seed=seed
        )
        df_test_gapped = gap_res[GAP_LENGTH]['gapped_data']
        
        # 2. Test each config
        for name, params in CONFIGS_TO_TEST.items():
            print(f"  > Testing {name} ({params})...")
            start = time.time()
            
            model = CustomMissForest(
                distance_matrix=dist, connectivity=conn,
                max_iter=params['max_iter'],
                n_estimators=params['n_estimators'],
                random_state=seed, # Use same seed for RF internal randomness
                temporal_feature_columns=temp_cols,
                initialization_method='historical_mean'
            )
            
            model.fit(df_train)
            df_imputed = model.transform(df_test_gapped)
            
            elapsed = time.time() - start
            
            # Evaluate
            y_true, y_pred = [], []
            for col in dis_cols:
                if col not in df_imputed.columns: continue
                mask = df_test_gapped[col].isna() & df_test_orig[col].notna()
                if mask.sum() > 0:
                    y_true.extend(df_test_orig.loc[mask, col].values)
                    y_pred.extend(df_imputed.loc[mask, col].values)
            
            metrics = evaluate_metrics(np.array(y_true), np.array(y_pred))
            
            results.append({
                "Configuration": name,
                "Seed": seed,
                "Time_Sec": elapsed,
                "KGE": metrics['KGE'],
                "NSE": metrics['NSE'],
                "RMSE": metrics['RMSE']
            })
            print(f"    [Done] Time: {elapsed:.1f}s | KGE: {metrics['KGE']:.4f}")

    # Results Analysis
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*60)
    print("FINAL VERIFICATION SUMMARY (Mean +/- Std Dev)")
    print("="*60)
    
    summary = df_res.groupby("Configuration")[["Time_Sec", "KGE", "NSE"]].agg(['mean', 'std'])
    print(summary)
    
    # Calculate gain
    avg_kge_light = summary.loc["Baseline_Light", ("KGE", "mean")]
    avg_kge_winner = summary.loc["Winner_Combined", ("KGE", "mean")]
    avg_time_light = summary.loc["Baseline_Light", ("Time_Sec", "mean")]
    avg_time_winner = summary.loc["Winner_Combined", ("Time_Sec", "mean")]
    
    print(f"\nComparison:")
    print(f"Accuracy Gain: +{avg_kge_winner - avg_kge_light:.4f} KGE")
    print(f"Time Cost:     x{avg_time_winner / avg_time_light:.1f} slower")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()