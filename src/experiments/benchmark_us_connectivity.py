import os
import random
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.utils.data import add_temporal_features
from src.utils.spatial import build_distance_matrix
from src.utils.metrics import evaluate_metrics
from src.imputers.missforest import MissForestImputer

from src.experiments.benchmark_us_data import load_us_data, create_gaps, evaluate_imputation_performance

TOP_10_STATIONS = [
    '01076500', '01078000', '01064500', '01048000', '01022500', 
    '01054200', '01052500', '01010000', '01055000', '01047000'
]

def build_custom_connectivity(stations):
    # Initialize all to 0
    connectivity = pd.DataFrame(0.0, index=stations, columns=stations)
    
    # 01076500 is upstream of 01078000 (01076500 contributes to 01078000)
    if '01078000' in connectivity.index and '01076500' in connectivity.columns:
        connectivity.loc['01078000', '01076500'] = 1.0
        
    # 01047000 is upstream of 01048000
    if '01048000' in connectivity.index and '01047000' in connectivity.columns:
        connectivity.loc['01048000', '01047000'] = 1.0
        
    # 01054200 is upstream of 01055000
    if '01055000' in connectivity.index and '01054200' in connectivity.columns:
        connectivity.loc['01055000', '01054200'] = 1.0
        
    return connectivity

def run_experiment():
    print("Loading US Data...")
    df_combined, df_coords = load_us_data()
    
    # Filter to only the top 10 stations
    existing_top_10 = [s for s in TOP_10_STATIONS if s in df_combined.columns]
    df_combined = df_combined[existing_top_10]
    
    df_with_features = add_temporal_features(df_combined)
    
    discharge_cols = existing_top_10
    temporal_features = [c for c in df_with_features.columns if c not in discharge_cols]
    
    distance_matrix = build_distance_matrix(df_coords, discharge_cols)
    connectivity_matrix = build_custom_connectivity(discharge_cols)
    
    print("\nInferred Connectivity Matrix:")
    print(connectivity_matrix)
    
    blocks = [
        ('2013-01-01', '2015-12-31'),
        ('2016-01-01', '2018-12-31'),
        ('2019-01-01', '2021-12-31'),
        ('2022-01-01', '2024-12-31')
    ]
    
    out_dir = f"benchmark_us_connectivity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = {}
    current_model = None
    
    for start_date, end_date in blocks:
        print(f"\n--- Processing Block {start_date} to {end_date} ---")
        df_block_original = df_with_features.loc[start_date:end_date]
        if df_block_original.empty:
            print(f"No data for {start_date} to {end_date}")
            continue
            
        df_block_gapped = create_gaps(df_block_original, discharge_cols, gap_length=7, min_percent=10.0)
        
        print("Training/Imputing with Distance+Connectivity-Weighted MissForest (Historical Mean Init)")
        if current_model is None:
            model = MissForestImputer(
                distance_matrix=distance_matrix,
                connectivity=connectivity_matrix,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=temporal_features,
                initialization_method='historical_mean',
                ordering_method='none'
            )
            model.fit(df_block_gapped)
            current_model = model
            df_block_imputed = model.transform(df_block_gapped)
        else:
            # Impute the new block using the old model
            df_block_imputed = current_model.transform(df_block_gapped)
            
            # Update the model using the imputed new block
            new_model = MissForestImputer(
                distance_matrix=distance_matrix,
                connectivity=connectivity_matrix,
                max_iter=10,
                n_estimators=100,
                random_state=42,
                distance_weighting_type='inverse',
                temporal_feature_columns=temporal_features,
                initialization_method='historical_mean',
                ordering_method='none'
            )
            new_model.fit(df_block_imputed)
            current_model = new_model
            
        metrics, _, _ = evaluate_imputation_performance(df_block_original, df_block_gapped, df_block_imputed, discharge_cols)
        print(f"Overall Metrics for block: {metrics}")
        all_results[f"{start_date}_to_{end_date}"] = metrics
        
        # Calculate per-station metrics for visibility
        print("Per-station KGE:")
        for station in discharge_cols:
            gap_mask = df_block_gapped[station].isnull() & df_block_original[station].notnull()
            if gap_mask.sum() > 0:
                pred = df_block_imputed.loc[gap_mask.index[gap_mask], station].values
                true = df_block_original.loc[gap_mask.index[gap_mask], station].values
                st_metrics = evaluate_metrics(true, pred)
                print(f"  {station}: {st_metrics['KGE']:.4f}")

    results_df = pd.DataFrame(all_results).T
    csv_out = os.path.join(out_dir, "us_connectivity_benchmark_results.csv")
    results_df.to_csv(csv_out)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print(results_df.round(4).to_string())
    print(f"Saved to {csv_out}")
    print("="*60)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_experiment()
