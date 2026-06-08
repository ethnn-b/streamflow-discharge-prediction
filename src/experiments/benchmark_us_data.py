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

def parse_dms(dms_str):
    # E.g., 464200N -> 46 deg 42 min 00 sec N
    if dms_str.endswith(('N', 'S')):
        deg = int(dms_str[:2])
        min_ = int(dms_str[2:4])
        sec = int(dms_str[4:6])
        sign = 1 if dms_str.endswith('N') else -1
    else:
        # Longitude can have 3 digits
        deg = int(dms_str[:3])
        min_ = int(dms_str[3:5])
        sec = int(dms_str[5:7])
        sign = 1 if dms_str.endswith('E') else -1
    return sign * (deg + min_/60 + sec/3600)

RAW_COORDS = """
1010000 	1 	EAST 	37 	ME 	464200N 	0694259W
1022500 	1 	EAST 	39 	ME 	443629N 	0675610W
1030500 	1 	EAST 	53 	ME 	453018N 	0681807W
1031500 	1 	EAST 	85 	ME 	451031N 	0691855W
1033500 	1 	EAST 	58 	ME 	451658N 	0690013W
1035000 	1 	EAST 	63 	ME 	451104N 	0682829W
1038000 	1 	EAST 	49 	ME 	441323N 	0693538W
1047000 	1 	EAST 	65 	ME 	445209N 	0695720W
1048000 	1 	EAST 	49 	ME 	444226N 	0695621W
1052500 	1 	EAST 	46 	NH 	445240N 	0710325W
1054200 	1 	EAST 	23 	ME 	442327N 	0705847W
1055000 	1 	EAST 	58 	ME 	443832N 	0703517W
1055500 	1 	EAST 	46 	ME 	441610N 	0701349W
1060000 	1 	EAST 	38 	ME 	434757N 	0701045W
1064500 	1 	EAST 	63 	NH 	435927N 	0710529W
1073000 	1 	EAST 	52 	NH 	430855N 	0705756W
1074500 	1 	EAST 	22 	NH 	440323N 	0713818W
1075000 	1 	EAST 	37 	NH 	435834N 	0714048W
1076000 	1 	EAST 	47 	NH 	434746N 	0715042W
1076500 	1 	EAST 	84 	NH 	434533N 	0714110W
1078000 	1 	EAST 	69 	NH 	433404N 	0714454W
1094000 	1 	EAST 	66 	NH 	425127N 	0713024W
1106000 	1 	EAST 	37 	RI 	413330N 	0710747W
"""

def evaluate_imputation_performance(df_original, df_gapped, df_imputed, discharge_cols):
    y_true_eval, y_pred_eval = [], []
    for station in discharge_cols:
        if station not in df_imputed.columns or station not in df_gapped.columns or station not in df_original.columns:
            continue
        
        gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
        if gap_mask.sum() > 0:
            predicted_vals = df_imputed.loc[gap_mask.index[gap_mask], station].values
            true_vals = df_original.loc[gap_mask.index[gap_mask], station].values
            
            y_pred_eval.extend(predicted_vals)
            y_true_eval.extend(true_vals)

    if y_true_eval:
        metrics = evaluate_metrics(np.array(y_true_eval), np.array(y_pred_eval))
        return metrics, y_true_eval, y_pred_eval
    else:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}, [], []

def load_us_data():
    data_dir = 'us_data/US_Data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    
    # Process coords
    coords = []
    for line in RAW_COORDS.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 7:
            sid = str(parts[0]).zfill(8)
            lat = parse_dms(parts[5])
            lon = parse_dms(parts[6])
            coords.append({'site_no': sid, 'Latitude': lat, 'Longitude': lon})
    df_coords = pd.DataFrame(coords).set_index('site_no')

    valid_sites = df_coords.index.tolist()
    
    for f in files:
        site_id = f.split('_')[0].replace('USGS-', '')
        if site_id not in valid_sites:
            continue
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['date'])
        # strip timezone if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        df = df[['date', 'value']].set_index('date')
        df.columns = [site_id]
        dfs.append(df)
        
    df_combined = pd.concat(dfs, axis=1)
    df_combined = df_combined.loc['2013-01-01':'2024-12-31']
    # resample to daily in case of missing days
    df_combined = df_combined.resample('D').mean()
    return df_combined, df_coords

def create_gaps(df, columns, gap_length=7, min_percent=10.0):
    df_gapped = df.copy()
    n_days = len(df)
    target_gaps_days = np.ceil(n_days * (min_percent / 100.0))
    n_gaps = int(np.ceil(target_gaps_days / gap_length))
    
    for col in columns:
        valid_indices = np.where(df[col].notna())[0]
        if len(valid_indices) < n_gaps * gap_length:
            continue
            
        gaps_placed = 0
        attempts = 0
        max_attempts = n_gaps * 10
        col_idx = df_gapped.columns.get_loc(col)
        
        while gaps_placed < n_gaps and attempts < max_attempts:
            attempts += 1
            start_idx = random.randint(0, n_days - gap_length)
            end_idx = start_idx + gap_length
            
            # check if not already gapped
            if df_gapped.iloc[start_idx:end_idx, col_idx].notna().all():
                df_gapped.iloc[start_idx:end_idx, col_idx] = np.nan
                gaps_placed += 1
                
    return df_gapped

def run_experiment():
    print("Loading US Data...")
    df_combined, df_coords = load_us_data()
    df_with_features = add_temporal_features(df_combined)
    
    discharge_cols = df_combined.columns.tolist()
    temporal_features = [c for c in df_with_features.columns if c not in discharge_cols]
    
    distance_matrix = build_distance_matrix(df_coords, discharge_cols)
    connectivity_matrix = pd.DataFrame() # empty for US data
    
    blocks = [
        ('2013-01-01', '2015-12-31'),
        ('2016-01-01', '2018-12-31'),
        ('2019-01-01', '2021-12-31'),
        ('2022-01-01', '2024-12-31')
    ]
    
    out_dir = f"benchmark_us_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        
        print("Training/Imputing with Distance-Weighted MissForest (Historical Mean Init)")
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
        print(f"Metrics: {metrics}")
        all_results[f"{start_date}_to_{end_date}"] = metrics
        
    results_df = pd.DataFrame(all_results).T
    csv_out = os.path.join(out_dir, "us_benchmark_results.csv")
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
