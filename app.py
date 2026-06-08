import gradio as gr
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import imputation modules
from src.utils.data import load_and_preprocess_data, add_temporal_features
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix
from src.imputers.missforest import MissForestImputer


def process_and_impute(discharge_file, latlong_file, contrib_file):
    """
    Main inference function for Gradio:
    Takes uploaded CSV file paths, processes them, runs Ordered MissForest,
    and saves the imputed dataset to a downloadable CSV.
    """
    if not discharge_file or not latlong_file:
        return None, "Error: Discharge and Lat/Long files are required."

    try:
        # 1. Load and preprocess the uploaded files
        df_original, df_contrib, df_coords, vcode_to_station, station_to_vcode = load_and_preprocess_data(
            discharge_file.name,
            latlong_file.name,
            contrib_file.name if contrib_file else None
        )

        if df_original is None or df_coords is None:
            return None, "Error parsing input data. Please ensure CSV formats match the standard templates."

        # 2. Add temporal features
        df_with_features = add_temporal_features(df_original)

        # 3. Identify columns
        all_cols = df_with_features.columns.tolist()
        discharge_cols = [c for c in all_cols if not c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
        temporal_features = [c for c in all_cols if c not in discharge_cols]
        all_stations = sorted(discharge_cols)

        # 4. Build spatial matrices
        distance_matrix = build_distance_matrix(df_coords, all_stations).loc[all_stations, all_stations]
        
        if df_contrib is not None:
             connectivity_matrix = build_connectivity_matrix(df_contrib, all_stations, station_to_vcode).loc[all_stations, all_stations]
        else:
             connectivity_matrix = None

        # 5. Run Ordered MissForest
        # We process the entire uploaded dataset as one block for inference since we don't know the size
        print("Starting Ordered MissForest imputation on uploaded data...")
        imputer = MissForestImputer(
            distance_matrix=distance_matrix,
            connectivity=connectivity_matrix,
            max_iter=10,
            n_estimators=100,
            random_state=42,
            distance_weighting_type='inverse',
            temporal_feature_columns=temporal_features,
            initialization_method='historical_mean',
            ordering_method='most_full_first'
        )
        
        # Fit and transform
        imputer.fit(df_with_features)
        df_imputed = imputer.transform(df_with_features)

        # 6. Save and return the output
        output_filename = "imputed_discharge_data.csv"
        df_imputed.to_csv(output_filename)
        
        summary = f"Successfully imputed {len(discharge_cols)} stations across {len(df_imputed)} days."
        return output_filename, summary

    except Exception as e:
        return None, f"An error occurred during processing: {str(e)}"


# Define Gradio Interface UI
with gr.Blocks(title="Streamflow Imputation Model (Ordered MissForest)") as app:
    gr.Markdown("# 🌊 Streamflow Discharge Imputation Server")
    gr.Markdown(
        "Upload your discharge dataset with missing values, along with station coordinates and (optionally) "
        "the river contributor matrix. The server will use an **Ordered MissForest** model enhanced with "
        "spatial and temporal features to impute all missing data."
    )
    
    with gr.Row():
        with gr.Column():
            discharge_input = gr.File(label="Discharge Data (CSV)", file_types=[".csv"])
            latlong_input = gr.File(label="Lat/Long Coordinates (CSV)", file_types=[".csv"])
            contrib_input = gr.File(label="Contributor Matrix (CSV - Optional)", file_types=[".csv"])
            submit_btn = gr.Button("Run Imputation", variant="primary")
            
        with gr.Column():
            status_output = gr.Textbox(label="Status / Summary", interactive=False)
            file_output = gr.File(label="Download Imputed CSV")

    submit_btn.click(
        fn=process_and_impute,
        inputs=[discharge_input, latlong_input, contrib_input],
        outputs=[file_output, status_output]
    )

if __name__ == "__main__":
    app.launch()
