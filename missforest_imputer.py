# missforest_imputer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

class ModifiedMissForest:
    """
    Custom MissForest for streamflow imputation with spatial/hydrological weighting.
    """
    def __init__(self, distance_matrix, connectivity, max_iter=10, n_estimators=100, random_state=42,
                 distance_weighting_type='inverse', decay_rate=0.1, temporal_feature_columns=None):
        
        # --- NEW: Check for vanilla mode ---
        if distance_matrix is None or connectivity is None:
            print("INFO: No distance or connectivity matrix provided. Running in 'Vanilla MissForest' mode (no weighting).")
            self.vanilla_mode = True
            # Create dummy matrices to satisfy the old logic, but they won't be used
            # if we check self.vanilla_mode first.
            self.distance_matrix = pd.DataFrame()
            self.connectivity = pd.DataFrame()
        else:
            self.vanilla_mode = False
            self.distance_matrix = distance_matrix
            self.connectivity = connectivity
        # --- End NEW ---
            
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.col_means = {}
        self.col_names = None
        self.discharge_columns = None
        self.temporal_feature_columns = temporal_feature_columns if temporal_feature_columns is not None else []
        self.site_to_idx = None
        self.distance_weighting_type = distance_weighting_type
        self.decay_rate = decay_rate

    def _calculate_weights(self, target_station, station_predictors):
        """Calculates blended spatial and hydrological weights for predictors."""
        
        # --- NEW: Check for vanilla mode ---
        if self.vanilla_mode:
            # In vanilla mode, all predictors are weighted equally (or not at all)
            # Returning an empty series will skip the .multiply() step later
            return pd.Series()
        # --- End NEW ---

        if target_station not in self.distance_matrix.index:
            return pd.Series(0.0, index=station_predictors)

        actual_predictors_dist = [p for p in station_predictors if p in self.distance_matrix.columns]
        if not actual_predictors_dist:
            return pd.Series(0.0, index=station_predictors)

        distances = self.distance_matrix.loc[target_station, actual_predictors_dist]
        dist_weights = 1 / (distances + 1e-9) if self.distance_weighting_type == 'inverse' else np.exp(-self.decay_rate * distances)
        dist_weights = dist_weights.fillna(0).replace([np.inf, -np.inf], 0)

        actual_predictors_conn = [p for p in station_predictors if p in self.connectivity.columns]
        connectivity_weights = self.connectivity.loc[target_station, actual_predictors_conn] if target_station in self.connectivity.index and actual_predictors_conn else pd.Series(0.0, index=actual_predictors_conn)

        aligned_dist_weights = dist_weights.reindex(station_predictors, fill_value=0.0)
        aligned_connectivity = connectivity_weights.reindex(station_predictors, fill_value=0.0)

        aligned_dist_weights = aligned_dist_weights / aligned_dist_weights.sum() if aligned_dist_weights.sum() > 0 else pd.Series(0.0, index=station_predictors)
        aligned_connectivity = aligned_connectivity / aligned_connectivity.sum() if aligned_connectivity.sum() > 0 else pd.Series(0.0, index=station_predictors)

        alpha = 0.5
        blended_weights = alpha * aligned_dist_weights + (1 - alpha) * aligned_connectivity

        return blended_weights / blended_weights.sum() if blended_weights.sum() > 0 else pd.Series(0.0, index=station_predictors)

    def fit(self, X_incomplete):
        """Trains RandomForest models iteratively to impute missing values."""
        X = X_incomplete.copy()
        self.col_names = X.columns.tolist()
        self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
        self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}

        # Initial mean imputation
        for col in self.discharge_columns:
            # Use mean of the *incomplete* data for initialization and fallback
            self.col_means[col] = X_incomplete[col].mean()
            X[col] = X[col].fillna(self.col_means[col])

        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        total_original_nans = original_missing_mask.sum().sum()
        
        if total_original_nans == 0:
            print("  No missing values in training data. Training models on full data.")
            # Still need to train models for the transform() step
            for col_name in self.discharge_columns:
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                # --- MODIFIED: Handle weighting vs. vanilla ---
                if not weights_for_stations.empty:
                    # Weighted mode
                    X_predictors_stations = X[station_predictors].multiply(weights_for_stations, axis=1)
                else:
                    # Vanilla mode (or no valid weights)
                    X_predictors_stations = X[station_predictors]

                X_predictors_combined = pd.concat([X_predictors_stations, X[temporal_predictors]], axis=1)
                # --- End MODIFIED ---
                
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                     self.models[col_name] = None
                     continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
            return self

        # Iterative imputation process
        for iteration in range(self.max_iter):
            X_prev = X.copy()
            
            # --- MODIFIED: Forward and Backward Pass ---
            # Forward pass
            for col_name in self.discharge_columns:
                X, y_known = self._fit_model_for_column(X, X_incomplete, original_missing_mask, col_name)

            # Backward pass
            for col_name in reversed(self.discharge_columns):
                X, y_known = self._fit_model_for_column(X, X_incomplete, original_missing_mask, col_name)
            # --- End MODIFIED ---

            # Check for convergence
            current_imputed_vals = X[self.discharge_columns][original_missing_mask].stack().values
            prev_imputed_vals = X_prev[self.discharge_columns][original_missing_mask].stack().values

            change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals) if current_imputed_vals.size > 0 else 0.0
            if change_norm < 1e-6:
                # print(f"  Converged after {iteration + 1} iterations.")
                break
        return self

    def _fit_model_for_column(self, X, X_incomplete, original_missing_mask, col_name):
        """Helper function for the iterative fit loop."""
        
        # We always train on the *known* values from the original data
        y_known = X_incomplete.loc[~original_missing_mask[col_name], col_name]
        if y_known.empty:
            self.models[col_name] = None
            return X, y_known # Return X unchanged

        station_predictors = [c for c in self.discharge_columns if c != col_name]
        temporal_predictors = self.temporal_feature_columns
        weights_for_stations = self._calculate_weights(col_name, station_predictors)
        
        # --- MODIFIED: Handle weighting vs. vanilla ---
        if not weights_for_stations.empty:
            # Weighted mode
            X_predictors_stations = X[station_predictors].multiply(weights_for_stations, axis=1)
        else:
            # Vanilla mode (or no valid weights)
            X_predictors_stations = X[station_predictors]

        X_predictors_combined = pd.concat([X_predictors_stations, X[temporal_predictors]], axis=1)
        # --- End MODIFIED ---
        
        X_train_for_model = X_predictors_combined.loc[y_known.index]
        y_train_for_model = y_known.loc[X_train_for_model.index]

        if X_train_for_model.empty or y_train_for_model.empty or (X_train_for_model == 0).all().all():
            self.models[col_name] = None
            return X, y_known # Return X unchanged

        model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
        model.fit(X_train_for_model, y_train_for_model)
        self.models[col_name] = model

        # Update the *imputed* values (originally missing)
        missing_in_col = original_missing_mask[col_name]
        if missing_in_col.any():
            X_predict_for_model = X_predictors_combined.loc[missing_in_col]
            if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                X.loc[X_predict_for_model.index, col_name] = model.predict(X_predict_for_model)
            else:
                # Fallback to mean if predictors are empty
                X.loc[missing_in_col, col_name] = self.col_means.get(col_name, 0)
        
        return X, y_known

    def transform(self, X_incomplete):
        """Imputes missing values in new data using trained models."""
        X_imp = X_incomplete.copy()

        if self.col_names is None: # Untrained imputer fallback
            print("Warning: Imputer not trained. Filling with column means.")
            self.col_names = X_incomplete.columns.tolist()
            self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
            self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}
            for col in self.discharge_columns:
                self.col_means[col] = X_incomplete[col].mean()
            return X_imp[self.discharge_columns].fillna(self.col_means)

        # Initial mean imputation
        for col in self.discharge_columns:
            # Fill with means calculated during fit()
            X_imp[col] = X_imp[col].fillna(self.col_means.get(col, 0))

        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        if not missing_mask_new_data.sum().sum():
            return X_imp # No missing data, return filled data

        for iteration in range(self.max_iter):
            X_prev_imp = X_imp.copy()

            # --- MODIFIED: Forward and Backward Pass ---
            # Forward pass
            for col_name in self.discharge_columns:
                X_imp = self._transform_column(X_imp, missing_mask_new_data, col_name)

            # Backward pass
            for col_name in reversed(self.discharge_columns):
                X_imp = self._transform_column(X_imp, missing_mask_new_data, col_name)
            # --- End MODIFIED ---

            # Check for convergence
            current_imputed_vals = X_imp[self.discharge_columns][missing_mask_new_data].stack().values
            prev_imputed_vals = X_prev_imp[self.discharge_columns][missing_mask_new_data].stack().values

            change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals) if current_imputed_vals.size > 0 else 0.0
            if change_norm < 1e-6:
                # print(f"  Transform converged after {iteration + 1} iterations.")
                break
        return X_imp

    def _transform_column(self, X_imp, missing_mask_new_data, col_name):
        """Helper function for the iterative transform loop."""
        
        if col_name not in self.models or self.models[col_name] is None:
            return X_imp # Return X_imp unchanged

        missing_in_col = missing_mask_new_data[col_name]
        if not missing_in_col.any(): 
            return X_imp # Return X_imp unchanged

        station_predictors = [c for c in self.discharge_columns if c != col_name]
        temporal_predictors = self.temporal_feature_columns
        weights_for_stations = self._calculate_weights(col_name, station_predictors)
        
        # --- MODIFIED: Handle weighting vs. vanilla ---
        if not weights_for_stations.empty:
            # Weighted mode
            X_predictors_stations = X_imp[station_predictors].multiply(weights_for_stations, axis=1)
        else:
            # Vanilla mode (or no valid weights)
            X_predictors_stations = X_imp[station_predictors]

        X_predictors_combined = pd.concat([X_predictors_stations, X_imp[temporal_predictors]], axis=1)
        # --- End MODIFIED ---

        X_predict_for_model = X_predictors_combined.loc[missing_in_col]

        if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
            X_imp.loc[X_predict_for_model.index, col_name] = self.models[col_name].predict(X_predict_for_model)
        else:
            # Fallback to mean if predictors are empty
            X_imp.loc[missing_in_col, col_name] = self.col_means.get(col_name, 0)
            
        return X_imp

