import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.imputers.baselines import initialize_for_missforest

class MissForestImputer:
    """
    Unified MissForest that handles:
    1. Vanilla MissForest (no distance/connectivity, simple mean initialization, no ordering)
    2. Custom MissForest (temporal features, distance/connectivity weighting)
    3. Ordered MissForest (station imputation order based on missingness)
    """
    def __init__(self, distance_matrix=None, connectivity=None, max_iter=10, n_estimators=100,
                 random_state=42, distance_weighting_type='inverse', decay_rate=0.1,
                 temporal_feature_columns=None, initialization_method='column_mean',
                 ordering_method='none'):
        self.vanilla_mode = (distance_matrix is None or connectivity is None or (distance_matrix.empty and connectivity.empty))
        self.distance_matrix = distance_matrix if distance_matrix is not None else pd.DataFrame()
        self.connectivity = connectivity if connectivity is not None else pd.DataFrame()
        
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.distance_weighting_type = distance_weighting_type
        self.decay_rate = decay_rate
        self.temporal_feature_columns = temporal_feature_columns if temporal_feature_columns is not None else []
        self.initialization_method = initialization_method
        self.ordering_method = ordering_method
        
        self.models = {}
        self.col_means = {}
        self.discharge_columns = None

    def _calculate_weights(self, target_station, station_predictors):
        """Calculates blended spatial and hydrological weights for predictors."""
        if self.vanilla_mode:
            return pd.Series(dtype=float)

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
        X = X_incomplete.copy()
        base_discharge_cols = [col for col in X.columns if col not in self.temporal_feature_columns]

        # Determine column order
        missing_counts = X_incomplete[base_discharge_cols].isna().sum()
        if self.ordering_method == 'most_full_first':
            self.discharge_columns = missing_counts.sort_values(ascending=True).index.tolist()
            print("INFO: Stations ordered by 'Most Full First'.")
        elif self.ordering_method == 'least_full_first':
            self.discharge_columns = missing_counts.sort_values(ascending=False).index.tolist()
            print("INFO: Stations ordered by 'Least Full First'.")
        else:
            self.discharge_columns = base_discharge_cols

        original_missing_mask = X_incomplete[self.discharge_columns].isna()

        # Initialization
        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()

        if self.initialization_method == 'column_mean' or self.vanilla_mode:
            print("Initializing missing values using column_mean method...")
            for col in self.discharge_columns:
                mean_val = self.col_means.get(col, 0)
                if pd.isna(mean_val): mean_val = 0
                X[col] = X[col].fillna(mean_val)
        else:
            print(f"Initializing missing values using {self.initialization_method}...")
            X = initialize_for_missforest(X_incomplete, self.discharge_columns, self.initialization_method)

        for iteration in range(self.max_iter):
            print(f"MissForest iteration {iteration + 1}/{self.max_iter}")
            previous_values = X[self.discharge_columns].copy()

            for col_name in self.discharge_columns:
                y_known = X_incomplete.loc[~original_missing_mask[col_name], col_name]
                if y_known.empty:
                    self.models[col_name] = None
                    continue

                station_predictors = [c for c in self.discharge_columns if c != col_name]
                weights_for_stations = self._calculate_weights(col_name, station_predictors)

                if not weights_for_stations.empty:
                    X_predictors_stations = X[station_predictors].multiply(weights_for_stations, axis=1)
                else:
                    X_predictors_stations = X[station_predictors]

                X_predictors_combined = pd.concat([X_predictors_stations, X[self.temporal_feature_columns]], axis=1) if len(self.temporal_feature_columns) > 0 else X_predictors_stations
                
                X_train_for_model = X_predictors_combined.loc[y_known.index]
                if X_train_for_model.empty or (X_train_for_model == 0).all().all():
                    self.models[col_name] = None
                    continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt', n_jobs=-1)
                model.fit(X_train_for_model, y_known)
                self.models[col_name] = model

                missing_in_col = original_missing_mask[col_name]
                if missing_in_col.any():
                    X_predict_for_model = X_predictors_combined.loc[missing_in_col]
                    if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                        X.loc[missing_in_col, col_name] = model.predict(X_predict_for_model)

            max_change = np.abs(X[self.discharge_columns] - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            if max_change < 1e-6:
                print(f"Converged after {iteration + 1} iterations")
                break

        return self

    def transform(self, X_incomplete):
        X_imp = X_incomplete.copy()

        if self.discharge_columns is None:
            print("Warning: Imputer not trained. Filling with column means.")
            self.discharge_columns = [col for col in X_incomplete.columns if col not in self.temporal_feature_columns]
            for col in self.discharge_columns:
                 X_imp[col] = X_imp[col].fillna(self.col_means.get(col, 0))
            return X_imp

        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        if not missing_mask_new_data.sum().sum():
            return X_imp

        # Initialization
        if self.initialization_method == 'column_mean' or self.vanilla_mode:
            print("Initializing test data using column_mean method...")
            for col in self.discharge_columns:
                X_imp[col] = X_imp[col].fillna(self.col_means.get(col, 0))
        else:
            print(f"Initializing test data using {self.initialization_method}...")
            # For testing, we use the training set means for initialization where possible? 
            # In standard missforest transit it just initializes the new data.
            X_imp = initialize_for_missforest(X_incomplete, self.discharge_columns, self.initialization_method)

        for iteration in range(self.max_iter):
            print(f"Test imputation iteration {iteration + 1}/{self.max_iter}")
            previous_values = X_imp[self.discharge_columns].copy()

            for col_name in self.discharge_columns:
                if col_name not in self.models or self.models[col_name] is None:
                    continue

                missing_in_col = missing_mask_new_data[col_name]
                if not missing_in_col.any(): 
                    continue

                station_predictors = [c for c in self.discharge_columns if c != col_name]
                weights_for_stations = self._calculate_weights(col_name, station_predictors)

                if not weights_for_stations.empty:
                    X_predictors_stations = X_imp[station_predictors].multiply(weights_for_stations, axis=1)
                else:
                    X_predictors_stations = X_imp[station_predictors]

                X_predictors_combined = pd.concat([X_predictors_stations, X_imp[self.temporal_feature_columns]], axis=1) if len(self.temporal_feature_columns) > 0 else X_predictors_stations
                X_predict_for_model = X_predictors_combined.loc[missing_in_col]

                if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                    X_imp.loc[missing_in_col, col_name] = self.models[col_name].predict(X_predict_for_model)
                else:
                    X_imp.loc[missing_in_col, col_name] = self.col_means.get(col_name, 0)

            max_change = np.abs(X_imp[self.discharge_columns] - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            if max_change < 1e-6:
                print(f"Test imputation converged after {iteration + 1} iterations")
                break

        return X_imp
