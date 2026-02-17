import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from custom_missforest import CustomMissForest
from simplified_utils import initialize_for_missforest

class OrderedMissForest(CustomMissForest):
    """
    A variation of CustomMissForest that orders the training loop based on station completeness.
    """
    
    def __init__(self, distance_matrix, connectivity, max_iter=10, n_estimators=100, 
                 random_state=42, distance_weighting_type='inverse', decay_rate=0.1, 
                 temporal_feature_columns=None, initialization_method='column_mean',
                 ordering_method='most_full_first'):
        """
        Args:
            ordering_method: 
                - 'most_full_first': Train stations with fewest missing values first (Ascending NaNs).
                - 'least_full_first': Train stations with most missing values first (Descending NaNs).
                - 'none': Use standard dataset column order.
        """
        super().__init__(distance_matrix, connectivity, max_iter, n_estimators, 
                        random_state, distance_weighting_type, decay_rate, 
                        temporal_feature_columns, initialization_method)
        self.ordering_method = ordering_method

    def fit(self, X_incomplete):
        """Train RandomForest models with defined column ordering."""
        X = X_incomplete.copy()
        self.col_names = X.columns.tolist()
        
        # Identify discharge columns
        base_discharge_cols = [col for col in self.col_names if col not in self.temporal_feature_columns]
        
        # --- NEW LOGIC: SORT COLUMNS BY MISSINGNESS ---
        missing_counts = X_incomplete[base_discharge_cols].isna().sum()
        
        if self.ordering_method == 'most_full_first':
            # Ascending: Fewest NaNs (Most Full) -> Most NaNs (Least Full)
            self.discharge_columns = missing_counts.sort_values(ascending=True).index.tolist()
            print("INFO: Stations ordered by 'Most Full First' (Ascending Missing Count).")
        elif self.ordering_method == 'least_full_first':
            # Descending: Most NaNs -> Fewest NaNs
            self.discharge_columns = missing_counts.sort_values(ascending=False).index.tolist()
            print("INFO: Stations ordered by 'Least Full First' (Descending Missing Count).")
        else:
            # Standard/None
            self.discharge_columns = base_discharge_cols
            print("INFO: Stations using standard column order.")
        # ----------------------------------------------

        self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}

        # Store original missing mask
        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        total_original_nans = original_missing_mask.sum().sum()
        
        # Initialize missing values
        print(f"Initializing missing values using {self.initialization_method}...")
        X_initialized = initialize_for_missforest(
            X_incomplete, 
            self.discharge_columns, 
            self.initialization_method
        )
        self.initialized_data = X_initialized.copy()
        
        # Store column means
        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()
        
        X = X_initialized

        # Iterative training loop (Now using the sorted self.discharge_columns)
        for iteration in range(self.max_iter):
            print(f"MissForest ({self.ordering_method}) iteration {iteration + 1}/{self.max_iter}")
            
            previous_values = X[self.discharge_columns].copy()
            
            for col_name in self.discharge_columns:
                # Predictors are all OTHER stations + temporal features
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                
                if temporal_predictors:
                    if not X_predictors_combined.empty:
                        X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1)
                    else:
                        X_predictors_combined = X[temporal_predictors]
                
                # Skip if no valid predictors
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                     self.models[col_name] = None
                     continue

                # Train model
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators, 
                    random_state=self.random_state, 
                    max_features='sqrt',
                    n_jobs=-1  # Speed up training
                )
                
                # Fit on all available data (imputed or observed)
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
                
                # Update missing values immediately for the next station in the loop to use
                missing_mask = original_missing_mask[col_name]
                if missing_mask.sum() > 0:
                    predictions = model.predict(X_predictors_combined[missing_mask])
                    X.loc[missing_mask, col_name] = predictions
            
            # Check convergence
            current_values = X[self.discharge_columns]
            max_change = np.abs(current_values - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            
            if max_change < 1e-6:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self