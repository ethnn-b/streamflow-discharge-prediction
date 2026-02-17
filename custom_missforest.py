# custom_missforest.py - Custom MissForest with different initialization methods
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from missforest_imputer import ModifiedMissForest
from simplified_utils import initialize_for_missforest

class CustomMissForest(ModifiedMissForest):
    """
    Custom MissForest that allows different initialization methods instead of just column means.
    """
    
    def __init__(self, distance_matrix, connectivity, max_iter=10, n_estimators=100, 
                 random_state=42, distance_weighting_type='inverse', decay_rate=0.1, 
                 temporal_feature_columns=None, initialization_method='column_mean'):
        """
        Initialize CustomMissForest with different initialization options.
        
        Args:
            initialization_method: Method to initialize missing values
                - 'column_mean': Use column means (default MissForest behavior)
                - 'historical_mean': Use historical mean for same day of year
                - 'seasonal_mean': Use seasonal mean within a window
        """
        super().__init__(distance_matrix, connectivity, max_iter, n_estimators, 
                        random_state, distance_weighting_type, decay_rate, temporal_feature_columns)
        self.initialization_method = initialization_method
        self.initialized_data = None
        
    def fit(self, X_incomplete):
        """Train RandomForest models with custom initialization."""
        X = X_incomplete.copy()
        self.col_names = X.columns.tolist()
        self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
        self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}

        # Store original missing mask for later use
        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        total_original_nans = original_missing_mask.sum().sum()
        
        if total_original_nans == 0:
            # No missing values, train models directly
            for col_name in self.discharge_columns:
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X[temporal_predictors]
                
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                     self.models[col_name] = None
                     continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
            return self

        # Initialize missing values using custom method
        print(f"Initializing missing values using {self.initialization_method} method...")
        X_initialized = initialize_for_missforest(
            X_incomplete, 
            self.discharge_columns, 
            self.initialization_method
        )
        self.initialized_data = X_initialized.copy()
        
        # Store column means for consistency (used in original implementation)
        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()
        
        # Verify initialization is complete (no NaN values)
        total_nans = X_initialized[self.discharge_columns].isnull().sum().sum()
        if total_nans > 0:
            print(f"ERROR: {total_nans} NaN values remain after initialization!")
            raise ValueError(f"Initialization failed: {total_nans} NaN values remain")
        
        # Use initialized data for training
        X = X_initialized

        # Continue with original MissForest training logic
        for iteration in range(self.max_iter):
            print(f"MissForest iteration {iteration + 1}/{self.max_iter}")
            
            # Track convergence
            previous_values = X[self.discharge_columns].copy()
            
            for col_name in self.discharge_columns:
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X[temporal_predictors]
                
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                    self.models[col_name] = None
                    continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
                
                # Predict and update missing values
                missing_mask = original_missing_mask[col_name]
                if missing_mask.sum() > 0:
                    predictions = model.predict(X_predictors_combined[missing_mask])
                    X.loc[missing_mask, col_name] = predictions
            
            # Check for convergence
            current_values = X[self.discharge_columns]
            max_change = np.abs(current_values - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            
            if max_change < 1e-6:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def transform(self, X_incomplete):
        """Transform new data using trained models."""
        X = X_incomplete.copy()
        
        # Initialize missing values using the same method used in training
        print(f"Initializing test data using {self.initialization_method} method...")
        X_initialized = initialize_for_missforest(
            X_incomplete, 
            self.discharge_columns, 
            self.initialization_method
        )
        X = X_initialized
        
        # Track which values were originally missing
        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        
        # Apply iterative imputation
        for iteration in range(self.max_iter):
            print(f"Test imputation iteration {iteration + 1}/{self.max_iter}")
            
            previous_values = X[self.discharge_columns].copy()
            
            for col_name in self.discharge_columns:
                if self.models[col_name] is None:
                    continue
                    
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1) if not X_predictors_combined.empty else X[temporal_predictors]
                
                # Predict for originally missing values
                missing_mask = original_missing_mask[col_name]
                if missing_mask.sum() > 0:
                    predictions = self.models[col_name].predict(X_predictors_combined[missing_mask])
                    X.loc[missing_mask, col_name] = predictions
            
            # Check for convergence
            current_values = X[self.discharge_columns]
            max_change = np.abs(current_values - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            
            if max_change < 1e-6:
                print(f"Test imputation converged after {iteration + 1} iterations")
                break
        
        return X
