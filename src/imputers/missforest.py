import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.imputers.baselines import initialize_for_missforest

# Distance reported when a row has no observation on one side (a gap at the record edge).
# Only its ordering matters to the trees: it just has to exceed every real distance.
NO_OBSERVATION_DISTANCE = 10_000


class MissForestImputer:
    """
    Unified MissForest that handles:
    1. Vanilla MissForest (no distance/connectivity, simple mean initialization, no ordering)
    2. PIMF (routing, weighted basin aggregate, target-station temporal features, seasonality)
    3. Ordered MissForest (station imputation order based on missingness)

    n_features chooses the target station's temporal features (N):
    'nearest_observed' (PIMF, default) - the last and next *observed* value and the
        distance to each, read from the observed record, so nothing the model
        estimates is fed back to it;
    'neighbour_day' (legacy, the v5/v6 paper's PIMF) - the day before and after read
        from the live iterate, which inside long gaps feeds the model its own
        estimates and drifts upward. Kept to reproduce earlier results.
    """
    def __init__(self, distance_matrix=None, connectivity=None, max_iter=10, n_estimators=100,
                 random_state=42, distance_weighting_type='inverse', decay_rate=0.1,
                 temporal_feature_columns=None, initialization_method='column_mean',
                 ordering_method='none', use_temporal_lags=True, use_routing=True,
                 celerity_km_per_day=100.0, n_features='nearest_observed'):
        if n_features not in ('nearest_observed', 'neighbour_day'):
            raise ValueError(f"n_features must be 'nearest_observed' or 'neighbour_day', not {n_features!r}")
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
        
        self.use_temporal_lags = use_temporal_lags
        self.use_routing = use_routing
        self.celerity_km_per_day = celerity_km_per_day
        self.n_features = n_features

        self.models = {}
        self.col_means = {}
        self.discharge_columns = None
        # Which cells were missing in the frame being imputed. Set by fit/transform;
        # the lag features need it to report how far each row sits from real data.
        # Knowing WHERE gaps are is not leakage - that is given at inference time -
        # but the VALUES must only ever come from X.
        self._current_missing = None
        # The gapped frame being imputed, and the nearest-observed features built
        # from it (one frame per station). Both are reset by fit and transform.
        self._observed = None
        self._nearest_cache = {}

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

    @staticmethod
    def _missing_run_length(gap):
        """Per row: how many consecutive missing rows end at this one (0 if observed)."""
        g = gap.astype(int)
        return g.groupby((g == 0).cumsum()).cumsum()

    def _lag_features(self, X, target_station):
        if self.n_features == 'neighbour_day':
            return self._neighbour_day_features(X, target_station)
        return dict(self._nearest_observed_features(target_station).items())

    def _nearest_observed_features(self, target_station):
        """The last and next observed value of the target, and the days to each.

        Values come from the observed record, never from the iterate, so the model's
        estimates are not fed back to it and nothing drifts over the sweeps.

        A missing row gets its real nearest observations. An observed (training) row
        would always have them one day away, so the model would never learn what a
        distant observation is worth; each one is instead given a pretend gap, its
        (days before, days after) drawn from the station's own missing cells, and
        reads the observations at or beyond those distances. Training rows thus see
        the distances of the rows to be imputed. A side with no observation at all
        copies the other side's value and reports NO_OBSERVATION_DISTANCE.
        """
        if target_station in self._nearest_cache:
            return self._nearest_cache[target_station]
        q = self._observed[target_station].to_numpy(float)
        miss = np.isnan(q)
        n = len(q)
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(~miss, idx, -1))            # last observation at or before k
        nxt = np.minimum.accumulate(np.where(~miss, idx, n)[::-1])[::-1]  # first observation at or after k
        run_before = self._missing_run_length(pd.Series(miss)).to_numpy()
        run_after = self._missing_run_length(pd.Series(miss[::-1])).to_numpy()[::-1]
        pairs = np.c_[run_before[miss], run_after[miss]]
        seed = None if self.random_state is None else [self.random_state, sum(map(ord, target_station))]
        rng = np.random.default_rng(seed)
        lo, hi = last.copy(), nxt.copy()
        if pairs.size:
            o = ~miss
            d = pairs[rng.integers(len(pairs), size=o.sum())]
            lo[o] = last[np.clip(idx[o] - d[:, 0], 0, n - 1)]
            lo[o] = np.where(idx[o] - d[:, 0] < 0, -1, lo[o])
            hi[o] = nxt[np.clip(idx[o] + d[:, 1], 0, n - 1)]
            hi[o] = np.where(idx[o] + d[:, 1] > n - 1, n, hi[o])
        else:  # nothing missing: the nearest observations are the neighbouring days
            lo = np.r_[-1, last[:-1]]
            hi = np.r_[nxt[1:], n]
        has_lo, has_hi = lo >= 0, hi < n
        a_lo = np.where(has_lo, q[np.clip(lo, 0, n - 1)], np.nan)
        a_hi = np.where(has_hi, q[np.clip(hi, 0, n - 1)], np.nan)
        a_lo = np.where(has_lo, a_lo, a_hi)
        a_hi = np.where(has_hi, a_hi, a_lo)
        feats = pd.DataFrame({'anchor_prev': a_lo, 'anchor_next': a_hi,
                              'dist_prev': np.where(has_lo, idx - lo, NO_OBSERVATION_DISTANCE),
                              'dist_next': np.where(has_hi, hi - idx, NO_OBSERVATION_DISTANCE)},
                             index=self._observed.index)
        self._nearest_cache[target_station] = feats.fillna(np.nanmean(q) if (~miss).any() else 0.0)
        return self._nearest_cache[target_station]

    def _neighbour_day_features(self, X, target_station):
        """Legacy N: the target's own neighbours in time, plus how far this row is from real data.

        Values are read from X, the live iterate, never the complete series, so nothing
        withheld leaks in; but inside a gap the neighbours are the model's own
        estimates, and the gap distances are always 0 on the (observed) training rows.
        """
        q = X[target_station]
        feats = {'lag_prev': q.shift(1), 'lag_next': q.shift(-1)}
        mask = self._current_missing
        if mask is not None and target_station in mask.columns:
            gap = mask[target_station]
            feats['gap_dist_prev'] = self._missing_run_length(gap)
            feats['gap_dist_next'] = self._missing_run_length(gap[::-1])[::-1]
        return {k: v.bfill().ffill().fillna(0.0) for k, v in feats.items()}

    def _routed_upstream(self, X, target_station, weights):
        """Weighted upstream discharge lagged by travel time: sum_j w_ij * Q_j(t - tau_ij).

        The one place connectivity carries information the raw columns do not
        already hold: topology says which gauge is upstream, celerity says how long
        its water takes to arrive, so the lag indexes a different point in time
        rather than rescaling the same one.
        """
        if self.connectivity.empty or target_station not in self.connectivity.index:
            return None
        row = self.connectivity.loc[target_station]
        upstream = [j for j in weights.index
                    if j != target_station and j in row.index and row[j] > 0]
        if not upstream:
            return None
        total = None
        for j in upstream:
            lag = 0
            if (target_station in self.distance_matrix.index
                    and j in self.distance_matrix.columns):
                d = self.distance_matrix.loc[target_station, j]
                if np.isfinite(d) and self.celerity_km_per_day > 0:
                    lag = int(round(d / self.celerity_km_per_day))
            term = X[j].shift(lag) * weights[j]
            total = term if total is None else total + term
        return total.bfill().ffill().fillna(0.0)

    def _build_predictors(self, X, target_station, station_predictors):
        """Predictor matrix for one target: the raw station discharges, a
        weighted aggregate of them, and the temporal features.

        The aggregate is what carries the physical weighting. Scaling each station
        column by its own positive weight cannot: a tree splits one feature at a
        time on the ordering of its values, and positive scaling preserves that
        ordering, so the fitted trees are identical. Summing the weighted columns
        into one feature makes the weights change values the trees can actually
        split on.
        """
        predictors = X[station_predictors]
        weights = self._calculate_weights(target_station, station_predictors)
        if not weights.empty and weights.sum() > 0:
            predictors = predictors.assign(
                weighted_upstream_discharge=predictors.mul(weights, axis=1).sum(axis=1)
            )
            if self.use_routing:
                routed = self._routed_upstream(X, target_station, weights)
                if routed is not None:
                    predictors = predictors.assign(routed_upstream_discharge=routed)
        # Vanilla mode must stay conventional MissForest: no engineered features at
        # all, or the Table 4 ablation baseline silently gains the lags too.
        if self.use_temporal_lags and not self.vanilla_mode:
            predictors = predictors.assign(**self._lag_features(X, target_station))
        if len(self.temporal_feature_columns) > 0:
            predictors = pd.concat([predictors, X[self.temporal_feature_columns]], axis=1)
        return predictors

    def fit(self, X_incomplete):
        self._observed, self._nearest_cache = X_incomplete, {}
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
        self._current_missing = original_missing_mask

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
                X_predictors_combined = self._build_predictors(X, col_name, station_predictors)
                
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
        self._observed, self._nearest_cache = X_incomplete, {}
        X_imp = X_incomplete.copy()

        if self.discharge_columns is None:
            print("Warning: Imputer not trained. Filling with column means.")
            self.discharge_columns = [col for col in X_incomplete.columns if col not in self.temporal_feature_columns]
            for col in self.discharge_columns:
                 X_imp[col] = X_imp[col].fillna(self.col_means.get(col, 0))
            return X_imp

        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        self._current_missing = missing_mask_new_data
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
                X_predictors_combined = self._build_predictors(X_imp, col_name, station_predictors)
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
