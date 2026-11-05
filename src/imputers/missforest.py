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
                 ordering_method='none', use_temporal_lags=True, use_routing=True,
                 use_aggregate=True, celerity_km_per_day=100.0, n_jobs=-1,
                 max_features='sqrt', lag_mode='anchor', routing_mode='real',
                 alpha=0.5):
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
        
        self.use_temporal_lags = use_temporal_lags      # L: Eqs. (14)-(15)
        self.use_routing = use_routing                  # R: Eq. (13)
        self.use_aggregate = use_aggregate              # A: Eq. (11)
        self.n_jobs = n_jobs
        self.celerity_km_per_day = celerity_km_per_day
        # 'sqrt' is sklearn's CLASSIFICATION default; R's missForest uses p/3 and
        # sklearn's regressor defaults to all p. Kept as the historical default.
        self.max_features = max_features
        # 'anchor' is the default because it IS the method: PIMF = {A, R, N}, i.e.
        # use_aggregate + use_routing + lag_mode='anchor'.
        # 'naive' is the superseded published L block, shift(+-1) of the target's
        # own column, kept only to reproduce it; see _anchor_features.
        self.lag_mode = lag_mode
        # 'placebo' replaces the real upstream set with a distance-matched
        # non-upstream set; see _placebo_set.
        self.routing_mode = routing_mode
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must lie in [0, 1], not {alpha!r}')
        self.alpha = alpha

        self.models = {}
        self.col_means = {}
        self.discharge_columns = None
        # Which cells were missing in the frame being imputed. Set by fit/transform;
        # the lag features need it to report how far each row sits from real data.
        # Knowing WHERE gaps are is not leakage - that is given at inference time -
        # but the VALUES must only ever come from X.
        self._current_missing = None
        # Columns with no observations at all in the frame being imputed. They are
        # initialized to a constant and carry no information, so they must not take
        # weight mass in the aggregate of Eq. (11) - on the Mahanadi record four
        # gauges are empty over 1980-1990 and would otherwise absorb 15% of the
        # weight on average, and 36% at the worst target.
        self._dead_columns = frozenset()

    def _set_missing(self, mask):
        self._current_missing = mask
        self._dead_columns = frozenset(mask.columns[mask.all()])

    def _calculate_weights(self, target_station, station_predictors):
        """Calculates blended spatial and hydrological weights for predictors."""
        if self.vanilla_mode:
            return pd.Series(dtype=float)

        if target_station not in self.distance_matrix.index:
            return pd.Series(0.0, index=station_predictors)

        live_predictors = [p for p in station_predictors if p not in self._dead_columns]
        actual_predictors_dist = [p for p in live_predictors if p in self.distance_matrix.columns]
        if not actual_predictors_dist:
            return pd.Series(0.0, index=station_predictors)

        distances = self.distance_matrix.loc[target_station, actual_predictors_dist]
        dist_weights = 1 / (distances + 1e-9) if self.distance_weighting_type == 'inverse' else np.exp(-self.decay_rate * distances)
        dist_weights = dist_weights.fillna(0).replace([np.inf, -np.inf], 0)

        actual_predictors_conn = [p for p in live_predictors if p in self.connectivity.columns]
        connectivity_weights = self.connectivity.loc[target_station, actual_predictors_conn] if target_station in self.connectivity.index and actual_predictors_conn else pd.Series(0.0, index=actual_predictors_conn)

        aligned_dist_weights = dist_weights.reindex(live_predictors, fill_value=0.0)
        aligned_connectivity = connectivity_weights.reindex(live_predictors, fill_value=0.0)

        aligned_dist_weights = aligned_dist_weights / aligned_dist_weights.sum() if aligned_dist_weights.sum() > 0 else pd.Series(0.0, index=live_predictors)
        aligned_connectivity = aligned_connectivity / aligned_connectivity.sum() if aligned_connectivity.sum() > 0 else pd.Series(0.0, index=live_predictors)

        # alpha = 1 is pure inverse distance, alpha = 0 is pure connectivity.
        blended_weights = (self.alpha * aligned_dist_weights
                           + (1 - self.alpha) * aligned_connectivity)

        blended_weights = blended_weights / blended_weights.sum() if blended_weights.sum() > 0 else pd.Series(0.0, index=live_predictors)
        return blended_weights.reindex(station_predictors, fill_value=0.0)

    def _mean_for(self, col, frame=None):
        """Fallback constant for a column: training mean, else this frame's mean, else 0.

        col_means[col] is NaN for a column that was entirely missing in the frame
        passed to fit(), and dict.get(col, 0) returns that NaN rather than the
        default, so every downstream fillna() silently left the cell missing and
        the metrics quietly dropped it.

        A station with no training data still usually has observed values in the
        frame being imputed - they are part of the given gapped record, not of the
        withheld truth - so prefer their mean over a bare 0, which for discharge is
        an arbitrary penalty rather than a neutral guess.
        """
        m = self.col_means.get(col, np.nan)
        if not pd.isna(m):
            return m
        if frame is not None and col in frame.columns:
            m = frame[col].mean()
            if not pd.isna(m):
                return m
        return 0.0

    @staticmethod
    def _missing_run_length(gap):
        """Per row: how many consecutive missing rows end at this one (0 if observed)."""
        g = gap.astype(int)
        return g.groupby((g == 0).cumsum()).cumsum()

    def _lag_features(self, X, target_station):
        """The target's own neighbours in time, plus how far this row is from real data.

        Streamflow is strongly autocorrelated, so the values bracketing a gap are
        the most informative predictors at short gaps - which is why linear
        interpolation beats the unlagged model at 3-day gaps. Values are read from
        X, the live iterate, never the complete series, so nothing withheld leaks in.
        """
        q = X[target_station]
        feats = {'lag_prev': q.shift(1), 'lag_next': q.shift(-1)}
        mask = self._current_missing
        if mask is not None and target_station in mask.columns:
            gap = mask[target_station]
            feats['gap_dist_prev'] = self._missing_run_length(gap)
            feats['gap_dist_next'] = self._missing_run_length(gap[::-1])[::-1]
        return {k: v.bfill().ffill().fillna(0.0) for k, v in feats.items()}

    @staticmethod
    def _last_obs_pos(obs):
        """Position of the last observed row at or before each row (-1 if none)."""
        idx = np.where(obs, np.arange(len(obs)), -1)
        return np.maximum.accumulate(idx)

    # Pretend gap lengths assigned to TRAINING rows. On a training row the target is
    # observed, so the true distance to the nearest reading is 1 and the distance
    # features are constant - the forest then scores them at exactly zero importance
    # and never learns to lean on a far-away anchor, which is the only case it meets
    # inside a long gap. Each training row is instead given the geometry of a real
    # gap: a length from this log-spaced ladder and a position inside it, so that
    # dist_prev + dist_next - 1 is that length, exactly as it is for a cell being
    # imputed. Pairing the two sides independently would never put both distances
    # near each other, which is every cell of a short gap. The ladder is deliberately
    # NOT drawn from the record's own missing runs: that collapses to distance 1 at a
    # station with no native gaps, reinstating the constant-distance failure above,
    # and ties the training features to where the test gaps happen to be.
    ANCHOR_TRAIN_GAPS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)

    def _anchor_side(self, q, obs, depths):
        """Value of, and distance to, the nearest observed row at least `depths` back."""
        n = len(q)
        pos = np.arange(n)
        src = pos - depths
        last = self._last_obs_pos(obs)
        j = np.where(src >= 0, last[np.clip(src, 0, n - 1)], -1)
        val = np.where(j >= 0, q[np.clip(j, 0, n - 1)], np.nan)
        dist = np.where(j >= 0, (pos - j).astype(float), np.nan)
        return val, dist

    def _anchor_features(self, X, target_station, synth_gaps):
        """Last and next genuinely observed reading of the target, and their distances.

        Replaces the shift(+-1) lags of _lag_features, which inside a gap read this
        imputer's own output from the previous round and so compound their own error.
        An anchor is by construction an observed cell of the given record: never a
        value this model produced, and never a withheld one.
        """
        q = X[target_station].values.astype(float)
        mask = self._current_missing
        miss = (mask[target_station].values if mask is not None
                and target_station in mask.columns else np.zeros(len(q), bool))
        obs = (~miss) & np.isfinite(q)
        n = len(q)
        if synth_gaps:
            ladder = self.ANCHOR_TRAIN_GAPS
            seed = (None if self.random_state is None
                    else [self.random_state, sum(map(ord, target_station))])
            rng = np.random.default_rng(seed)
            L = rng.choice(ladder, n)
            dp = rng.integers(1, L + 1)
            dn = L + 1 - dp
        else:
            dp = dn = np.ones(n, dtype=int)
        vp, sp = self._anchor_side(q, obs, dp)
        vn, sn = self._anchor_side(q[::-1], obs[::-1], dn[::-1])
        out = {'anchor_prev': vp, 'anchor_next': vn[::-1],
               'anchor_dist_prev': sp, 'anchor_dist_next': sn[::-1]}
        return {k: pd.Series(v, index=X.index).bfill().ffill().fillna(0.0)
                for k, v in out.items()}

    def _placebo_set(self, target_station, weights, upstream, lagmap):
        """Non-upstream gauges matched to the real contributors on distance, count,
        weight and travel-time lag.

        Isolates the river map from simple proximity: if the routed term works only
        because nearby gauges correlate with the target, a set matched on everything
        except being upstream must score the same.

        shortcut: where the non-upstream pool is smaller than the real upstream set
        the match is partial and the weights are renormalised to the real total; the
        matched count is returned so it can be reported.
        """
        D = self.distance_matrix
        if target_station not in D.index:
            return [], weights, lagmap
        real = sorted(upstream, key=lambda j: D.loc[target_station, j])
        pool = [j for j in weights.index
                if j != target_station and j not in upstream
                and j not in self._dead_columns and j in D.columns
                and np.isfinite(D.loc[target_station, j])]
        picked, used = [], set()
        for j in real:
            cand = [c for c in pool if c not in used]
            if not cand:
                break
            c = min(cand, key=lambda c: abs(D.loc[target_station, c] - D.loc[target_station, j]))
            used.add(c)
            picked.append((c, j))
        if not picked:
            return [], weights, lagmap
        w = pd.Series(0.0, index=weights.index, dtype=float)
        total = float(weights[real].sum())
        raw = np.array([float(weights[j]) for _, j in picked])
        if raw.sum() > 0:
            raw = raw * (total / raw.sum())
        for (c, _), x in zip(picked, raw):
            w[c] = x
        return [c for c, _ in picked], w, {c: lagmap[j] for c, j in picked}

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
                    if j != target_station and j in row.index and row[j] > 0
                    and j not in self._dead_columns]
        if not upstream:
            return None
        lagmap = {}
        for j in upstream:
            lag = 0
            if (target_station in self.distance_matrix.index
                    and j in self.distance_matrix.columns):
                d = self.distance_matrix.loc[target_station, j]
                if np.isfinite(d) and self.celerity_km_per_day > 0:
                    lag = int(round(d / self.celerity_km_per_day))
            lagmap[j] = lag
        if self.routing_mode == 'placebo':
            upstream, weights, lagmap = self._placebo_set(
                target_station, weights, upstream, lagmap)
            if not upstream:
                return None
        total = None
        for j in upstream:
            term = X[j].shift(lagmap[j]) * weights[j]
            total = term if total is None else total + term
        return total.bfill().ffill().fillna(0.0)

    def _build_predictors(self, X, target_station, station_predictors, synth_gaps=False):
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
            # A and R are independent: the routed term of Eq. (13) needs the weights
            # but not the aggregate of Eq. (11), so {R} and {R,L} are real arms.
            if self.use_aggregate:
                predictors = predictors.assign(
                    weighted_upstream_discharge=predictors.mul(weights, axis=1).sum(axis=1)
                )
            if self.use_routing:
                routed = self._routed_upstream(X, target_station, weights)
                if routed is not None:
                    predictors = predictors.assign(routed_upstream_discharge=routed)
        # Vanilla mode must stay conventional MissForest: no engineered features at
        # all, or the ablation baseline silently gains the lags too.
        if self.use_temporal_lags and not self.vanilla_mode:
            if self.lag_mode == 'anchor':
                predictors = predictors.assign(
                    **self._anchor_features(X, target_station, synth_gaps))
            else:
                predictors = predictors.assign(**self._lag_features(X, target_station))
        if len(self.temporal_feature_columns) > 0:
            predictors = pd.concat([predictors, X[self.temporal_feature_columns]], axis=1)
        return predictors

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
        self._set_missing(original_missing_mask)

        # Initialization
        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()

        if self.initialization_method == 'column_mean' or self.vanilla_mode:
            print("Initializing missing values using column_mean method...")
            for col in self.discharge_columns:
                X[col] = X[col].fillna(self._mean_for(col))
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
                
                # The forest is trained on rows carrying simulated anchor distances and
                # applied to rows carrying their real ones; everything else is identical.
                train_src = X_predictors_combined
                if (self.lag_mode == 'anchor' and self.use_temporal_lags
                        and not self.vanilla_mode):
                    train_src = self._build_predictors(X, col_name, station_predictors,
                                                       synth_gaps=True)
                X_train_for_model = train_src.loc[y_known.index]
                if X_train_for_model.empty or (X_train_for_model == 0).all().all():
                    self.models[col_name] = None
                    continue
                model = RandomForestRegressor(n_estimators=self.n_estimators,
                                              random_state=self.random_state,
                                              max_features=self.max_features, n_jobs=self.n_jobs)
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
                 X_imp[col] = X_imp[col].fillna(self._mean_for(col, X_incomplete))
            return X_imp

        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        self._set_missing(missing_mask_new_data)
        if not missing_mask_new_data.sum().sum():
            return X_imp

        # Initialization
        if self.initialization_method == 'column_mean' or self.vanilla_mode:
            print("Initializing test data using column_mean method...")
            for col in self.discharge_columns:
                X_imp[col] = X_imp[col].fillna(self._mean_for(col, X_incomplete))
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
                    X_imp.loc[missing_in_col, col_name] = \
                        self.models[col_name].predict(X_predict_for_model)
                else:
                    X_imp.loc[missing_in_col, col_name] = self._mean_for(col_name, X_incomplete)

            max_change = np.abs(X_imp[self.discharge_columns] - previous_values).max().max()
            print(f"  Maximum change: {max_change:.6f}")
            if max_change < 1e-6:
                print(f"Test imputation converged after {iteration + 1} iterations")
                break

        return X_imp
