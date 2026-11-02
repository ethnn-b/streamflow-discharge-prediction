"""Anchor-N / cleaned-network / placebo-routing experiment, one (seed, gap) cell.

Network: the 16 gauges of the connectivity matrix (V1-V16; the Hirakud reservoir
row V17 has no discharge series), read from --data, whose matrix has the
paramanpur -> basantpur link removed. kelo and paramanpur have no data in
1980-1990: they stay in the frame as (constant) predictor columns but are never
fitted or scored.

Gaps are placed anywhere in 1980-1990, so the 10% is of the whole record. With
--seed-block the most complete 3-year window is held out ungapped instead, as in
the paper's protocol and in the committed e22/e23 results; neither the MissForest
arms nor the KNN/Kalman baselines need it.

Arms
  vanilla        MissForest on discharge columns only (as rerun_expts vanilla_single)
  pimf           final PIMF: A, R, N = nearest observed values and distances, S
  pimf_old       the v5/v6 paper's PIMF: N read from the live iterate (n_features='neighbour_day')
  pimfL          PIMF without N
  pimfL_placebo  pimfL with R built from non-connected gauges (see PlaceboImputer)

Pickles written before the rename call the final PIMF 'pimfA' and the old one
'pimf'; analyse.py maps them to the new names.
"""
import argparse, contextlib, io, os, pickle, sys, time, warnings

p = argparse.ArgumentParser()
p.add_argument('--root'); p.add_argument('--seed', type=int); p.add_argument('--gap', type=int)
p.add_argument('--arms', default='vanilla,pimf,pimf_old,pimfL,pimfL_placebo')
p.add_argument('--njobs', type=int, default=4); p.add_argument('--out')
p.add_argument('--data', help='folder holding the mahanadi_*.csv files')
p.add_argument('--pattern', default='fixed', choices=['fixed', 'native'],
               help="native = review_expts E6 pattern: lengths drawn from the native gap runs (2-365 d)")
p.add_argument('--min-completeness', type=float, default=None,
               help='drop stations whose completeness between their first and last 1980-1990 reading '
                    'is below this percentage: they are neither filled nor used as predictors')
p.add_argument('--seed-block', action='store_true',
               help='hold the most complete 3-year window out of gapping (reproduces the committed results)')
a = p.parse_args()
sys.path.insert(0, a.root); os.chdir(a.root)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import src.imputers.missforest as mfmod
from src.imputers.missforest import MissForestImputer
from src.utils.data import (load_and_preprocess_data, add_temporal_features,
                            find_best_data_window, create_contiguous_segment_gaps_by_percent)
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix

# n_jobs does not change a fitted forest, only how it is computed.
mfmod.RandomForestRegressor = lambda **k: RandomForestRegressor(**{**k, 'n_jobs': a.njobs})

class PlaceboImputer(MissForestImputer):
    """R with the topology scrambled: each real upstream contributor j of target i
    is replaced by a gauge k that is neither upstream nor downstream of i, chosen
    to match D_ij as closely as possible. The lag and the weight of the real pair
    are kept, so the column count, the lag distribution and the weights are those
    of the real R; only which gauge's water is read differs."""
    placebo = {}

    def _routed_upstream(self, X, target_station, weights):
        if target_station not in self.placebo:
            return None
        total = None
        for k, lag, w in self.placebo[target_station]:
            term = X[k].shift(lag) * w
            total = term if total is None else total + term
        return total.bfill().ffill().fillna(0.0)


warnings.simplefilter('ignore')
quiet = contextlib.redirect_stdout(io.StringIO())
with quiet:
    df_all, df_contrib, df_coords, _, s2v = load_and_preprocess_data(
        f'{a.data}/mahanadi_discharge_data.csv',
        f'{a.data}/mahanadi_station_coordinates.csv',
        f'{a.data}/mahanadi_connectivity_matrix.csv')
matrix_sites = set(df_contrib.station) | set(df_contrib.contributor) | {
    s for s in pd.read_csv(f'{a.data}/mahanadi_connectivity_matrix.csv', index_col=0)['Name of site']
    .map(lambda n: n.strip().lower().replace(' ', ''))}
df = add_temporal_features(df_all).loc['1980-01-01':'1990-12-31'].copy()
temporal = [c for c in df.columns if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
stations = sorted(c for c in df.columns if c not in temporal and c in matrix_sites)
if a.min_completeness is not None:
    def in_period_completeness(q):
        if not q.notna().any():
            return 0.0
        return 100 * q.loc[q.first_valid_index():q.last_valid_index()].notna().mean()
    stations = [s for s in stations if in_period_completeness(df[s]) >= a.min_completeness]
df = df[stations + temporal]
if a.seed_block:
    with quiet:
        s0, s1 = find_best_data_window(df, stations, '1980-01-01', '1990-12-31', 3 * 365)
    seed_blk = df.loc[s0:s1].copy()
else:
    seed_blk = df.iloc[:0].copy()
ev_orig = df.loc[~df.index.isin(seed_blk.index)].copy()
if a.pattern == 'fixed':
    with quiet:
        ev_gap = create_contiguous_segment_gaps_by_percent(ev_orig, stations, a.gap, 10.0, random_seed=a.seed)
else:  # copied from review_expts/common/data.py make_gaps(pattern='native')
    pool = []
    for s in stations:
        r = df[s].isna().astype(int)
        runs = r.groupby((r == 0).cumsum()).sum()
        pool += [int(v) for v in runs if v > 0]
    pool = np.array([L for L in pool if 2 <= L <= 365])
    rng = np.random.default_rng(a.seed)
    ev_gap, n = ev_orig.copy(), len(ev_orig)
    for s in stations:
        target, placed, tries = 0.10 * n, 0, 0
        col, used = ev_gap.columns.get_loc(s), np.zeros(n, bool)
        while placed < target and tries < 10000:
            tries += 1
            L = int(rng.choice(pool)); st = int(rng.integers(0, n - L))
            if used[st:st + L].any():
                continue
            used[st:st + L] = True
            ev_gap.iloc[st:st + L, col] = np.nan
            placed += L
full_gap = pd.concat([seed_blk, ev_gap]).sort_index()
has_data = [s for s in stations if df[s].notna().any()]

with quiet:
    D = build_distance_matrix(df_coords, stations).loc[stations, stations]
    C = build_connectivity_matrix(df_contrib, stations, s2v).loc[stations, stations]
receivers = [s for s in stations if C.loc[s].sum() > 0]

# Placebo map, built once from the real R's lags and weights.
ref = MissForestImputer(distance_matrix=D, connectivity=C)
for i in receivers:
    w = ref._calculate_weights(i, [c for c in stations if c != i])
    ups = [j for j in stations if j != i and C.loc[i, j] > 0]
    downs = [j for j in stations if C.loc[j, i] > 0]
    pool = [k for k in has_data if k != i and k not in ups and k not in downs]
    used, sub = set(), []
    for j in sorted(ups, key=lambda j: D.loc[i, j]):
        cands = sorted(pool, key=lambda k: (k in used, abs(D.loc[i, k] - D.loc[i, j])))
        k = cands[0]; used.add(k)
        sub.append((k, int(round(D.loc[i, j] / 100.0)), float(w[j])))
    PlaceboImputer.placebo[i] = sub


def pimf(cls, lags, n_features='nearest_observed'):
    return cls(distance_matrix=D, connectivity=C, max_iter=10, n_estimators=100,
               random_state=a.seed, distance_weighting_type='inverse',
               temporal_feature_columns=temporal, initialization_method='historical_mean',
               ordering_method='most_full_first', use_temporal_lags=lags,
               use_routing=True, celerity_km_per_day=100.0, n_features=n_features)


res, secs = {}, {}
for arm in a.arms.split(','):
    t0 = time.time()
    with quiet:
        if arm == 'vanilla':
            m = MissForestImputer(distance_matrix=None, connectivity=None, random_state=a.seed)
            m.fit(full_gap[stations]); res[arm] = m.transform(full_gap[stations])
        else:
            m = {'pimf': lambda: pimf(MissForestImputer, True),
                 'pimf_old': lambda: pimf(MissForestImputer, True, 'neighbour_day'),
                 'pimfL': lambda: pimf(MissForestImputer, False),
                 'pimfL_placebo': lambda: pimf(PlaceboImputer, False)}[arm]()
            m.fit(full_gap); res[arm] = m.transform(full_gap)
    secs[arm] = time.time() - t0

rows = []
for s in stations:
    msk = (ev_gap[s].isna() & ev_orig[s].notna()).to_numpy()
    if not msk.any():
        continue
    t = ev_orig.index[msk]
    m = full_gap[s].isna().astype(int)
    runlen = m.groupby((m == 0).cumsum()).transform('sum')  # length of the missing run each cell sits in
    blk = pd.DataFrame({'station': s, 'date': t, 'obs': ev_orig.loc[t, s].to_numpy(),
                        'run_len': runlen.loc[t].to_numpy()})
    for k, v in res.items():
        blk[k] = v.loc[t, s].to_numpy()
    rows.append(blk)
cells = pd.concat(rows, ignore_index=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump(dict(cells=cells, seed=a.seed, gap=a.gap, receivers=receivers, secs=secs,
                 placebo=PlaceboImputer.placebo, sd={s: float(df[s].std()) for s in stations}),
            open(a.out, 'wb'))
print(a.seed, a.gap, {k: round(v) for k, v in secs.items()}, len(cells), flush=True)
