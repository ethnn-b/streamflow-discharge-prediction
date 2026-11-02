"""Diagnostic for the long-gap loss. One (seed, gap) cell on Mahanadi under the
v6 Section 4.1 protocol, using the unchanged code on review/rerun-vs-vanilla.

Saves, per arm, the imputed gap cells plus the historical-mean init, the
distance of each gap cell to the nearest observed value, and the PIMF
feature importances. Arms:
  pimf        as in the paper
  pimfL       PIMF without N
  vanilla     vanilla_single
  pimf_warm   PIMF whose initial frame is vanilla's output (diagnostic probe only)
"""
import argparse, contextlib, io, json, os, pickle, sys, time, warnings

p = argparse.ArgumentParser()
p.add_argument('--root'); p.add_argument('--seed', type=int); p.add_argument('--gap', type=int)
p.add_argument('--out')
a = p.parse_args()
sys.path.insert(0, a.root); os.chdir(a.root)

import numpy as np
import pandas as pd
import src.imputers.missforest as mfmod
from src.imputers.baselines import initialize_for_missforest
from src.utils.data import (load_and_preprocess_data, add_temporal_features,
                            find_best_data_window, create_contiguous_segment_gaps_by_percent)
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix
from src.imputers.missforest import MissForestImputer

warnings.simplefilter('ignore')
quiet = contextlib.redirect_stdout(io.StringIO())
with quiet:
    df_all, df_contrib, df_coords, _, s2v = load_and_preprocess_data(
        'mahanadi_data/mahanadi_discharge_data.csv',
        'mahanadi_data/mahanadi_station_coordinates.csv',
        'mahanadi_data/mahanadi_connectivity_matrix.csv')
df = add_temporal_features(df_all).loc['1980-01-01':'1990-12-31'].copy()
temporal = [c for c in df.columns if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
stations = sorted(c for c in df.columns if c not in temporal)
D = build_distance_matrix(df_coords, stations).loc[stations, stations]
C = build_connectivity_matrix(df_contrib, stations, s2v).loc[stations, stations]

with quiet:
    s0, s1 = find_best_data_window(df, stations, '1980-01-01', '1990-12-31', 3 * 365)
seed_blk = df.loc[s0:s1].copy()
ev_orig = df.loc[~df.index.isin(seed_blk.index)].copy()
with quiet:
    ev_gap = create_contiguous_segment_gaps_by_percent(ev_orig, stations, a.gap, 10.0, random_seed=a.seed)
full_gap = pd.concat([seed_blk, ev_gap]).sort_index()

miss = full_gap[stations].isna()
run = lambda g: g.astype(int).groupby((g.astype(int) == 0).cumsum()).cumsum()
gprev = miss.apply(run)
gnext = miss.iloc[::-1].apply(run).iloc[::-1]
hist_init = initialize_for_missforest(full_gap, stations, 'historical_mean')

def pimf(lags, init_frame=None):
    m = MissForestImputer(distance_matrix=D, connectivity=C, max_iter=10, n_estimators=100,
                          random_state=a.seed, distance_weighting_type='inverse',
                          temporal_feature_columns=temporal, initialization_method='historical_mean',
                          ordering_method='most_full_first', use_temporal_lags=lags, n_features='neighbour_day',
                          use_routing=True, celerity_km_per_day=100.0)
    orig = mfmod.initialize_for_missforest
    if init_frame is not None:
        mfmod.initialize_for_missforest = lambda X, cols, meth, **k: X.fillna(init_frame)
    try:
        m.fit(full_gap); out = m.transform(full_gap)
    finally:
        mfmod.initialize_for_missforest = orig
    return m, out

res = {}
t0 = time.time()
with quiet:
    mv = MissForestImputer(distance_matrix=None, connectivity=None, random_state=a.seed)
    mv.fit(full_gap[stations]); res['vanilla'] = mv.transform(full_gap[stations])
    m_p, res['pimf'] = pimf(True)
    _, res['pimfL'] = pimf(False)
    _, res['pimf_warm'] = pimf(True, init_frame=res['vanilla'][stations])
imp = {s: dict(zip(m.feature_names_in_, m.feature_importances_))
       for s, m in m_p.models.items() if m is not None}

rows = []
for s in stations:
    msk = ev_gap[s].isna() & ev_orig[s].notna()
    for t in ev_orig.index[msk]:
        r = dict(station=s, date=t, obs=ev_orig.at[t, s], init=hist_init.at[t, s],
                 gprev=int(gprev.at[t, s]), gnext=int(gnext.at[t, s]))
        for k, v in res.items():
            r[k] = v.at[t, s]
        rows.append(r)
cells = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump(dict(cells=cells, importances=imp, seed=a.seed, gap=a.gap,
                 sd={s: float(df[s].std()) for s in stations}), open(a.out, 'wb'))
print(a.seed, a.gap, f'{time.time() - t0:.0f}s', len(cells))
