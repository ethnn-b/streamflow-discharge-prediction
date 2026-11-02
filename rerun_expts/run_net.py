"""One (arm, seed, gap, celerity) cell under the v5 Section 4.1 protocol.

Arms
  pimf            all features (A, R, N, S), single fit over the whole gapped record
  pimfL           PIMF without the neighbour-day block (use_temporal_lags=False)
  vanilla_single  MissForest on discharge columns only, column-mean init, no ordering,
                  single fit over the whole gapped record (same schedule as PIMF)
  vanilla_seedfit as committed in benchmark_1980_1990.py: fit on the 3-year seed
                  block only, then transform the gapped evaluation period

Writes one JSON with pooled / receiver / per-station errors on the gap cells.
"""
import argparse, json, os, sys, time, warnings, contextlib, io

ROOT = sys.argv[sys.argv.index('--root') + 1]
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from src.utils.data import (load_and_preprocess_data, add_temporal_features,
                            find_best_data_window, create_contiguous_segment_gaps_by_percent)
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix
from src.utils.metrics import evaluate_metrics
from src.imputers.missforest import MissForestImputer

p = argparse.ArgumentParser()
p.add_argument('--root'); p.add_argument('--arm'); p.add_argument('--seed', type=int)
p.add_argument('--gap', type=int); p.add_argument('--celerity', type=float, default=100.0)
p.add_argument('--out'); p.add_argument('--net', default='mahanadi')
a = p.parse_args()

warnings.simplefilter('ignore')
quiet = contextlib.redirect_stdout(io.StringIO())
NETS = {
    'mahanadi': ('mahanadi_data/mahanadi_discharge_data.csv', 'mahanadi_data/mahanadi_station_coordinates.csv',
                 'mahanadi_data/mahanadi_connectivity_matrix.csv', '1980-01-01', '1990-12-31'),
    'kaveri': ('kaveri_data/cauv_discharge.csv', 'kaveri_data/lat_long_cauv.csv', None, '1980-01-01', '1990-12-31'),
    'us': ('us_data/us_top10_discharge_data_demo.csv', 'us_data/us_top10_station_coordinates.csv',
           'us_data/us_top10_connectivity_matrix.csv', '2013-01-01', '2024-12-31'),
}
dpath, cpath, kpath, T0, T1 = NETS[a.net]
with quiet:
    df_all, df_contrib, df_coords, _, s2v = load_and_preprocess_data(dpath, cpath, kpath)
df = add_temporal_features(df_all).loc[T0:T1].copy()
_st = [c for c in df.columns if not c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
df[_st] = df[_st].mask(df[_st] <= -999999)  # USGS no-data sentinel
temporal = [c for c in df.columns if c.startswith(('day_of_year_', 'month_', 'week_of_year_'))]
stations = sorted(c for c in df.columns if c not in temporal)
D = build_distance_matrix(df_coords, stations).loc[stations, stations]
C = build_connectivity_matrix(df_contrib, stations, s2v).loc[stations, stations]
receivers = [s for s in stations if C.loc[s].sum() > 0]

win = 3 * 365 + 3 // 4
with quiet:
    s0, s1 = find_best_data_window(df, stations, T0, T1, win)
seed_blk = df.loc[s0:s1].copy()
ev_orig = df.loc[~df.index.isin(seed_blk.index)].copy()
with quiet:
    ev_gap = create_contiguous_segment_gaps_by_percent(ev_orig, stations, a.gap, 10.0, random_seed=a.seed)
full_gap = pd.concat([seed_blk, ev_gap]).sort_index()

t0 = time.time()
with quiet:
    if a.arm in ('pimf', 'pimfL'):
        m = MissForestImputer(distance_matrix=D, connectivity=C, max_iter=10, n_estimators=100,
                              random_state=a.seed, distance_weighting_type='inverse',
                              temporal_feature_columns=temporal, initialization_method='historical_mean',
                              ordering_method='most_full_first', use_temporal_lags=(a.arm == 'pimf'),
                              use_routing=True, celerity_km_per_day=a.celerity)
        m.fit(full_gap); imp = m.transform(full_gap).loc[ev_orig.index]
    elif a.arm == 'vanilla_single':
        m = MissForestImputer(distance_matrix=None, connectivity=None, random_state=a.seed)
        m.fit(full_gap[stations]); imp = m.transform(full_gap[stations]).loc[ev_orig.index]
    elif a.arm == 'linear':
        from src.imputers.baselines import linear_interpolation_imputation
        imp = linear_interpolation_imputation(ev_gap, stations)
    elif a.arm == 'vanilla_seedfit':
        m = MissForestImputer(distance_matrix=None, connectivity=None)
        m.fit(seed_blk); imp = m.transform(ev_gap)
    else:
        raise SystemExit('unknown arm')
el = time.time() - t0

def score(cols):
    yt, yp = [], []
    for s in cols:
        msk = ev_gap[s].isna() & ev_orig[s].notna()
        if msk.sum():
            yt.append(ev_orig.loc[msk, s].values); yp.append(imp.loc[msk, s].values)
    if not yt:
        return None
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    r = evaluate_metrics(yt, yp)
    r['PBIAS'] = float(100 * (yp - yt).sum() / yt.sum())
    r['n'] = int(len(yt))
    return {k: float(v) for k, v in r.items()}

res = dict(net=a.net, arm=a.arm, seed=a.seed, gap=a.gap, celerity=a.celerity, seconds=el,
           seed_block=[str(s0.date()), str(s1.date())], receivers=receivers,
           pooled=score(stations), recv=score(receivers),
           per_station={s: score([s]) for s in stations})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print(a.arm, a.seed, a.gap, a.celerity, f"{el:.0f}s", 'pooled RMSE', round(res['pooled']['RMSE'], 2),
      'recv RMSE', round(res['recv']['RMSE'], 2) if res['recv'] else None, res['seed_block'], len(res['receivers']))
