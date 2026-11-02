"""One (arm, fold, gap, seed) cell of the feature ablation.

Protocol
--------
The record is split into `--n-folds` contiguous blocks. One block is the target
imputation period: contiguous gaps of `--gap` days are cut in it to a 10% target,
and it is the only place metrics are computed. The model is fitted on the rows
OUTSIDE that block, so no row containing a withheld cell is ever a training row.

transform() is then run on the whole record, with gaps present only in the target
block. That gives the block's first and last rows their real temporal neighbours
for the lag and routing features instead of a bfill against the block edge; it
leaks nothing, because the withheld cells are NaN in that frame too and the fitted
models are not updated by transform().

Arms
----
vanilla   conventional MissForest: discharge columns only, column-mean init, no
          ordering, no derived features. Reference point, outside the factorial.
base      MissForest with the seasonal columns, historical-mean initialization and
          most-full-first ordering, and none of A/R/L.
A R L     base plus any combination of
            A = weighted basin discharge, Eq. (11)
            R = travel-time routed upstream discharge, Eq. (13)
            L = temporal neighbour and gap-distance features, Eqs. (14)-(15)
          so ARL is PIMF and AR is PIMF-L. Everything other than the named
          features is identical across all eight, which is what isolates them.
"""
import argparse, json, os, sys, time, warnings, contextlib, io

ROOT = sys.argv[sys.argv.index('--root') + 1]
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from rerun_expts.common import load_network, folds
from src.utils.data import create_contiguous_segment_gaps_by_percent
from src.utils.metrics import evaluate_metrics
from src.imputers.missforest import MissForestImputer

p = argparse.ArgumentParser()
p.add_argument('--root'); p.add_argument('--net', default='mahanadi')
p.add_argument('--arm'); p.add_argument('--fold', type=int); p.add_argument('--seed', type=int)
p.add_argument('--gap', type=int); p.add_argument('--n-folds', type=int, default=3)
p.add_argument('--celerity', type=float, default=100.0); p.add_argument('--out')
# n_jobs for the forests. 1 thread per cell with many cells in parallel is far
# better throughput than n_jobs=-1 per cell: 100 small trees scale to only ~3x
# on 24 cores. A fixed random_state makes the forest deterministic in n_jobs.
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()

warnings.simplefilter('ignore')
quiet = contextlib.redirect_stdout(io.StringIO())

df, stations, temporal, D, C, receivers = load_network(a.net)
blocks = folds(df.index, a.n_folds)
test_idx = blocks[a.fold]
train_idx = df.index.difference(test_idx)

test_orig = df.loc[test_idx].copy()
with quiet:
    test_gap = create_contiguous_segment_gaps_by_percent(
        test_orig, stations, a.gap, 10.0, random_seed=a.seed)
full_gap = pd.concat([df.loc[train_idx], test_gap]).sort_index()

FACTORIAL = {'base': (0, 0, 0), 'A': (1, 0, 0), 'R': (0, 1, 0), 'L': (0, 0, 1),
             'AR': (1, 1, 0), 'AL': (1, 0, 1), 'RL': (0, 1, 1), 'ARL': (1, 1, 1)}

t0 = time.time()
with quiet:
    if a.arm == 'vanilla':
        m = MissForestImputer(distance_matrix=None, connectivity=None, random_state=a.seed,
                              n_jobs=a.threads)
        m.fit(full_gap.loc[train_idx, stations])
        imp = m.transform(full_gap[stations]).loc[test_idx]
    elif a.arm == 'linear':
        from src.imputers.baselines import linear_interpolation_imputation
        imp = linear_interpolation_imputation(full_gap, stations).loc[test_idx]
    elif a.arm == 'histmean':
        from src.imputers.baselines import historical_mean_imputation
        # day-of-year means are taken from the training rows only
        imp = historical_mean_imputation(test_gap, stations,
                                         training_data=full_gap.loc[train_idx])
    elif a.arm == 'kalman':
        from src.imputers.kalman import kalman_imputation
        imp = kalman_imputation(full_gap.loc[train_idx], test_gap, stations)
    elif a.arm == 'knn':
        from src.imputers.knn import knn_imputation
        imp = knn_imputation(full_gap.loc[train_idx], test_gap, list(full_gap.columns), k=5)
    elif a.arm in FACTORIAL:
        useA, useR, useL = FACTORIAL[a.arm]
        m = MissForestImputer(distance_matrix=D, connectivity=C, max_iter=10, n_estimators=100,
                              random_state=a.seed, distance_weighting_type='inverse',
                              temporal_feature_columns=temporal,
                              initialization_method='historical_mean',
                              ordering_method='most_full_first',
                              use_aggregate=bool(useA), use_routing=bool(useR),
                              use_temporal_lags=bool(useL), celerity_km_per_day=a.celerity,
                              n_jobs=a.threads)
        m.fit(full_gap.loc[train_idx])
        imp = m.transform(full_gap).loc[test_idx]
    else:
        raise SystemExit(f'unknown arm {a.arm}')
el = time.time() - t0


def score(cols):
    yt, yp = [], []
    for s in cols:
        msk = test_gap[s].isna() & test_orig[s].notna()
        if msk.sum():
            yt.append(test_orig.loc[msk, s].values); yp.append(imp.loc[msk, s].values)
    if not yt:
        return None
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    n_bad = int(np.isnan(yp).sum())
    assert n_bad == 0, f'{n_bad}/{len(yp)} predictions are NaN; the arm left cells unimputed'
    r = evaluate_metrics(yt, yp)
    r['PBIAS'] = float(100 * (yp - yt).sum() / yt.sum())
    r['n'] = int(len(yt))
    return {k: float(v) for k, v in r.items()}


cut = (test_gap[stations].isna() & test_orig[stations].notna())
res = dict(net=a.net, arm=a.arm, fold=a.fold, n_folds=a.n_folds, seed=a.seed, gap=a.gap,
           celerity=a.celerity, seconds=el,
           test_block=[str(test_idx[0].date()), str(test_idx[-1].date())],
           n_train_rows=len(train_idx), n_test_rows=len(test_idx),
           achieved_missing_pct=float(100 * cut.sum().sum() / test_orig[stations].notna().sum().sum()),
           receivers=receivers, pooled=score(stations), recv=score(receivers),
           per_station={s: score([s]) for s in stations})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print(f"{a.arm:7s} f{a.fold} s{a.seed} {a.gap:>3}d {el:4.0f}s  pooled {res['pooled']['RMSE']:8.2f}"
      f"  recv {res['recv']['RMSE'] if res['recv'] else float('nan'):8.2f}"
      f"  miss {res['achieved_missing_pct']:.2f}%  test {res['test_block'][0]}..{res['test_block'][1]}")
