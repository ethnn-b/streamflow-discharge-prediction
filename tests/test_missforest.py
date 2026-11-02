"""Self-check for the _mean_for fix: a column with no observations in the fitted
frame must come back imputed, not NaN."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter('ignore')
import numpy as np, pandas as pd
from src.imputers.missforest import MissForestImputer

idx = pd.date_range('2000-01-01', periods=200)
rng = np.random.default_rng(0)
seed_blk = pd.DataFrame({'a': rng.normal(10, 2, 200), 'b': rng.normal(5, 1, 200),
                         'dead': np.nan}, index=idx)
ev = pd.DataFrame({'a': rng.normal(10, 2, 200), 'b': rng.normal(5, 1, 200),
                   'dead': rng.normal(7, 1, 200)}, index=idx + pd.Timedelta(days=200))
ev_gap = ev.copy()
ev_gap.iloc[50:70, ev_gap.columns.get_loc('dead')] = np.nan
ev_gap.iloc[50:70, ev_gap.columns.get_loc('a')] = np.nan

m = MissForestImputer(distance_matrix=None, connectivity=None)
m.fit(seed_blk)
out = m.transform(ev_gap)

assert np.isnan(m.col_means['dead']), 'precondition: dead column has a NaN mean'
cut = ev_gap['dead'].isna() & ev['dead'].notna()
assert cut.sum() == 20, cut.sum()
got = out.loc[cut, 'dead']
assert got.notna().all(), f'{int(got.isna().sum())}/20 cells left NaN by transform()'
assert out.loc[ev_gap['a'].isna(), 'a'].notna().all(), 'normal column regressed'
# untrainable column must fall back to the transform frame's own observed mean,
# not to a bare 0 - ev['dead'] is drawn around 7
frame_mean = ev_gap['dead'].mean()
assert abs(got.iloc[0] - frame_mean) < 1e-9, f'got {got.iloc[0]}, frame mean {frame_mean}'
assert got.iloc[0] > 5, 'fell back to 0 instead of the frame mean'
print('OK: untrainable column filled with the frame mean (%.2f), no NaN escapes transform()'
      % got.iloc[0])

# and 0 only when there is genuinely nothing to average
m2 = MissForestImputer(distance_matrix=None, connectivity=None)
m2.fit(seed_blk)
empty = ev_gap.copy(); empty['dead'] = np.nan
assert m2._mean_for('dead', empty) == 0.0, 'no data anywhere must give 0'


# --- dead columns must not take weight mass in the Eq. (11) aggregate ---
D = pd.DataFrame([[0., 10., 20.], [10., 0., 15.], [20., 15., 0.]],
                 index=['a', 'b', 'dead'], columns=['a', 'b', 'dead'])
C0 = pd.DataFrame(0.0, index=D.index, columns=D.columns)

gapped = pd.DataFrame({'a': rng.normal(10, 2, 200), 'b': rng.normal(5, 1, 200),
                       'dead': np.nan}, index=idx)
gapped.iloc[20:40, gapped.columns.get_loc('a')] = np.nan

pm = MissForestImputer(distance_matrix=D, connectivity=C0, max_iter=1, n_estimators=5)
pm.fit(gapped)
assert pm._dead_columns == frozenset({'dead'}), pm._dead_columns
w = pm._calculate_weights('a', ['b', 'dead'])
assert set(w.index) == {'b', 'dead'}, 'weights must stay aligned to the caller list'
assert w['dead'] == 0.0, f"dead column took weight {w['dead']}"
assert abs(w['b'] - 1.0) < 1e-9, f'live weights must renormalize to 1, got {w["b"]}'

# with every predictor live, weights are unchanged by the fix
pm2 = MissForestImputer(distance_matrix=D, connectivity=C0, max_iter=1, n_estimators=5)
full = gapped.copy(); full['dead'] = rng.normal(3, 1, 200)
pm2.fit(full)
w2 = pm2._calculate_weights('a', ['b', 'dead'])
assert pm2._dead_columns == frozenset(), pm2._dead_columns
assert abs(w2.sum() - 1.0) < 1e-9 and w2['dead'] > 0, w2.to_dict()
print('OK: dead columns carry zero weight; live weights renormalize to 1')


# --- the {A,R,L} factorial must produce 8 distinct predictor sets ---
Dm = pd.DataFrame([[0., 10., 20.], [10., 0., 15.], [20., 15., 0.]],
                  index=['a', 'b', 'c'], columns=['a', 'b', 'c'])
Cm = pd.DataFrame(0.0, index=Dm.index, columns=Dm.columns)
Cm.loc['a', 'b'] = 1.0                      # b is upstream of a -> a is a receiver
frame = pd.DataFrame({'a': rng.normal(10, 2, 300), 'b': rng.normal(5, 1, 300),
                      'c': rng.normal(8, 2, 300)},
                     index=pd.date_range('2000-01-01', periods=300))
frame['day_of_year_sin'] = np.sin(2 * np.pi * frame.index.dayofyear / 366.0)
frame.iloc[40:60, 0] = np.nan

cols = {}
for A in (False, True):
    for R in (False, True):
        for L in (False, True):
            im = MissForestImputer(distance_matrix=Dm, connectivity=Cm, max_iter=1,
                                   n_estimators=5, temporal_feature_columns=['day_of_year_sin'],
                                   use_aggregate=A, use_routing=R, use_temporal_lags=L)
            im._set_missing(frame[['a', 'b', 'c']].isna())
            cols[(A, R, L)] = tuple(im._build_predictors(frame.fillna(0), 'a', ['b', 'c']).columns)

assert len(set(cols.values())) == 8, f'only {len(set(cols.values()))} distinct arms: {cols}'
assert 'weighted_upstream_discharge' in cols[(True, False, False)]
assert 'routed_upstream_discharge' in cols[(False, True, False)]
assert 'weighted_upstream_discharge' not in cols[(False, True, False)], 'R must not drag A in'
assert 'lag_prev' in cols[(False, False, True)]
assert cols[(False, False, False)] == ('b', 'c', 'day_of_year_sin'), cols[(False, False, False)]
print('OK: 8 distinct {A,R,L} arms; R is independent of A')
