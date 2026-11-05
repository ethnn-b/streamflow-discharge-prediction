"""Self-check for the two corrected feature blocks.

anchor mode: the target's temporal features must read only observed cells (never the
imputer's own output, never a withheld one), their distances must track real gap
geometry at prediction time, and must vary across training rows -- the naive block's
distance features are constant 0 on every training row and so carry zero importance.

placebo mode: the fake upstream set must match the real one on count, total weight and
travel-time lags while sharing none of its members.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter('ignore')
import numpy as np, pandas as pd
from src.imputers.missforest import MissForestImputer

idx = pd.date_range('2000-01-01', periods=800)
rng = np.random.default_rng(0)
st = [f's{i}' for i in range(6)]
df = pd.DataFrame({s: rng.normal(10 + i, 1, 800) for i, s in enumerate(st)}, index=idx)

D = pd.DataFrame(0.0, index=st, columns=st)
for i, a in enumerate(st):
    for j, b in enumerate(st):
        D.loc[a, b] = abs(i - j) * 60.0            # 60 km apart per step
C = pd.DataFrame(0.0, index=st, columns=st)
C.loc['s0', 's1'] = C.loc['s0', 's2'] = 1.0        # only s1, s2 are upstream of s0

gapped = df.copy()
gapped.iloc[100:120, gapped.columns.get_loc('s0')] = np.nan   # a known 20-day gap
mask = gapped.isna()

# The live iterate the imputer would hold: gap cells filled with its own output.
SENTINEL = 1e9
live = gapped.copy()
live.iloc[100:120, live.columns.get_loc('s0')] = SENTINEL

m = MissForestImputer(distance_matrix=D, connectivity=C, temporal_feature_columns=[],
                      lag_mode='anchor', routing_mode='real', celerity_km_per_day=100.0)
m.discharge_columns = st
m._set_missing(mask)

# --- 1. anchors never read an unobserved cell, and the naive block does ---
a = m._anchor_features(live, 's0', synth_gaps=False)
assert not (a['anchor_prev'] == SENTINEL).any(), 'anchor_prev read the imputer own output'
assert not (a['anchor_next'] == SENTINEL).any(), 'anchor_next read the imputer own output'
naive = m._lag_features(live, 's0')
assert (naive['lag_prev'] == SENTINEL).any(), 'precondition: the naive lag does read it'

# --- 2. real gap geometry: distances count outward from the gap edges ---
dp = a['anchor_dist_prev'].values[100:120]
dn = a['anchor_dist_next'].values[100:120]
assert list(dp) == list(range(1, 21)), dp
assert list(dn) == list(range(20, 0, -1)), dn
assert a['anchor_prev'].values[100] == live['s0'].values[99]
assert a['anchor_next'].values[119] == live['s0'].values[120]

# --- 3. training rows: naive distances are constant, synthetic anchor ones are not ---
train = mask.index[~mask['s0']]
assert m._missing_run_length(mask['s0']).loc[train].nunique() == 1, 'naive dist varies'
syn = m._anchor_features(live, 's0', synth_gaps=True)
dsp, dsn = syn['anchor_dist_prev'].loc[train], syn['anchor_dist_next'].loc[train]
assert dsp.nunique() > 5, sorted(dsp.unique())
assert dsp.max() >= 89, dsp.max()          # the long-gap regime must be in the training support
assert not (syn['anchor_prev'] == SENTINEL).any(), 'synthetic anchor read unobserved data'

# Each training row carries the geometry of one real gap: the two distances sum to a
# ladder length + 1, so both sides are small together for a short pretend gap. Checked
# on s1, which has no missing cell, so a distance is exactly the offset asked for; on
# a gappy station the search skips past missing rows and the distance can only grow.
L = max(MissForestImputer.ANCHOR_TRAIN_GAPS)
s1 = m._anchor_features(live, 's1', synth_gaps=True)
mid = slice(L, len(idx) - L)               # rows with both sides inside the frame
pair = (s1['anchor_dist_prev'].values[mid] + s1['anchor_dist_next'].values[mid] - 1)
assert set(pair) <= set(MissForestImputer.ANCHOR_TRAIN_GAPS), sorted(set(pair))
assert (s1['anchor_dist_prev'].values[mid] < 3).any(), 'no short pretend gaps drawn'

# --- 4. placebo matches the real set on count, weight and lag, and shares no member ---
w = m._calculate_weights('s0', [c for c in st if c != 's0'])
up = ['s1', 's2']
lag = {'s1': int(round(D.loc['s0', 's1'] / 100)), 's2': int(round(D.loc['s0', 's2'] / 100))}
fake, fw, flag = m._placebo_set('s0', w, up, lag)
assert len(fake) == len(up), fake
assert not (set(fake) & set(up)), fake
assert abs(fw[fake].sum() - w[up].sum()) < 1e-12, (fw[fake].sum(), w[up].sum())
assert sorted(flag.values()) == sorted(lag.values()), (flag, lag)

# --- 5. the default IS the method: PIMF = {A, R, N}, no bands ---
d = MissForestImputer(distance_matrix=D, connectivity=C, temporal_feature_columns=[])
assert d.lag_mode == 'anchor' and d.routing_mode == 'real'
assert d.use_aggregate and d.use_routing and d.use_temporal_lags
d.discharge_columns = st; d._set_missing(mask)
cols = d._build_predictors(live, 's0', [c for c in st if c != 's0']).columns
assert 'anchor_prev' in cols and 'lag_prev' not in cols, list(cols)

# the superseded naive block stays reachable, byte for byte
n = MissForestImputer(distance_matrix=D, connectivity=C, temporal_feature_columns=[],
                      lag_mode='naive')
n.discharge_columns = st; n._set_missing(mask)
ncols = n._build_predictors(live, 's0', [c for c in st if c != 's0']).columns
assert 'lag_prev' in ncols and 'anchor_prev' not in ncols, list(ncols)

print('anchor/placebo self-check OK')
