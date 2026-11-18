"""Section 3.3.4: column weighting is inert for a forest.

One 100-tree forest per Mahanadi gauge (12 x 100 = 1,200 trees) fitted on the same
rows with and without each predictor column scaled by its blended weight w_ij
(alpha=0.5, p=1). Weights are rounded to powers of two, which rescale a float
exactly, so the trees must be identical: same node count and same split variable at
every node, predictions equal to rounding. With the raw weights float32 storage in
scikit-learn can move a value across a threshold, so those are reported separately.
Usage: PYTHONPATH=. python rerun_expts/check_invariance.py
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rerun_expts.common import load_network

df, st, _t, D, C, _r = load_network('mahanadi')
n_trees = same_shape = same_split = 0
worst = {'pow2': 0.0, 'raw': 0.0}
for i in st:
    oth = [j for j in st if j != i]
    wd = np.array([1 / (D.loc[i, j] + 1e-9) for j in oth]); wd /= wd.sum()
    wc = np.array([C.loc[i, j] for j in oth], float); wc = wc / wc.sum() if wc.sum() else wc
    w = 0.5 * wd + 0.5 * wc; w /= w.sum()
    X = df[oth].fillna(df[oth].mean())
    obs = df[i].notna().values
    X, y = X.values[obs], df[i].values[obs]
    base = RandomForestRegressor(100, max_features='sqrt', random_state=0).fit(X, y)
    for tag, ww in (('pow2', 2.0 ** np.round(np.log2(w))), ('raw', w)):
        f = RandomForestRegressor(100, max_features='sqrt', random_state=0).fit(X * ww, y)
        worst[tag] = max(worst[tag], np.abs(f.predict(X * ww) - base.predict(X)).max())
        if tag == 'pow2':
            for a, b in zip(base.estimators_, f.estimators_):
                n_trees += 1
                same_shape += a.tree_.node_count == b.tree_.node_count
                same_split += (a.tree_.node_count == b.tree_.node_count
                               and np.array_equal(a.tree_.feature, b.tree_.feature))
print(f'trees {n_trees}; same node count {same_shape}; same split variable at every node {same_split}')
print(f"max |prediction difference|: powers of two {worst['pow2']:.2e}, raw weights {worst['raw']:.2e} (cumec)")

# ---- Part 2: the full imputer. The `wcol` arm (run_fold.py) is the ablation base with every
# predictor column scaled by its weight; res_invariance/ holds its 20 transductive cells
#   python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms wcol --out res_invariance
# and res_main/ holds the unscaled base. Raw weights, so float32 rounding can move a value
# across a split threshold; the question is whether that has any systematic effect.
import glob, json, os
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
scale = {s: float(df[s].mean()) for s in st}


def nrmse(path):
    ps = json.load(open(path))['per_station']
    return np.mean([100 * v['RMSE'] / scale[s] for s, v in ps.items() if v and s in scale])


per_gap, allc = {}, []
for f in sorted(glob.glob(os.path.join(HERE, 'res_invariance', 'mahanadi_wcol_*.json'))):
    r = json.load(open(f))
    ref = os.path.join(HERE, 'res_main', f"mahanadi_base_f0_s{r['seed']}_{r['gap']}.json")
    d = 100 * (nrmse(f) - nrmse(ref)) / nrmse(ref)
    per_gap.setdefault(r['gap'], []).append(d); allc.append(d)
if allc:
    print('\nfull imputer, weighted vs unweighted columns, change in nRMSE (%):')
    for g_, v in sorted(per_gap.items()):
        print(f'  gap {g_:>3} d: mean {np.mean(v):+.2f}  range [{min(v):+.2f}, {max(v):+.2f}]  n={len(v)}')
    t, p = stats.ttest_1samp([np.mean(v) for v in per_gap.values()], 0)
    print(f'  all {len(allc)} cells: mean {np.mean(allc):+.2f}, {sum(x < 0 for x in allc)} negative, '
          f'range [{min(allc):+.2f}, {max(allc):+.2f}], blocked t-test p={p:.3f}')
