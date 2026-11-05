"""Feature-ablation readout: what A, R and L each contribute.

Replicates are fold x seed (15 per arm per gap length). Gap length is a blocking
factor, not a replicate: the effects differ systematically with it, so pooled
tests are run over the per-gap-length means (df = 3) and the per-gap means are
reported alongside.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
GAPS = [3, 7, 30, 100]
FACTORIAL = ['base', 'A', 'R', 'L', 'AR', 'AL', 'RL', 'ARL']
ANCHOR = ['Lanc', 'ALanc', 'RLanc', 'ARLanc']
PLACEBO = ['Rplac', 'ARplac']
NET = sys.argv[1] if len(sys.argv) > 1 else 'mahanadi'
RESDIR = sys.argv[2] if len(sys.argv) > 2 else 'res_ablation'

R = {}
for f in glob.glob(os.path.join(HERE, RESDIR, '*.json')):
    r = json.load(open(f))
    R[(r['net'], r['arm'], r['fold'], r['seed'], r['gap'])] = r
CELLS = sorted({(k[2], k[3]) for k in R if k[0] == NET})          # (fold, seed)
PROTO = sorted({r.get('protocol', 'fold') for k, r in R.items() if k[0] == NET})

# Gauge scopes: every gauge weighted equally, each one's RMSE divided by its own mean
# discharge. Pooling the residuals of all gauges into one RMSE instead makes the
# metric scale-weighted, so it measures the largest gauge: at Mahanadi tikarapara is
# 61% of total network variance and carries 48-78% of the pooled squared error, so it
# decides the pooled result on its own. Deleting it does not fix that - basantpur then
# becomes 47% of what is left. Normalising and averaging over gauges is what keeps
# every gauge in the study without letting one of them be the study.
#
#   gauge_conn  PRIMARY. The gauges that take part in the connectivity graph, i.e. the
#               ones A and R can actually reach; see common.connected for the rule.
#   gauge       SECONDARY. Every gauge with data, including those outside the graph,
#               as the robustness check that the direction does not depend on the scope.
# Both are derived here from per_station, so the scope can change without re-running.
from rerun_expts.common import load_network, connected

SCALE, CONN = {}, None
try:
    _df, _st, _t, _D, _C, _recv = load_network(NET)
    SCALE = {s: float(_df[s].mean()) for s in _st if np.isfinite(_df[s].mean())}
    CONN = set(connected(_st, _C, _df))
except Exception as e:                                   # metrics still work without it
    print(f'note: per-gauge scales unavailable ({e}); the gauge scopes are skipped')


def _gauge(ps, keep=None):
    rows = [(s, v) for s, v in ps.items()
            if v and SCALE.get(s) and (keep is None or s in keep)]
    if not rows:
        return None
    # NSE and KGE are MEDIANS over gauges. Both are unbounded below, so a single gauge
    # with almost no variance inside its gaps drags a mean to nonsense: the 100-day
    # historical-mean arm averaged NSE = -22102 over the same gauges whose median is
    # near 0. RMSE and MAE are means of a bounded, scale-normalised quantity.
    return {'RMSE': float(np.mean([100 * v['RMSE'] / SCALE[s] for s, v in rows])),
            'MAE': float(np.mean([100 * v['MAE'] / SCALE[s] for s, v in rows])),
            'NSE': float(np.median([v['NSE'] for s, v in rows])),
            'KGE': float(np.median([v['KGE'] for s, v in rows])),
            'PBIAS': float(np.mean([v['PBIAS'] for s, v in rows])),
            'n': int(sum(v['n'] for s, v in rows)), 'n_gauges': len(rows)}


for _r in R.values():
    ps = _r.get('per_station') or {}
    _r['gauge'] = _gauge(ps)
    _r['gauge_conn'] = _gauge(ps, CONN) if CONN else None


def get(arm, fold, seed, gap):
    return R.get((NET, arm, fold, seed, gap))


def mean_table(arms, scope):
    print(f'\n== {NET} {scope}: mean over {len(CELLS)} fold x seed cells '
          f'(nRMSE% nMAE% medNSE medKGE PBIAS) [n]')
    for g in GAPS:
        for arm in arms:
            rs = [get(arm, f, s, g) for f, s in CELLS]
            rs = [r[scope] for r in rs if r and r.get(scope)]
            if not rs:
                continue
            m = {k: np.mean([r[k] for r in rs]) for k in ('RMSE', 'MAE', 'NSE', 'KGE', 'PBIAS')}
            print(f"{g:>3}d {arm:8s} {m['RMSE']:9.2f} {m['MAE']:8.2f} {m['NSE']:6.3f} "
                  f"{m['KGE']:6.3f} {m['PBIAS']:+6.1f}% [{len(rs)}]")


def paired(a, b, scope, gaps=GAPS):
    """Per-cell % RMSE change of a vs b, blocked over gap length."""
    per_gap, d = {}, []
    for g in gaps:
        for f, s in CELLS:
            ra, rb = get(a, f, s, g), get(b, f, s, g)
            if not (ra and rb and ra.get(scope) and rb.get(scope)):
                continue
            x = 100 * (ra[scope]['RMSE'] - rb[scope]['RMSE']) / rb[scope]['RMSE']
            d.append(x); per_gap.setdefault(g, []).append(x)
    if not d:
        return None
    gm = [np.mean(v) for v in per_gap.values()]
    t, p = stats.ttest_1samp(gm, 0) if len(gm) >= 2 else stats.ttest_1samp(d, 0)
    return dict(mean=np.mean(d), t=t, p=p, wins=sum(x < 0 for x in d), n=len(d),
                per_gap={g: (np.mean(v), sum(x < 0 for x in v), len(v)) for g, v in per_gap.items()})


def marginal(feature, scope):
    """Main effect of one feature: its mean paired contrast over the four
    subsets of the other two, which is what a factorial buys over a one-at-a-time
    ablation - it averages over the presence and absence of the others."""
    pairs = {'A': [('A', 'base'), ('AR', 'R'), ('AL', 'L'), ('ARL', 'RL')],
             'R': [('R', 'base'), ('AR', 'A'), ('RL', 'L'), ('ARL', 'AL')],
             'L': [('L', 'base'), ('AL', 'A'), ('RL', 'R'), ('ARL', 'AR')],
             # the same main effect for the corrected L block
             'Lanc': [('Lanc', 'base'), ('ALanc', 'A'), ('RLanc', 'R'), ('ARLanc', 'AR')],
             }[feature]
    d, per_gap = [], {}
    for on, off in pairs:
        for g in GAPS:
            for f, s in CELLS:
                ra, rb = get(on, f, s, g), get(off, f, s, g)
                if not (ra and rb and ra.get(scope) and rb.get(scope)):
                    continue
                x = 100 * (ra[scope]['RMSE'] - rb[scope]['RMSE']) / rb[scope]['RMSE']
                d.append(x); per_gap.setdefault(g, []).append(x)
    if not d:
        return None
    gm = [np.mean(v) for v in per_gap.values()]
    t, p = stats.ttest_1samp(gm, 0)
    return dict(mean=np.mean(d), t=t, p=p, wins=sum(x < 0 for x in d), n=len(d),
                per_gap={g: (np.mean(v), sum(x < 0 for x in v), len(v)) for g, v in per_gap.items()})


def show(label, r):
    if r is None:
        print(f'{label}: n/a'); return
    pg = '  '.join(f'{g}d {v[0]:+.1f}%' for g, v in sorted(r['per_gap'].items()))
    print(f"{label}: {r['mean']:+6.2f}%  better in {r['wins']}/{r['n']}  "
          f"[blocked t={r['t']:+.2f} p={r['p']:.4f}] | {pg}")


arms_present = [a for a in ['histmean', 'linear', 'kalman', 'knn', 'vanilla']
                + FACTORIAL + ANCHOR + PLACEBO
                if any(k[1] == a for k in R if k[0] == NET)]
scopes = [sc for sc in ('gauge_conn',)
          if any(r.get(sc) for k, r in R.items() if k[0] == NET)] + ['pooled']
if any(r.get('recv') for k, r in R.items() if k[0] == NET and k[1] == 'base'):
    scopes.append('recv')
print(f"== {NET}: protocol {'+'.join(PROTO)}, {len(CELLS)} replicate cells, "
      f"{len(scopes)} scopes {scopes}"
      + (f", {len(CONN)} connected gauges" if CONN else ''))

for scope in scopes:
    mean_table(arms_present, scope)

for scope in scopes:
    print(f'\n== {NET} {scope}: main effect of each feature (averaged over the other two)')
    for feat in ('A', 'R', 'L', 'Lanc'):
        if any(k[1] == feat for k in R if k[0] == NET):
            show(f'  {feat} vs no-{feat}', marginal(feat, scope))

    print(f'\n== {NET} {scope}: each arm against the ablation base')
    for arm in arms_present:
        if arm != 'base':
            show(f'  {arm:8s} vs base', paired(arm, 'base', scope))

    print(f'\n== {NET} {scope}: reference comparisons')
    # AL is the best arm on a network with no connectivity matrix, where R is undefined
    for a, b in (('ARL', 'vanilla'), ('ARL', 'linear'), ('base', 'vanilla'), ('AR', 'ARL'),
                 ('AL', 'vanilla'), ('AL', 'base'), ('AL', 'linear'),
                 # corrected L block against the published one and against dropping it
                 ('ARLanc', 'ARL'), ('ARLanc', 'AR'), ('ARLanc', 'vanilla'),
                 ('ALanc', 'AL'), ('ALanc', 'vanilla'), ('Lanc', 'L'),
                 # the connectivity control: real routing against matched non-upstream
                 ('Rplac', 'R'), ('ARplac', 'AR'), ('Rplac', 'base'), ('ARplac', 'A')):
        if all(any(k[1] == x for k in R if k[0] == NET) for x in (a, b)):
            show(f'  {a:8s} vs {b:8s}', paired(a, b, scope))

miss = [r['achieved_missing_pct'] for k, r in R.items() if k[0] == NET]
if miss:
    by_gap = {}
    for k, r in R.items():
        if k[0] == NET:
            by_gap.setdefault(k[4], []).append(r['achieved_missing_pct'])
    print('\nachieved missingness in the target block: '
          + ', '.join(f'{g}d {np.mean(v):.2f}%' for g, v in sorted(by_gap.items())))
