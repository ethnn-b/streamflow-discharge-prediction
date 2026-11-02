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
GAPS = [3, 7, 30, 100]
FACTORIAL = ['base', 'A', 'R', 'L', 'AR', 'AL', 'RL', 'ARL']
NET = sys.argv[1] if len(sys.argv) > 1 else 'mahanadi'

R = {}
for f in glob.glob(os.path.join(HERE, 'res_ablation', '*.json')):
    r = json.load(open(f))
    R[(r['net'], r['arm'], r['fold'], r['seed'], r['gap'])] = r
CELLS = sorted({(k[2], k[3]) for k in R if k[0] == NET})          # (fold, seed)


def get(arm, fold, seed, gap):
    return R.get((NET, arm, fold, seed, gap))


def mean_table(arms, scope):
    print(f'\n== {NET} {scope}: mean over {len(CELLS)} fold x seed cells '
          f'(RMSE MAE NSE KGE PBIAS) [n]')
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
             'L': [('L', 'base'), ('AL', 'A'), ('RL', 'R'), ('ARL', 'AR')]}[feature]
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


arms_present = [a for a in ['histmean', 'linear', 'kalman', 'knn', 'vanilla'] + FACTORIAL
                if any(k[1] == a for k in R if k[0] == NET)]
scopes = ['pooled'] + (['recv'] if any(
    (r.get('recv') for k, r in R.items() if k[0] == NET and k[1] == 'base')) else [])

for scope in scopes:
    mean_table(arms_present, scope)

for scope in scopes:
    print(f'\n== {NET} {scope}: main effect of each feature (averaged over the other two)')
    for feat in ('A', 'R', 'L'):
        if any(k[1] == feat for k in R if k[0] == NET):
            show(f'  {feat} vs no-{feat}', marginal(feat, scope))

    print(f'\n== {NET} {scope}: each arm against the ablation base')
    for arm in arms_present:
        if arm != 'base':
            show(f'  {arm:8s} vs base', paired(arm, 'base', scope))

    print(f'\n== {NET} {scope}: reference comparisons')
    for a, b in (('ARL', 'vanilla'), ('ARL', 'linear'), ('base', 'vanilla'), ('AR', 'ARL')):
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
