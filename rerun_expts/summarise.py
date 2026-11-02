import glob, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, GAPS, CS = [42, 1, 2], [3, 7, 30, 100], [25, 50, 75, 100, 150, 200, 350]


def load(pattern):
    out = {}
    for f in glob.glob(os.path.join(HERE, pattern)):
        r = json.load(open(f))
        out[(r.get('net', 'mahanadi'), r['arm'], r['seed'], r['gap'], int(r['celerity']))] = r
    return out


R, _seen = {}, {}
for _d in ('res', 'res2', 'res3'):
    for _k, _r in load(f'{_d}/*.json').items():
        if _k in R:
            raise SystemExit(
                f"{_k} is present in both {_seen[_k]}/ and {_d}/. Two copies of one cell "
                "means the reported number depends on load order; delete the stale one.")
        R[_k], _seen[_k] = _r, _d


def get(net, arm, s, g, c=100):
    return R.get((net, arm, s, g, c))


def table(net, arms, scope='pooled'):
    print(f'\n== {net} {scope}: mean of seeds (RMSE MAE NSE KGE PBIAS) [n seeds]')
    for g in GAPS:
        for arm in arms:
            rs = [get(net, arm, s, g) for s in SEEDS]
            rs = [r[scope] for r in rs if r and r[scope]]
            if not rs:
                continue
            m = {k: np.mean([r[k] for r in rs]) for k in ['RMSE', 'MAE', 'NSE', 'KGE', 'PBIAS']}
            print(f"{g:>3}d {arm:16s} {m['RMSE']:9.2f} {m['MAE']:8.2f} {m['NSE']:6.3f} {m['KGE']:6.3f} {m['PBIAS']:+6.1f}% [{len(rs)}]")


def paired(net, a, b, scope='pooled', gaps=GAPS, cs=(100, 100)):
    """Paired % RMSE change of arm `a` against `b` over the seed x gap cells.

    The per-cell changes are NOT exchangeable across gap lengths - on the Mahanadi
    the effect is negative at 3-30 d and positive at 100 d - so a one-sample t-test
    over all 12 cells pooled is not a valid test of a common effect. We report:
      mean/wins  descriptive, over every cell (no p);
      per_gap    mean and wins at each gap length (3 seeds; df=2, so no p either);
      p_block    a t-test over the per-gap-length means, which treats gap length as
                 the unit of replication and so does not assume a common effect.
    `p_block` is the only p-value here, and with 4 gap lengths it has df=3.
    """
    d, wins, per_gap = [], 0, {}
    for g in gaps:
        for s_ in SEEDS:
            ra, rb = get(net, a, s_, g, cs[0]), get(net, b, s_, g, cs[1])
            if not (ra and rb and ra[scope] and rb[scope]):
                continue
            x = 100 * (ra[scope]['RMSE'] - rb[scope]['RMSE']) / rb[scope]['RMSE']
            d.append(x); wins += x < 0; per_gap.setdefault(g, []).append(x)
    if len(d) < 2:
        return None
    gap_means = [np.mean(v) for v in per_gap.values()]
    if len(gap_means) >= 2:
        t, p = stats.ttest_1samp(gap_means, 0)
    else:  # single gap length: fall back to the seeds, df=2
        t, p = stats.ttest_1samp(d, 0)
    return dict(mean=np.mean(d), t=t, p_block=p, n_blocks=len(gap_means),
                wins=wins, n=len(d),
                per_gap={g: (np.mean(v), sum(x < 0 for x in v), len(v)) for g, v in per_gap.items()})


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def station_mean_change(net, a, b, gaps=GAPS):
    """Mean over scored stations of the per-station % RMSE change, averaged over cells."""
    vals = []
    for g in gaps:
        for s in SEEDS:
            ra, rb = get(net, a, s, g), get(net, b, s, g)
            if not (ra and rb):
                continue
            ch = [100 * (ra['per_station'][k]['RMSE'] - rb['per_station'][k]['RMSE']) / rb['per_station'][k]['RMSE']
                  for k in ra['per_station'] if ra['per_station'][k] and rb['per_station'].get(k)]
            vals.append((g, np.mean(ch), np.median(ch), sum(c < 0 for c in ch), len(ch)))
    return vals


def show_paired(label, r):
    if r is None:
        print(label, 'n/a'); return
    pg = '  '.join(f"{g}d {v[0]:+.1f}% ({v[1]}/{v[2]})" for g, v in sorted(r['per_gap'].items()))
    print(f"{label}: mean {r['mean']:+.2f}%  wins {r['wins']}/{r['n']}  "
          f"[blocked over {r['n_blocks']} gap lengths: t={r['t']:.2f} p={r['p_block']:.4f}] | {pg}")


for net in ['mahanadi', 'kaveri', 'us']:
    arms = ['linear', 'vanilla_seedfit', 'vanilla_single', 'vanilla_single_tf', 'pimf', 'pimfL']
    table(net, arms, 'pooled')
    if net != 'kaveri':
        table(net, arms, 'recv')
    print()
    for scope in (['pooled', 'recv'] if net != 'kaveri' else ['pooled']):
        show_paired(f'{net} {scope} PIMF vs vanilla_single', paired(net, 'pimf', 'vanilla_single', scope))
        show_paired(f'{net} {scope} PIMF vs vanilla_single_tf', paired(net, 'pimf', 'vanilla_single_tf', scope))
        show_paired(f'{net} {scope} PIMF vs vanilla_seedfit', paired(net, 'pimf', 'vanilla_seedfit', scope))
        show_paired(f'{net} {scope} PIMF vs linear', paired(net, 'pimf', 'linear', scope))
        show_paired(f'{net} {scope} PIMF-L vs PIMF @100d', paired(net, 'pimfL', 'pimf', scope, gaps=[100]))
    print(f'{net} per-station mean/median % change PIMF vs vanilla_single (gap, mean, median, n better, n):')
    for v in station_mean_change(net, 'pimf', 'vanilla_single'):
        print('   ', v[0], f'{v[1]:+.1f}', f'{v[2]:+.1f}', f'{v[3]}/{v[4]}')

print('\n== Celerity sweep, PIMF, receivers and pooled')
for g in [7, 100]:
    for scope in ['recv', 'pooled']:
        rows = {}
        for c in CS:
            rs = [get('mahanadi', 'pimf', s, g, c) for s in SEEDS]
            if all(rs):
                rows[c] = [r[scope]['RMSE'] for r in rs]
        if not rows:
            continue
        ranks = {c: [] for c in rows}
        for i in range(3):
            order = sorted(rows, key=lambda c: rows[c][i])
            for k, c in enumerate(order):
                ranks[c].append(k + 1)
        base = np.mean(rows.get(100, [np.nan]))
        print(f'-- {g}d {scope}')
        # 6 celerities are compared against c=100 in each panel; without a
        # multiplicity correction one nominal p<0.05 per panel is the null
        # expectation, so report Holm-adjusted p alongside the raw one.
        cand = [c for c in rows if c != 100]
        raw = {c: paired('mahanadi', 'pimf', 'pimf', scope, gaps=[g], cs=(c, 100))['p_block']
               for c in cand}
        adj = dict(zip(cand, holm([raw[c] for c in cand]))) if cand else {}
        for c in rows:
            m = np.mean(rows[c])
            pr = paired('mahanadi', 'pimf', 'pimf', scope, gaps=[g], cs=(c, 100)) if c != 100 else None
            ptxt = (f"paired {pr['mean']:+.1f}% p={raw[c]:.3f} p_holm={adj[c]:.3f}" if pr else '')
            print(f'  c={c:>3} {m:8.2f} ({100*(m-base)/base:+5.1f}% of means) ranks {ranks[c]} {ptxt}')
