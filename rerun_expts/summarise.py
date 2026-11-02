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


R = load('res/*.json')
R.update(load('res2/*.json'))
R.update(load('res3/*.json'))  # US runs with the -999999 sentinel masked


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
    d, wins, per_gap = [], 0, {}
    for g in gaps:
        for s in SEEDS:
            ra, rb = get(net, a, s, g, cs[0]), get(net, b, s, g, cs[1])
            if not (ra and rb and ra[scope] and rb[scope]):
                continue
            x = 100 * (ra[scope]['RMSE'] - rb[scope]['RMSE']) / rb[scope]['RMSE']
            d.append(x); wins += x < 0; per_gap.setdefault(g, []).append(x)
    if len(d) < 2:
        return None
    t, p = stats.ttest_1samp(d, 0)
    return dict(mean=np.mean(d), t=t, p=p, wins=wins, n=len(d),
                per_gap={g: (np.mean(v), sum(x < 0 for x in v), len(v)) for g, v in per_gap.items()})


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
    print(f"{label}: mean {r['mean']:+.2f}%  t={r['t']:.2f} p={r['p']:.4f} wins {r['wins']}/{r['n']} | {pg}")


for net in ['mahanadi', 'kaveri', 'us']:
    arms = ['linear', 'vanilla_seedfit', 'vanilla_single', 'pimf', 'pimfL']
    table(net, arms, 'pooled')
    if net != 'kaveri':
        table(net, arms, 'recv')
    print()
    for scope in (['pooled', 'recv'] if net != 'kaveri' else ['pooled']):
        show_paired(f'{net} {scope} PIMF vs vanilla_single', paired(net, 'pimf', 'vanilla_single', scope))
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
        for c in rows:
            m = np.mean(rows[c])
            pr = paired('mahanadi', 'pimf', 'pimf', scope, gaps=[g], cs=(c, 100)) if c != 100 else None
            ptxt = f"paired {pr['mean']:+.1f}% p={pr['p']:.3f}" if pr else ''
            print(f'  c={c:>3} {m:8.2f} ({100*(m-base)/base:+5.1f}% of means) ranks {ranks[c]} {ptxt}')
