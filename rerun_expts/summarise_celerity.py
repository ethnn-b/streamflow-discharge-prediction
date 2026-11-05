"""Celerity sweep readout under the fold protocol.

The headline test is "any time shift vs none": per (fold, seed) cell, the mean
RMSE over the interior celerities is compared against c = 350, where every lag
rounds to zero. That is one paired observation per cell, 15 per gap length, and
it holds the design matrix fixed - same columns, same weights, only the day each
upstream series is read differs.

The per-celerity rows then say whether the *value* of c matters beyond that, and
carry a Holm correction over the six comparisons against c = 100.
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CS = [25, 50, 75, 100, 150, 200, 350]
INTERIOR = [50, 75, 100, 150, 200]      # lags resolve to >=1 day for some pairs
NULL_C = 350                            # every tau rounds to 0
GAPS = [7, 100]

NET = 'mahanadi'            # the LIVE map below and the whole sweep are Mahanadi-only
R = {}
for pat in ('res_celerity/*.json', 'res_main/*.json'):   # c=100 cells live in res_main
    for f in glob.glob(os.path.join(HERE, pat)):
        r = json.load(open(f))
        # The net MUST be in the key. res_main holds us_ and kaveri_ cells whose arm
        # names and celerity collide with the Mahanadi ones, so a net-less key let the
        # US c=100 cell overwrite the Mahanadi reference row, in whatever order glob
        # happened to return the files.
        if r['net'] != NET:
            continue
        R[(r['arm'], int(r['celerity']), r['fold'], r['seed'], r['gap'])] = r
CELLS = sorted({(k[2], k[3]) for k in R if k[1] != 100})


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def rmse(arm, c, f, s, g, scope):
    r = R.get((arm, c, f, s, g))
    return r[scope]['RMSE'] if r and r.get(scope) else None


for arm in ('R', 'AR'):
    for scope in ('recv', 'pooled'):
        for g in GAPS:
            rows = {c: [rmse(arm, c, f, s, g, scope) for f, s in CELLS] for c in CS}
            rows = {c: v for c, v in rows.items() if all(x is not None for x in v)}
            if NULL_C not in rows or len(rows) < 3:
                continue
            print(f'\n== {arm}  {scope}  {g}d   ({len(CELLS)} fold x seed cells)')

            # headline: mean over interior celerities vs the no-shift control
            inter = [c for c in INTERIOR if c in rows]
            d = []
            for i in range(len(CELLS)):
                lo = np.mean([rows[c][i] for c in inter])
                d.append(100 * (lo - rows[NULL_C][i]) / rows[NULL_C][i])
            t, p = stats.ttest_1samp(d, 0)
            print(f'  time shift (mean of c={inter} vs c={NULL_C}): '
                  f'{np.mean(d):+.2f}%  median {np.median(d):+.2f}%  '
                  f'better in {sum(x < 0 for x in d)}/{len(d)}  t={t:+.2f} p={p:.4f}')

            # per-celerity detail against c=100
            base = np.mean(rows[100]) if 100 in rows else np.nan
            cand = [c for c in rows if c != 100]
            raw = {}
            for c in cand:
                dd = [100 * (rows[c][i] - rows[100][i]) / rows[100][i] for i in range(len(CELLS))]
                raw[c] = (np.mean(dd), np.median(dd), sum(x < 0 for x in dd),
                          stats.ttest_1samp(dd, 0)[1])
            adj = dict(zip(cand, holm([raw[c][3] for c in cand]))) if cand else {}
            print(f"  {'c':>5} {'RMSE':>9} {'vs c=100':>9} {'median':>8} {'better':>8} "
                  f"{'p':>7} {'p_holm':>7}")
            for c in CS:
                if c not in rows:
                    continue
                m = np.mean(rows[c])
                if c == 100:
                    print(f'  {c:>5} {m:>9.2f} {"ref":>9}')
                else:
                    mu, md, w, p_ = raw[c]
                    print(f'  {c:>5} {m:>9.2f} {mu:>+8.1f}% {md:>+7.1f}% {w:>5}/{len(CELLS)} '
                          f'{p_:>7.3f} {adj[c]:>7.3f}')


# --- per-gauge: does the gain follow which gauges are live at each celerity? ---
# simga's only contributor sits at 63.3 km, so it is live at c <= 100 and dead at
# c >= 150. rajim's at 6.8 km is dead everywhere. If routing acts through travel
# time, each gauge's advantage over the no-shift control should track its own
# live/dead status, not just the celerity.
LIVE = {25: {'basantpur', 'jondhra', 'seorinarayan', 'simga'},
        50: {'basantpur', 'jondhra', 'seorinarayan', 'simga'},
        75: {'basantpur', 'jondhra', 'seorinarayan', 'simga'},
        100: {'basantpur', 'jondhra', 'seorinarayan', 'simga'},
        150: {'basantpur', 'jondhra', 'seorinarayan'},
        200: {'basantpur', 'jondhra', 'seorinarayan'},
        350: set()}

for arm in ('R', 'AR'):
    for g in GAPS:
        have = [c for c in CS if all(R.get((arm, c, f, s, g)) for f, s in CELLS)]
        if NULL_C not in have or len(have) < 3:
            continue
        gauges = R[(arm, NULL_C, *CELLS[0], g)]['receivers']
        print(f'\n== {arm} {g}d: per-gauge median % RMSE vs the c={NULL_C} no-shift control'
              f'   (* = that gauge is live at that c)')
        print('  ' + 'gauge'.ljust(14) + ''.join(f'{c:>9}' for c in have if c != NULL_C))
        for gauge in sorted(gauges, key=lambda x: -len([c for c in have if x in LIVE[c]])):
            cells = []
            for c in have:
                if c == NULL_C:
                    continue
                d = []
                for f, s in CELLS:
                    a, b = R[(arm, c, f, s, g)], R[(arm, NULL_C, f, s, g)]
                    pa, pb = a['per_station'].get(gauge), b['per_station'].get(gauge)
                    if pa and pb and pb['RMSE'] > 0:
                        d.append(100 * (pa['RMSE'] - pb['RMSE']) / pb['RMSE'])
                mark = '*' if gauge in LIVE[c] else ' '
                cells.append(f'{np.median(d):>+8.1f}{mark}' if d else f'{"-":>9}')
            print('  ' + gauge.ljust(14) + ''.join(cells))
