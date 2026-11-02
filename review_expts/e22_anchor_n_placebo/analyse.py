"""Summarise e22/e23-style cells. Usage: python analyse.py <folder of cell_*.pkl>

Statistics are station-level: each station's RMSE ratio is averaged (geometrically)
over seeds, so a station counts once, and tests are Wilcoxon over stations with a
Holm correction across gap lengths. Case-level (station x seed) figures are shown
for reference only: the cases of one station are not independent.
"""
import glob, pickle, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Pickles written before the rename: 'pimf' was the old PIMF, 'pimfA' the final one.
LEGACY_NAMES = {'pimf': 'pimf_old', 'pimfA': 'pimf'}
ARMS = ['vanilla', 'pimf', 'pimf_old', 'pimfL', 'pimfL_placebo']

out = sys.argv[1]
cs = []
for f in sorted(glob.glob(out + '/cell_*.pkl')):
    d = pickle.load(open(f, 'rb'))
    c = d['cells']
    if 'pimfA' in c:
        c = c.rename(columns=LEGACY_NAMES)
    c = c.assign(seed=d['seed'], gap=d['gap'])
    cs.append(c); receivers = d['receivers']; placebo = d['placebo']
c = pd.concat(cs)
arms = [a for a in ARMS if a in c]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('seeds per gap:', c.groupby('gap').seed.nunique().to_dict(),
      '| scored stations:', c.station.nunique(), sorted(c.station.unique()))


def rmse(g, k):
    return np.sqrt(((g[k] - g.obs) ** 2).mean())


def holm(ps):
    ps = pd.Series(ps)
    order = ps.sort_values().index
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(ps) - i)))
        adj[k] = running
    return pd.Series(adj)[ps.index]


def pooled(sub, title):
    print(f'\n######## {title}')
    per = sub.groupby(['gap', 'seed']).apply(lambda g: pd.Series({k: rmse(g, k) for k in arms}))
    pb = sub.groupby(['gap', 'seed']).apply(lambda g: pd.Series({k: 100 * (g[k] - g.obs).sum() / g.obs.sum() for k in arms}))
    print('-- pooled RMSE (cumec), mean over seeds'); print(per.groupby('gap').mean().round(1))
    ch = per[arms[1:]].div(per.vanilla, axis=0).sub(1).mul(100)
    print('-- pooled RMSE % change vs vanilla, mean over seeds (seeds better)')
    print(ch.groupby('gap').agg(lambda x: f'{x.mean():+.1f}% ({(x < 0).sum()}/{len(x)})'))
    print('-- pooled PBIAS %, mean over seeds'); print(pb.groupby('gap').mean().round(1))


pooled(c, 'POOLED over all scored gauges')
pooled(c[c.station.isin(receivers)], 'RECEIVERS (' + ', '.join(receivers) + ')')

st = c.groupby(['gap', 'seed', 'station']).apply(lambda g: pd.Series({k: rmse(g, k) for k in arms}))
pbs = c.groupby(['gap', 'seed', 'station']).apply(
    lambda g: pd.Series({k: abs(100 * (g[k] - g.obs).sum() / g.obs.sum()) for k in arms}))


def station_test(num, den, title):
    """Per gap: geometric-mean RMSE ratio num/den over stations, stations won, Wilcoxon p, Holm p."""
    lr = np.log(st[num] / st[den]).groupby(['gap', 'station']).mean()
    rows = {gap: {'ratio': np.exp(t.mean()), 'won': f'{(t < 0).sum()}/{len(t)}',
                  'p': wilcoxon(t).pvalue} for gap, t in lr.groupby('gap')}
    T = pd.DataFrame(rows).T
    T['p_holm'] = holm(T.p.astype(float))
    print(f'-- {title}'); print(T.round(3))


print('\n######## STATION LEVEL (each station averaged over seeds; ratio < 1 = first arm better)')
for k in arms[1:]:
    station_test(k, 'vanilla', f'{k} vs vanilla')
if {'pimf', 'pimfL'} <= set(arms):
    station_test('pimf', 'pimfL', 'pimf vs pimfL (does N add anything?)')
if {'pimf', 'pimf_old'} <= set(arms):
    station_test('pimf', 'pimf_old', 'pimf vs pimf_old')
print('-- median |PBIAS| % over station x seed'); print(pbs.groupby('gap').median().round(1))
print('-- case level, for reference only: share of station x seed cases beating vanilla')
print(st.groupby('gap').apply(lambda t: pd.Series({k: (t[k] < t.vanilla).mean() for k in arms[1:]})).round(2))

if 'pimfL_placebo' in arms:
    print('\n######## PLACEBO: real R vs placebo R (pimfL vs pimfL_placebo) at the receivers')
    rr = np.log(st.pimfL / st.pimfL_placebo).loc[st.index.get_level_values('station').isin(receivers)]
    per_station = rr.groupby('station').mean()
    tab = pd.DataFrame({'ratio real/placebo': np.exp(per_station),
                        'cells won': rr.groupby('station').apply(lambda x: f'{(x < 0).sum()}/{len(x)}')})
    print(tab.round(3))
    print(f'receivers favouring real R: {(per_station < 0).sum()}/{len(per_station)}, '
          f'Wilcoxon over receivers p = {wilcoxon(per_station).pvalue:.3f} '
          f'(n = {len(per_station)}: the smallest possible two-sided p is {2 / 2 ** len(per_station):.3f})')
    print('placebo map:', {k: [(s, l) for s, l, _ in v] for k, v in placebo.items()})
