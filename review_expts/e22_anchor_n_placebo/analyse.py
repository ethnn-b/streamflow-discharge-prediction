import glob, pickle, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = sys.argv[1]
ARMS = ['vanilla', 'pimf', 'pimfL', 'pimfA', 'pimfL_placebo']
cs = []
for f in sorted(glob.glob(out + '/cell_*.pkl')):
    d = pickle.load(open(f, 'rb'))
    c = d['cells'].assign(seed=d['seed'], gap=d['gap'])
    c['sd'] = c.station.map(d['sd'])
    cs.append(c); receivers = d['receivers']; placebo = d['placebo']
c = pd.concat(cs)
arms = [a for a in ARMS if a in c]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('cells per gap:', c.groupby('gap').seed.nunique().to_dict())

def rmse(g, k): return np.sqrt(((g[k] - g.obs) ** 2).mean())

def block(sub, title):
    print(f'\n######## {title}')
    per = sub.groupby(['gap', 'seed']).apply(lambda g: pd.Series({k: rmse(g, k) for k in arms}))
    pb = sub.groupby(['gap', 'seed']).apply(lambda g: pd.Series({k: 100 * (g[k] - g.obs).sum() / g.obs.sum() for k in arms}))
    rows = []
    for gap, t in per.groupby('gap'):
        r = {'gap': gap, 'n_seeds': len(t)}
        for k in arms:
            r[k] = t[k].mean()
        for k in arms[1:]:
            ch = 100 * (t[k] / t.vanilla - 1)
            r[f'{k} vs van %'] = ch.mean()
            r[f'{k} wins'] = f'{(ch < 0).sum()}/{len(ch)}'
            r[f'{k} p'] = wilcoxon(ch).pvalue if len(ch) > 5 and (ch != 0).any() else np.nan
        rows.append(r)
    T = pd.DataFrame(rows).set_index('gap')
    print('-- mean pooled RMSE over seeds'); print(T[arms].round(1))
    print('-- paired % change vs vanilla (mean over seeds), wins, Wilcoxon p')
    print(T[[c for c in T if ' vs van' in c or 'wins' in c or c.endswith(' p')]].round(3))
    print('-- mean pooled PBIAS %'); print(pb.groupby('gap').mean().round(1))

block(c, 'POOLED over all scored gauges')
block(c[c.station != 'tikarapara'], 'POOLED excluding tikarapara')
block(c[c.station.isin(receivers)], 'RECEIVERS (' + ', '.join(receivers) + ')')

print('\n######## PER STATION (scale-free): RMSE/sd per station x seed')
ns = c.groupby(['gap', 'seed', 'station']).apply(lambda g: pd.Series({k: rmse(g, k) / g.sd.iloc[0] for k in arms}))
print('-- median RMSE/sd'); print(ns.groupby('gap').median().round(3))
print('-- fraction of station x seed where arm beats vanilla, and Wilcoxon p on log-ratio')
rows = {}
for gap, t in ns.groupby('gap'):
    r = {}
    for k in arms[1:]:
        lr = np.log(t[k] / t.vanilla)
        r[k + ' win'] = (lr < 0).mean(); r[k + ' p'] = wilcoxon(lr).pvalue
    rows[gap] = r
print(pd.DataFrame(rows).T.round(3))
print('-- pimfA vs pimf and pimfA vs pimfL (station x seed win fraction)')
print(ns.groupby('gap').apply(lambda t: pd.Series({'A beats pimf': (t.pimfA < t.pimf).mean(),
                                                    'A beats L': (t.pimfA < t.pimfL).mean()})).round(2))

print('\n######## PLACEBO: real R vs placebo R (pimfL vs pimfL_placebo), per receiver, % RMSE change real vs placebo')
rr = c[c.station.isin(receivers)].groupby(['gap', 'seed', 'station']).apply(
    lambda g: 100 * (rmse(g, 'pimfL') / rmse(g, 'pimfL_placebo') - 1)).rename('d').reset_index()
tab = rr.groupby(['station', 'gap']).d.mean().unstack().round(2)
tab['all gaps'] = rr.groupby('station').d.mean().round(2)
tab['wins'] = rr.groupby('station').d.apply(lambda x: f'{(x < 0).sum()}/{len(x)}')
tab['p'] = rr.groupby('station').d.apply(lambda x: wilcoxon(x).pvalue).round(4)
print(tab)
print('pooled over receivers, each cell:', rr.d.mean().round(2), 'p', wilcoxon(rr.d).pvalue)
print('placebo map:', {k: [(s, l) for s, l, _ in v] for k, v in placebo.items()})
