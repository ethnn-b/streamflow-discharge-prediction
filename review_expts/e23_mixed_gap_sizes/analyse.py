import glob, pickle, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Pickles written before the rename: 'pimf' was the old PIMF, 'pimfA' the final one.
LEGACY_NAMES = {'pimf': 'pimf_old', 'pimfA': 'pimf'}
arms = ['vanilla', 'pimf', 'pimf_old']
cs = []
for f in sorted(glob.glob(sys.argv[1] + '/cell_*.pkl')):
    d = pickle.load(open(f, 'rb'))
    c = d['cells']
    if 'pimfA' in c:
        c = c.rename(columns=LEGACY_NAMES)
    c = c.assign(seed=d['seed']); c['sd'] = c.station.map(d['sd']); cs.append(c)
c = pd.concat(cs)
c['size'] = pd.cut(c.run_len, [0, 7, 30, 100, 10_000], labels=['2-7 d', '8-30 d', '31-100 d', '>100 d'])
pd.set_option('display.width', 200)
rmse = lambda g, k: np.sqrt(((g[k] - g.obs) ** 2).mean())

print('scored cells by gap size:', c.groupby('size', observed=True).size().to_dict())
print('gap-run length summary:', c.groupby(['seed', 'station']).run_len.max().describe().round(0).to_dict())

for name, sub in [('ALL GAP SIZES TOGETHER', c)] + [(f'GAP SIZE {s}', g) for s, g in c.groupby('size', observed=True)]:
    print(f'\n#### {name}  (n={len(sub)})')
    per = sub.groupby('seed').apply(lambda g: pd.Series({k: rmse(g, k) for k in arms}))
    pb = sub.groupby('seed').apply(lambda g: pd.Series({k: 100 * (g[k] - g.obs).sum() / g.obs.sum() for k in arms}))
    print('pooled RMSE per seed'); print(per.round(1))
    print('pooled PBIAS % per seed'); print(pb.round(1))
    ns = sub.groupby(['seed', 'station']).apply(lambda g: pd.Series({k: rmse(g, k) / g.sd.iloc[0] for k in arms}))
    r = {}
    for k in ['pimf', 'pimf_old']:
        lr = np.log(ns[k] / ns.vanilla)
        r[f'{k} beats vanilla'] = f'{(lr < 0).mean():.0%} of {len(lr)}, p={wilcoxon(lr).pvalue:.3f}' if len(lr) > 5 else f'{(lr < 0).mean():.0%} of {len(lr)}'
    lr = np.log(ns.pimf / ns.pimf_old)
    r['pimf beats pimf_old'] = f'{(lr < 0).mean():.0%} of {len(lr)}, p={wilcoxon(lr).pvalue:.3f}' if len(lr) > 5 else f'{(lr < 0).mean():.0%} of {len(lr)}'
    print('per station-seed:', r, '| median RMSE/sd', ns.median().round(3).to_dict())
