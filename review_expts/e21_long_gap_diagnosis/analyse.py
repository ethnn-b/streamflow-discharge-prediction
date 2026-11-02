import glob, pickle, sys
import numpy as np
import pandas as pd

out = sys.argv[1]
arms = ['vanilla', 'pimf', 'pimfL', 'pimf_warm']
cells, imps = [], []
for f in sorted(glob.glob(out + '/cell_*.pkl')):
    d = pickle.load(open(f, 'rb'))
    c = d['cells'].assign(seed=d['seed'], gap=d['gap'])
    c['sd'] = c.station.map(d['sd'])
    cells.append(c)
    imps.append(pd.DataFrame(d['importances']).T.assign(seed=d['seed'], gap=d['gap']))
c = pd.concat(cells)
c['depth'] = np.minimum(c.gprev, c.gnext)
c['hybrid5'] = np.where(c.depth <= 5, c.pimf, c.pimfL)
c['hybrid10'] = np.where(c.depth <= 10, c.pimf, c.pimfL)
arms += ['hybrid5', 'hybrid10', 'init']

def summ(g):
    r = {}
    for k in arms:
        e = g[k] - g.obs
        r[k + '_rmse'] = np.sqrt((e ** 2).mean())
        r[k + '_pb'] = 100 * e.sum() / g.obs.sum()
    return pd.Series(r)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('== pooled RMSE per (gap, seed) ==')
print(c.groupby(['gap', 'seed']).apply(summ)[[k + '_rmse' for k in arms]].round(1))
print('\n== pooled PBIAS per (gap, seed) ==')
print(c.groupby(['gap', 'seed']).apply(summ)[[k + '_pb' for k in arms]].round(1))

# Scale-free view: per-station RMSE / station sd, median across station-seed
print('\n== station-normalised RMSE (RMSE/sd), median over station x seed ==')
ns = c.groupby(['gap', 'seed', 'station']).apply(
    lambda g: pd.Series({k: np.sqrt(((g[k] - g.obs) ** 2).mean()) / g.sd.iloc[0] for k in arms}))
print(ns.groupby('gap').median().round(3))
print('\n== station x seed wins vs vanilla (fraction lower RMSE) ==')
print(ns.groupby('gap').apply(lambda t: pd.Series({k: (t[k] < t.vanilla).mean() for k in arms if k != 'vanilla'})).round(2))

bins = [0, 1, 3, 5, 10, 20, 35, 50]
c['dbin'] = pd.cut(c.depth, bins, right=True)
print('\n== by depth into gap (days to nearest observation): normalised MAE = mean|err|/sd, and signed bias/sd ==')
for g, t in c.groupby('gap'):
    rows = {}
    for b, u in t.groupby('dbin', observed=True):
        r = {'n': len(u)}
        for k in ['vanilla', 'pimf', 'pimfL', 'pimf_warm', 'init']:
            r[k + '_nmae'] = ((u[k] - u.obs).abs() / u.sd).mean()
            r[k + '_nbias'] = ((u[k] - u.obs) / u.sd).mean()
        rows[str(b)] = r
    print(f'gap {g}'); print(pd.DataFrame(rows).T.round(3))

print('\n== how close PIMF is to the init vs to vanilla, deep in 100-d gaps (depth > 10) ==')
u = c[(c.gap == 100) & (c.depth > 10)]
for k in ['pimf', 'pimfL', 'pimf_warm', 'vanilla']:
    print(k, 'corr(pred, init)=%.3f corr(pred, obs)=%.3f' % (np.corrcoef(u[k], u.init)[0, 1], np.corrcoef(u[k], u.obs)[0, 1]))

print('\n== PIMF feature importances (mean over stations, seeds) ==')
I = pd.concat(imps)
cols = [k for k in ['lag_prev', 'lag_next', 'gap_dist_prev', 'gap_dist_next', 'weighted_upstream_discharge',
                    'routed_upstream_discharge', 'day_of_year_sin', 'day_of_year_cos'] if k in I]
print(I.groupby('gap')[cols].mean().round(4))
print('max importance of gap_dist features anywhere:', I[['gap_dist_prev', 'gap_dist_next']].max().max())
