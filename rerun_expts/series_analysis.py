"""Analyses that need the imputed series themselves (res_series/series/*.npz):

  seasonal and flow-regime stratification, peak magnitude and timing, per-gauge
  results against catchment area, and an upstream-downstream consistency check.
Also summarises res_missing (5% and 20% missingness) and the Kalman annual-cycle variant.

Run from the repository root:  python rerun_expts/series_analysis.py
Series are produced by:  python rerun_expts/drive_ablation.py mahanadi --protocol transductive \
    --arms ARLanc,vanilla,linear,histmean,knn,kalman,kalmanfour --out res_series --series
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); os.chdir(os.path.dirname(HERE))
from rerun_expts.common import load_network

SEEDS, GAPS = (42, 1, 2, 3, 4), (3, 7, 30, 100)
AREA = {'andhiyarkhore': 2236.21, 'bamnidhi': 9736.59, 'baronda': 3191.19, 'basantpur': 53160.35, 'ghatora': 2930.69,
        'jondhra': 21841.90, 'kotni': 7011.30, 'patharidih': 2487.04, 'rajim': 8390.39, 'rampur': 3405.41,
        'seorinarayan': 44513.53, 'simga': 16675.00}   # Table 1 of the manuscript
df, ST, _, D, C, RECV = load_network('mahanadi')
SC = {s: float(df[s].mean()) for s in ST}
Q90 = {s: float(df[s].quantile(.9)) for s in ST}; Q10 = {s: float(df[s].quantile(.1)) for s in ST}
SER = 'rerun_expts/res_series/series'

def load(arm, seed, gap):
    z = np.load(f'{SER}/mahanadi_{arm}_f0_s{seed}_{gap}.npz')
    return pd.DatetimeIndex(z['index']), [str(x) for x in z['stations']], z['obs'], z['imp'], z['withheld']

def blocked(vals_by_gap):   # vals_by_gap: {gap: [pct diffs]} -> mean, negatives, p (df = 3)
    allv = [x for v in vals_by_gap.values() for x in v]
    t, p = stats.ttest_1samp([np.mean(v) for v in vals_by_gap.values()], 0)
    return np.mean(allv), sum(x < 0 for x in allv), len(allv), p

def cellmetric(arm, seed, gap, sel, metric):
    """Mean over gauges of the gauge-normalised error on the withheld cells that satisfy sel(index, gauge, obs)."""
    idx, stn, obs, imp, w = load(arm, seed, gap); out = []
    for j, s in enumerate(stn):
        m = w[:, j] & sel(idx, s, obs[:, j])
        if m.sum() < 3: continue
        e = imp[m, j] - obs[m, j]
        out.append(100 * (np.sqrt((e ** 2).mean()) if metric == 'RMSE' else np.abs(e).mean()) / SC[s] if metric != 'PBIAS'
                   else 100 * e.sum() / obs[m, j].sum())
    return np.mean(out) if out else np.nan

def contrast(sel, metric, a='ARLanc', b='vanilla'):
    d = {g: [] for g in GAPS}
    for g in GAPS:
        for s in SEEDS:
            xa, xb = cellmetric(a, s, g, sel, metric), cellmetric(b, s, g, sel, metric)
            if np.isfinite(xa) and np.isfinite(xb): d[g].append(100 * (xa - xb) / abs(xb))
    return blocked(d)

def fmt(r): return f'{r[0]:+.1f}% ({r[1]}/{r[2]} cells lower, p={r[3]:.3f})'

if __name__ == '__main__':
    # the saved series reproduce the stored results (pooled RMSE of every PIMF cell)
    worst = 0
    for sd in SEEDS:
        for g in GAPS:
            idx, stn, obs, imp, w = load('ARLanc', sd, g)
            rm = np.sqrt(np.mean((imp[w] - obs[w]) ** 2))
            worst = max(worst, abs(rm - json.load(open(f'rerun_expts/res_main/mahanadi_ARLanc_f0_s{sd}_{g}.json'))['pooled']['RMSE']))
    print(f'series vs res_main pooled RMSE, largest difference over 20 PIMF cells: {worst:.2e}')
    print('== seasonal (wet = Jun-Oct, dry = Nov-May); PIMF vs vanilla, gauge-normalised error')
    seasons = {'wet': lambda i, s, o: np.isin(i.month, range(6, 11)), 'dry': lambda i, s, o: ~np.isin(i.month, range(6, 11))}
    for k, f in seasons.items():
        for m in ('MAE', 'RMSE'): print(f'  {k:4s} n{m}:', fmt(contrast(f, m)))
    print('== flow regime (gauge-specific percentiles of the 1980-90 record)')
    reg = {'high (>=Q90)': lambda i, s, o: o >= Q90[s], 'low (<=Q10)': lambda i, s, o: o <= Q10[s], 'middle': lambda i, s, o: (o > Q10[s]) & (o < Q90[s])}
    for k, f in reg.items():
        for m in ('MAE', 'RMSE'): print(f'  {k:13s} n{m}:', fmt(contrast(f, m)))
        print(f'  {k:13s} |PBIAS| change:', end=' ')
        d = {g: [] for g in GAPS}
        for g in GAPS:
            for s in SEEDS:
                xa, xb = cellmetric('ARLanc', s, g, f, 'PBIAS'), cellmetric('vanilla', s, g, f, 'PBIAS')
                if np.isfinite(xa) and np.isfinite(xb): d[g].append(100 * (abs(xa) - abs(xb)) / abs(xb))
        print(fmt(blocked(d)))
        for arm in ('ARLanc', 'vanilla'):
            print(f'     mean PBIAS {arm}: {np.nanmean([cellmetric(arm, s, g, f, "PBIAS") for g in GAPS for s in SEEDS]):+.1f}%')
    print('== peaks: segments of >=7 withheld days whose observed maximum is >= the gauge Q90')
    for arm in ('ARLanc', 'vanilla', 'linear', 'knn'):
        mag, tim = [], []
        for g in (7, 30, 100):
            for s in SEEDS:
                idx, stn, obs, imp, w = load(arm, s, g)
                for j, st in enumerate(stn):
                    x = w[:, j].astype(int); edges = np.flatnonzero(np.diff(np.r_[0, x, 0]))
                    for a_, b_ in zip(edges[::2], edges[1::2]):
                        if b_ - a_ < 7 or obs[a_:b_, j].max() < Q90[st]: continue
                        mag.append(100 * (imp[a_:b_, j].max() - obs[a_:b_, j].max()) / obs[a_:b_, j].max())
                        tim.append(int(np.argmax(imp[a_:b_, j]) - np.argmax(obs[a_:b_, j])))
        tim = np.array(tim)
        print(f'  {arm:8s} n={len(mag):4d} peak magnitude error median {np.median(mag):+.1f}% (IQR {np.percentile(mag,25):+.1f}..{np.percentile(mag,75):+.1f}); '
              f'|timing error| median {np.median(np.abs(tim)):.0f} d, within 1 d {100*np.mean(np.abs(tim)<=1):.0f}%, within 3 d {100*np.mean(np.abs(tim)<=3):.0f}%')
    print('== per gauge (mean over the 20 cells): area (km2), native missing %, nMAE and nRMSE, vanilla -> PIMF')
    allsel = lambda i, s_, o: np.ones(len(i), bool)
    for j, st in enumerate(ST):
        v = {}
        for arm in ('vanilla', 'ARLanc'):
            for m in ('MAE', 'RMSE'):
                xs = []
                for g in GAPS:
                    for sd in SEEDS:
                        idx, stn, obs, imp, w = load(arm, sd, g)
                        if not w[:, j].any(): continue   # no withheld cell at this gauge in this cell
                        e = imp[w[:, j], j] - obs[w[:, j], j]
                        xs.append(100 * (np.abs(e).mean() if m == 'MAE' else np.sqrt((e ** 2).mean())) / SC[st])
                v[arm, m] = np.mean(xs)
        print(f'  {st:14s} {AREA[st]:9.0f} {100*df[st].isna().mean():5.1f}  nMAE {v["vanilla","MAE"]:6.1f} -> {v["ARLanc","MAE"]:6.1f} ({100*(v["ARLanc","MAE"]/v["vanilla","MAE"]-1):+.1f}%)'
              f'  nRMSE {v["vanilla","RMSE"]:6.1f} -> {v["ARLanc","RMSE"]:6.1f} ({100*(v["ARLanc","RMSE"]/v["vanilla","RMSE"]-1):+.1f}%)')
    print('== upstream-downstream consistency over the 28 pairs, on days where the downstream value was withheld')
    print('   rho = ln(Q_down(t) / Q_up(t - tau)); error = |rho_imputed - rho_observed|; reversal = sign(rho) differs from the observed sign')
    pairs = [(r, u, int(round(D.loc[r, u] / 100.0))) for r in RECV for u in ST if u != r and C.loc[r, u] > 0]
    for arm in ('ARLanc', 'vanilla', 'linear', 'knn'):
        err, rev, n = [], 0, 0
        for g in GAPS:
            for sd in SEEDS:
                idx, stn, obs, imp, w = load(arm, sd, g); ix = {s_: k for k, s_ in enumerate(stn)}
                for r, u, tau in pairs:
                    jr, ju = ix[r], ix[u]
                    t = np.flatnonzero(w[:, jr]); t = t[t >= tau]
                    ok = np.isfinite(obs[t, jr]) & np.isfinite(obs[t - tau, ju]) & (obs[t, jr] > 0) & (obs[t - tau, ju] > 0)
                    t = t[ok]
                    if not len(t): continue
                    ro = np.log(obs[t, jr] / obs[t - tau, ju]); ri = np.log(np.maximum(imp[t, jr], 1e-3) / np.maximum(imp[t - tau, ju], 1e-3))
                    err += list(np.abs(ri - ro)); rev += int((np.sign(ri) != np.sign(ro)).sum()); n += len(t)
        print(f'  {arm:8s} n={n}  median |rho error| {np.median(err):.3f}  mean {np.mean(err):.3f}  sign reversals {100*rev/n:.1f}%')
    print('== missingness 5% / 20% (res_missing), PIMF vs vanilla')
    for pct in (5, 20):
        R = {}
        for f in glob.glob(f'rerun_expts/res_missing/mahanadi_*_p{pct}.json'):
            r = json.load(open(f)); R[(r['arm'], r['seed'], r['gap'])] = r
        keep = set(ST)
        for m in ('MAE', 'RMSE'):
            d = {g: [] for g in GAPS}
            for g in GAPS:
                for sd in SEEDS:
                    if ('ARLanc', sd, g) in R and ('vanilla', sd, g) in R:
                        f_ = lambda a: np.mean([100 * v[m] / SC[s_] for s_, v in R[(a, sd, g)]['per_station'].items() if v and s_ in keep])
                        d[g].append(100 * (f_('ARLanc') - f_('vanilla')) / f_('vanilla'))
            if all(d.values()): print(f'  {pct}% n{m}:', fmt(blocked(d)))
            else: print(f'  {pct}% n{m}: incomplete', {g: len(v) for g, v in d.items()})
        for arm in ('ARLanc', 'vanilla', 'linear', 'knn'):
            for m in ('MAE', 'RMSE'):
                print(f'    {arm:8s} n{m} by gap:', [round(np.mean([np.mean([100 * v[m] / SC[s_] for s_, v in R[(arm, sd, g)]['per_station'].items() if v]) for sd in SEEDS if (arm, sd, g) in R]), 1) for g in GAPS])
    print('== Kalman-SARIMA with an annual cycle (2 Fourier harmonics) vs the weekly-seasonal original')
    for arm in ('kalman', 'kalmanfour'):
        for m in ('MAE', 'RMSE'):
            row = []
            for g in GAPS:
                v = [np.mean([100 * x[m] / SC[s_] for s_, x in json.load(open(f))['per_station'].items() if x])
                     for f in glob.glob(f'rerun_expts/res_series/mahanadi_{arm}_f0_s*_{g}.json')]
                row.append(round(np.mean(v), 1))
            print(f'  {arm:11s} n{m} by gap {row}')
