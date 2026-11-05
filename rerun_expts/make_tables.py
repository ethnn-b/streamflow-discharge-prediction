"""Emit the LaTeX tables of Section 5.1 straight from the result JSONs.

Writes <paper>/tables_main.tex, which main.tex \\inputs. Numbers in the paper are
therefore never transcribed by hand: re-run this after any sweep and the tables
follow. Usage: python rerun_expts/make_tables.py [net] [resdir] [paper_dir]
"""
import contextlib, glob, io, json, os, sys, warnings
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter('ignore')
from rerun_expts.common import load_network, connected

NET = sys.argv[1] if len(sys.argv) > 1 else 'mahanadi'
RESDIR = sys.argv[2] if len(sys.argv) > 2 else 'res_main'
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = sys.argv[3] if len(sys.argv) > 3 else 'paper_latex_v6'
OUT = os.path.join(os.path.dirname(HERE), PAPER, 'tables_main.tex')

GAPS = [3, 7, 30, 100]
# display name -> arm key in the JSONs
ROWS = [('Historical mean', 'histmean'), ('Linear interpolation', 'linear'),
        ('Kalman (SARIMAX)', 'kalman'), ('kNN ($k=5$)', 'knn'),
        ('Vanilla MissForest', 'vanilla'), ('$\\varnothing$ (ablation base)', 'base'),
        ('PIMF-L $\\{A,R\\}$', 'AR'), ('PIMF$_{\\mathrm{naive}}$ $\\{A,R,L\\}$', 'ARL'),
        ('\\textbf{PIMF} $\\{A,R,N\\}$', 'ARLanc'),
        ('PIMF, placebo routing', 'ARplac')]

R = {}
for f in glob.glob(os.path.join(HERE, RESDIR, '*.json')):
    r = json.load(open(f))
    if r['net'] == NET:
        R[(r['arm'], r['seed'], r['gap'])] = r
SEEDS = sorted({k[1] for k in R})
PROTO = sorted({r.get('protocol', 'fold') for r in R.values()})
assert R, f'no results for {NET} in {RESDIR}'

with contextlib.redirect_stdout(io.StringIO()):
    df, st, _t, D, C, recv = load_network(NET)
SCALE = {s: float(df[s].mean()) for s in st if np.isfinite(df[s].mean())}
CONN = set(connected(st, C, df))


def gauge(ps, keep=None):
    """Per-gauge metrics averaged with equal weight; see the paper's Section 4.3."""
    rows = [(s, v) for s, v in ps.items()
            if v and SCALE.get(s) and (keep is None or s in keep)]
    if not rows:
        return None
    return {'RMSE': np.mean([100 * v['RMSE'] / SCALE[s] for s, v in rows]),
            'MAE': np.mean([100 * v['MAE'] / SCALE[s] for s, v in rows]),
            'NSE': np.median([v['NSE'] for s, v in rows]),
            'KGE': np.median([v['KGE'] for s, v in rows]),
            'PBIAS': np.mean([v['PBIAS'] for s, v in rows])}


def scope(r, name):
    if name == 'conn':
        return gauge(r['per_station'], CONN)
    if name == 'all':
        return gauge(r['per_station'])
    if name == 'recv':
        return gauge(r['per_station'], set(recv))
    return r['pooled']                                   # absolute cumec, variance-weighted


def cell(arm, gap, sc, metric):
    vs = [scope(R[(arm, s, gap)], sc)[metric] for s in SEEDS if (arm, s, gap) in R]
    return float(np.mean(vs)) if vs else None


def metric_table(sc, label, caption, tag):
    """One block per gap length, one row per method, five metrics."""
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         f'\\caption{{{caption}}}', f'\\label{{tab:{tag}}}',
         '\\begin{tabular}{l rrrrr}', '\\hline',
         'Method & nRMSE (\\%) & nMAE (\\%) & NSE & KGE & PBIAS (\\%) \\\\', '\\hline']
    for gap in GAPS:
        L.append(f'\\multicolumn{{6}}{{l}}{{\\textit{{Gap length {gap} days}}}} \\\\')
        best = {m: min((cell(a, gap, sc, m) for _, a in ROWS if cell(a, gap, sc, m) is not None),
                       default=None) for m in ('RMSE', 'MAE')}
        for name, arm in ROWS:
            v = {m: cell(arm, gap, sc, m) for m in ('RMSE', 'MAE', 'NSE', 'KGE', 'PBIAS')}
            if v['RMSE'] is None:
                continue
            def f(m, fmt, bold=False):
                x = f'{v[m]:{fmt}}'
                return f'\\textbf{{{x}}}' if bold else x
            L.append(f"{name} & {f('RMSE', '.1f', v['RMSE'] == best['RMSE'])} & "
                     f"{f('MAE', '.1f', v['MAE'] == best['MAE'])} & "
                     f"{v['NSE']:.3f} & {v['KGE']:.3f} & {v['PBIAS']:+.1f} \\\\")
        L.append('\\hline')
    L += ['\\end{tabular}', '\\end{table}', '']
    return L


def contrast(a, b, sc):
    """Per-cell % RMSE difference of a vs b, with a t-test on the per-gap means."""
    per_gap, d = {}, []
    for g in GAPS:
        for s in SEEDS:
            if (a, s, g) not in R or (b, s, g) not in R:
                continue
            xa, xb = scope(R[(a, s, g)], sc)['RMSE'], scope(R[(b, s, g)], sc)['RMSE']
            x = 100 * (xa - xb) / xb
            d.append(x); per_gap.setdefault(g, []).append(x)
    if not d:
        return None
    gm = [np.mean(v) for v in per_gap.values()]
    t, p = stats.ttest_1samp(gm, 0)
    return dict(mean=np.mean(d), wins=sum(x < 0 for x in d), n=len(d), t=t, p=p,
                per_gap={g: np.mean(v) for g, v in per_gap.items()})


def metric_contrast(a, b, sc, metric):
    """As contrast(), on any metric. PBIAS is compared in absolute value: the question
    is how far from zero the bias sits, not which way it leans."""
    per_gap, d = {}, []
    for g in GAPS:
        for sd in SEEDS:
            if (a, sd, g) not in R or (b, sd, g) not in R:
                continue
            xa, xb = scope(R[(a, sd, g)], sc)[metric], scope(R[(b, sd, g)], sc)[metric]
            if metric == 'PBIAS':
                xa, xb = abs(xa), abs(xb)
            if xb == 0:
                continue
            x = 100 * (xa - xb) / abs(xb)
            d.append(x); per_gap.setdefault(g, []).append(x)
    if not d:
        return None
    t, p = stats.ttest_1samp([np.mean(v) for v in per_gap.values()], 0)
    return dict(mean=np.mean(d), wins=sum(x < 0 for x in d), n=len(d), t=t, p=p)


def routing_table():
    """Per-gauge time-shift contrast: the routed feature against the same feature
    computed at c=350, where every lag rounds to zero.

    The column count, the upstream subset and the weights are identical on both
    sides, so the contrast isolates the time shift. The discriminating prediction is
    per-gauge: a gauge benefits only while at least one of its contributors is far
    enough away for round(D/c) to be non-zero, so rajim (one contributor at 6.8 km)
    should never benefit and simga (63.3 km) should stop benefiting above c=100.
    """
    CS = [25, 50, 75, 100, 150, 200]
    CEL = {}
    for d in (RESDIR, 'res_celerity'):
        for f in glob.glob(os.path.join(HERE, d, '*.json')):
            r = json.load(open(f))
            if r['net'] == NET and r['arm'] in ('R', 'AR'):
                CEL[(r['arm'], int(r['celerity']), r['seed'], r['gap'])] = r
    if not CEL:
        return []
    # A gauge is "live" at c when at least one of its contributors has round(D/c) >= 1.
    live = {}
    for g_ in recv:
        ds = [D.loc[g_, j] for j in st if j != g_ and C.loc[g_, j] > 0]
        live[g_] = {c: any(round(d / c) >= 1 for d in ds) for c in CS}
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{The time shift isolated per gauge. Each entry is the median over '
         'five seeds of the percentage change in that gauge\'s RMSE from the $c=350$ '
         'control, at which every travel-time lag rounds to zero, to the stated '
         'celerity; negative means the time shift reduced error. Both sides carry the '
         'routed column, the same upstream subset and the same weights, so the column '
         'count is fixed and only the day each upstream series is read differs. A '
         'dagger marks a celerity at which that gauge has at least one contributor with '
         '$\\tau\\geq1$~day. Rajim\'s single contributor lies 6.8~km away and Simga\'s '
         'nearest at 33.6~km, so the mechanism names in advance which cells should show '
         'no effect.}',
         '\\label{tab:routing}',
         '\\begin{tabular}{ll rrrrrr}', '\\hline',
         'Arm & Gauge & ' + ' & '.join(f'$c={c}$' for c in CS) + ' \\\\', '\\hline']
    for arm in ('R', 'AR'):
        for gi, g_ in enumerate(sorted(recv, key=lambda x: -sum(live[x].values()))):
            out = [arm if gi == 0 else '', g_]
            for c in CS:
                d = []
                for sd in SEEDS:
                    for gp in (7, 100):
                        a, b = CEL.get((arm, c, sd, gp)), CEL.get((arm, 350, sd, gp))
                        if not (a and b):
                            continue
                        pa, pb = a['per_station'].get(g_), b['per_station'].get(g_)
                        if pa and pb and pb['RMSE'] > 0:
                            d.append(100 * (pa['RMSE'] - pb['RMSE']) / pb['RMSE'])
                mark = '$^\\dagger$' if live[g_][c] else ''
                out.append(f'{np.median(d):+.1f}{mark}' if d else '--')
            L.append(' & '.join(out) + ' \\\\')
        L.append('\\hline')
    L += ['\\end{tabular}', '\\end{table}', '']
    return L


def bias_table():
    """The anchor block against the published neighbour block, on three metrics.

    The naive block reads shift(+-1) of the target's own column, which inside a gap is
    this imputer's own output from the previous round, so its errors compound and the
    interior of a long gap drifts high. That drift is a first-moment effect: it shows
    in PBIAS and in MAE, and is largely masked in RMSE, whose squared weighting is
    carried by the monsoon peaks at the gap edges.
    """
    pairs = [('\\textbf{PIMF} vs PIMF$_{\\mathrm{naive}}$', 'ARLanc', 'ARL'),
             ('\\quad $\\{N\\}$ vs $\\{L\\}$', 'Lanc', 'L'),
             ('\\quad $\\{A,N\\}$ vs $\\{A,L\\}$', 'ALanc', 'AL'),
             ('\\quad $\\{R,N\\}$ vs $\\{R,L\\}$', 'RLanc', 'RL'),
             ('\\textbf{PIMF} vs vanilla', 'ARLanc', 'vanilla'),
             ('PIMF$_{\\mathrm{naive}}$ vs vanilla', 'ARL', 'vanilla')]
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{The anchor block is a bias correction. Paired contrasts on the '
         f'primary scope over the {len(SEEDS) * len(GAPS)} seed $\\times$ gap cells, in '
         'normalised RMSE, normalised MAE and absolute PBIAS; negative favours the first '
         'arm and ``wins\'\' counts cells in which it is lower. The four indented rows are '
         'the same substitution of $N$ for $L$ made in each subset of the remaining '
         'features, so the effect is not an artefact of one particular arm. $t$ and $p$ '
         'are computed on the four per-gap-length means and carry three degrees of '
         'freedom.}',
         '\\label{tab:bias}',
         '\\begin{tabular}{l rr rr rr}', '\\hline',
         '& \\multicolumn{2}{c}{$\\Delta$nRMSE} & \\multicolumn{2}{c}{$\\Delta$nMAE}'
         ' & \\multicolumn{2}{c}{$\\Delta|$PBIAS$|$} \\\\',
         'Contrast & \\% & $p$ & \\% & $p$ & \\% & $p$ \\\\', '\\hline']
    for name, a, b in pairs:
        cs = [metric_contrast(a, b, 'conn', m) for m in ('RMSE', 'MAE', 'PBIAS')]
        if any(c is None for c in cs):
            continue
        L.append(name + ' & ' + ' & '.join(
            f"{c['mean']:+.1f} & {c['p']:.3f}" for c in cs) + ' \\\\')
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


def contrast_table():
    pairs = [('\\textbf{PIMF} vs vanilla', 'ARLanc', 'vanilla'),
             ('PIMF-L $\\{A,R\\}$ vs vanilla', 'AR', 'vanilla'),
             ('PIMF$_{\\mathrm{naive}}$ vs vanilla', 'ARL', 'vanilla'),
             ('$\\varnothing$ base vs vanilla', 'base', 'vanilla'),
             ('PIMF vs PIMF$_{\\mathrm{naive}}$', 'ARLanc', 'ARL'),
             ('PIMF vs PIMF-L', 'ARLanc', 'AR'),
             ('placebo routing vs real', 'ARplac', 'AR')]
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{Paired contrasts on the primary scope (the '
         f'{len(CONN)} gauges of the connectivity graph, Section 4.3). $\\Delta$RMSE is the '
         f'mean over the {len(SEEDS) * len(GAPS)} seed $\\times$ gap cells of the per-cell '
         'percentage difference in normalised RMSE; negative favours the first arm. '
         '``Wins\'\' counts cells in which the first arm has the lower error. Because the '
         'effect varies with gap length, $t$ and $p$ are computed on the four per-gap-length '
         'means and carry three degrees of freedom.}',
         '\\label{tab:contrasts}',
         '\\begin{tabular}{l r r r r rrrr}', '\\hline',
         'Contrast & $\\Delta$RMSE & Wins & $t$ & $p$ & 3 d & 7 d & 30 d & 100 d \\\\',
         '\\hline']
    for name, a, b in pairs:
        c = contrast(a, b, 'conn')
        if not c:
            continue
        pg = ' & '.join(f"{c['per_gap'][g]:+.1f}" for g in GAPS)
        L.append(f"{name} & {c['mean']:+.2f}\\% & {c['wins']}/{c['n']} & {c['t']:+.2f} & "
                 f"{c['p']:.4f} & {pg} \\\\")
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


def robustness_table():
    scopes = [('conn', f'Gauge-normalised, all {len(CONN)} gauges (primary)'),
              ('recv', f'Receivers only ({len(recv)})'),
              ('pooled', 'Pooled residuals (cumec, variance-weighted)')]
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{Scope robustness of the headline contrast. PIMF against vanilla '
         'MissForest under each reporting scope. Every gauge in the network is scored, '
         'so the primary scope is the whole network; the receiver rows restrict to the '
         'gauges that have an upstream contributor and therefore a routed term at all. '
         'The pooled row weights each residual equally rather than each gauge, so it is '
         'dominated by the largest gauge; it is shown to demonstrate that the conclusion '
         'does not depend on the choice.}',
         '\\label{tab:scopes}',
         '\\begin{tabular}{l r r r r}', '\\hline',
         'Scope & $\\Delta$RMSE vs vanilla & Wins & $t$ & $p$ \\\\', '\\hline']
    for sc, name in scopes:
        c = contrast('ARLanc', 'vanilla', sc)
        if c:
            L.append(f"{name} & {c['mean']:+.2f}\\% & {c['wins']}/{c['n']} & "
                     f"{c['t']:+.2f} & {c['p']:.4f} \\\\")
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


def cross_network_table():
    """One row per network: the headline contrast, plus the arms that isolate why.

    Rebuilt per network because the arm names differ: kaveri has no connectivity
    matrix, so R does not exist there and the PIMF arm is {A,N} rather than {A,R,N}.
    """
    nets = [('mahanadi', 'Mahanadi (12 gauges, 5 receivers)', 'ARLanc', 'ARL', 'AR'),
            ('kaveri', 'Kaveri (12 gauges, no connectivity matrix)', 'ALanc', 'AL', 'A'),
            ('us', 'US (6 gauges, 3 receivers, all lags 0 d)', 'ARLanc', 'ARL', 'AR')]
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{PIMF against conventional MissForest on all three networks, on each '
         'network\'s primary scope (Section 3.7.1). PIMF is $\\{A,R,N\\}$ where a connectivity '
         'matrix exists and $\\{A,N\\}$ on the Kaveri, which has none. The last two columns '
         'isolate the temporal block: PIMF$_{\\mathrm{naive}}$ carries the superseded '
         'neighbour formulation, and PIMF-L carries no temporal block at all. Each entry is '
         'the mean over the 20 seed $\\times$ gap cells of the per-cell percentage difference '
         'in normalised RMSE, with wins and the blocked $p$ beneath.}',
         '\\label{tab:networks}',
         '\\begin{tabular}{l rrr}', '\\hline',
         'Network & PIMF vs vanilla & PIMF$_{\\mathrm{naive}}$ vs vanilla & PIMF vs PIMF-L \\\\',
         '\\hline']
    import glob as _g
    for net, name, pimf, naive, nol in nets:
        Rn = {}
        for f in _g.glob(os.path.join(HERE, RESDIR, '*.json')):
            r = json.load(open(f))
            if r['net'] == net:
                Rn[(r['arm'], r['seed'], r['gap'])] = r
        if not Rn:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dfn, stn, _tn, _Dn, Cn, _rn = load_network(net)
        sc = {s: float(dfn[s].mean()) for s in stn if np.isfinite(dfn[s].mean())}
        cn = set(connected(stn, Cn, dfn))
        seeds = sorted({k[1] for k in Rn})

        def g_(ps):
            rows = [(s, v) for s, v in ps.items() if v and sc.get(s) and s in cn]
            return np.mean([100 * v['RMSE'] / sc[s] for s, v in rows]) if rows else None

        def con(a, b):
            pg = {}
            for g in GAPS:
                vals = [100 * (g_(Rn[(a, s, g)]['per_station']) - g_(Rn[(b, s, g)]['per_station']))
                        / g_(Rn[(b, s, g)]['per_station'])
                        for s in seeds if (a, s, g) in Rn and (b, s, g) in Rn]
                if vals:
                    pg[g] = vals
            if not pg:
                return None
            d = [x for v in pg.values() for x in v]
            t, pv = stats.ttest_1samp([np.mean(v) for v in pg.values()], 0)
            return np.mean(d), sum(x < 0 for x in d), len(d), pv

        cells = []
        for a, b in ((pimf, 'vanilla'), (naive, 'vanilla'), (pimf, nol)):
            c = con(a, b)
            cells.append('n/a' if c is None else
                         f'{c[0]:+.2f}\\% ({c[1]}/{c[2]}, $p$={c[3]:.3f})')
        L.append(f"{name} & " + ' & '.join(cells) + ' \\\\')
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


FACT_ROWS = [('Vanilla MissForest', 'vanilla'), ('$\\{A\\}$', 'A'), ('$\\{R\\}$', 'R'),
             ('$\\{L\\}$', 'L'), ('$\\{N\\}$', 'Lanc'), ('$\\{A,R\\}$ = PIMF-L', 'AR'),
             ('$\\{A,L\\}$', 'AL'), ('$\\{R,L\\}$', 'RL'), ('$\\{A,N\\}$', 'ALanc'),
             ('$\\{R,N\\}$', 'RLanc'), ('$\\{A,R,L\\}$ = PIMF$_{\\mathrm{naive}}$', 'ARL'),
             ('$\\{A,R,N\\}$ = \\textbf{PIMF}', 'ARLanc'),
             ('$\\{R\\}$, placebo routing', 'Rplac'),
             ('$\\{A,R\\}$, placebo routing', 'ARplac')]

MARGINALS = {'A': [('A', 'base'), ('AR', 'R'), ('AL', 'L'), ('ARL', 'RL')],
             'R': [('R', 'base'), ('AR', 'A'), ('RL', 'L'), ('ARL', 'AL')],
             'L': [('L', 'base'), ('AL', 'A'), ('RL', 'R'), ('ARL', 'AR')],
             'N': [('Lanc', 'base'), ('ALanc', 'A'), ('RLanc', 'R'), ('ARLanc', 'AR')]}


def factorial_table():
    """Every arm against the ablation base, then each feature's main effect."""
    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{Factorial ablation of the three engineered feature groups on the primary '
         'scope: $A$ = weighted basin discharge (Eq. (11)), $R$ = travel-time routed upstream '
         'discharge (Eq. (13)), and the temporal block in either its superseded form $L$ '
         '(Eq. (14) read from the working frame) or as the anchors $N$ of Eqs. (14)--(15). '
         'Each arm differs from the ablation base $\\varnothing$ only in the features named; '
         'initialization, ordering, seasonal terms, schedule and forest hyperparameters are '
         'identical throughout. Upper block: each arm against $\\varnothing$ over the 20 seed '
         '$\\times$ gap cells. Lower block: each feature\'s main effect, the mean of its four '
         'paired contrasts over the subsets of the other two, which is what the factorial buys '
         'over a one-at-a-time ablation. $p$ is a paired $t$-test on the four per-gap-length '
         'means and carries three degrees of freedom.}',
         '\\label{tab:factorial}',
         '\\begin{tabular}{l r r r rrrr}', '\\hline',
         'Arm & $\\Delta$RMSE & Wins & $p$ & 3 d & 7 d & 30 d & 100 d \\\\', '\\hline']

    def pair(a, b):
        pg = {}
        for g in GAPS:
            v = [100 * (scope(R[(a, s, g)], 'conn')['RMSE'] - scope(R[(b, s, g)], 'conn')['RMSE'])
                 / scope(R[(b, s, g)], 'conn')['RMSE'] for s in SEEDS
                 if (a, s, g) in R and (b, s, g) in R]
            if v:
                pg[g] = v
        if not pg:
            return None
        d = [x for v in pg.values() for x in v]
        t, pv = stats.ttest_1samp([np.mean(v) for v in pg.values()], 0)
        return np.mean(d), sum(x < 0 for x in d), len(d), pv, {g: np.mean(v) for g, v in pg.items()}

    for name, arm in FACT_ROWS:
        c = pair(arm, 'base')
        if not c:
            continue
        pg = ' & '.join(f'{c[4][g]:+.1f}' if g in c[4] else '--' for g in GAPS)
        L.append(f'{name} & {c[0]:+.2f}\\% & {c[1]}/{c[2]} & {c[3]:.4f} & {pg} \\\\')
    L += ['\\hline', '\\multicolumn{8}{l}{\\textit{Main effect of each feature}} \\\\']
    for feat, pairs in MARGINALS.items():
        d, pg = [], {}
        for on, off in pairs:
            for g in GAPS:
                for sd in SEEDS:
                    if (on, sd, g) not in R or (off, sd, g) not in R:
                        continue
                    x = (100 * (scope(R[(on, sd, g)], 'conn')['RMSE']
                                - scope(R[(off, sd, g)], 'conn')['RMSE'])
                         / scope(R[(off, sd, g)], 'conn')['RMSE'])
                    d.append(x); pg.setdefault(g, []).append(x)
        if not d:
            continue
        t, pv = stats.ttest_1samp([np.mean(v) for v in pg.values()], 0)
        row = ' & '.join(f'{np.mean(pg[g]):+.1f}' for g in GAPS)
        L.append(f'${feat}$ vs no-${feat}$ & {np.mean(d):+.2f}\\% & '
                 f'{sum(x < 0 for x in d)}/{len(d)} & {pv:.4f} & {row} \\\\')
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


def celerity_table():
    """The wave-celerity sweep on the {R} arm at the receivers."""
    CS = [25, 50, 75, 100, 150, 200, 350]
    INTERIOR = [50, 75, 100, 150, 200]
    CEL = {}
    for d in (RESDIR, 'res_celerity'):
        for f in glob.glob(os.path.join(HERE, d, '*.json')):
            r = json.load(open(f))
            if r['net'] == NET and r['arm'] == 'R':
                CEL[(int(r['celerity']), r['seed'], r['gap'])] = r
    if not CEL:
        return []
    rec = set(recv)

    def val(c, sd, g):
        r = CEL.get((c, sd, g))
        return None if r is None else gauge(r['per_station'], rec)['RMSE']

    L = ['\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\caption{Wave-celerity sweep on the $\\{R\\}$ arm at the five receiver gauges. '
         '$\\{R\\}$ is used because the routed predictor is then the only derived feature, so '
         'nothing else can supply temporal information. At $c=350$~km\\,day\\textsuperscript{-1} '
         'every one of the 28 upstream lags rounds to zero, so that column holds the column '
         'count, the upstream subset and the weights fixed and removes only the time shift; '
         'the contrast between the interior celerities and $c=350$ therefore isolates the '
         'shift itself. Entries are normalised RMSE averaged over the five receivers, mean of '
         'five seeds. $p$ is a paired $t$-test against $c=100$ across seeds, Holm-corrected '
         'over the six comparisons.}',
         '\\label{tab:celerity}',
         '\\begin{tabular}{r rrr rrr}', '\\hline',
         '& \\multicolumn{3}{c}{Gap 7 d} & \\multicolumn{3}{c}{Gap 100 d} \\\\',
         '$c$ (km\\,day$^{-1}$) & nRMSE & vs $c{=}100$ & Holm $p$ '
         '& nRMSE & vs $c{=}100$ & Holm $p$ \\\\', '\\hline']
    cells = {}
    for g in (7, 100):
        rows = {c: [val(c, sd, g) for sd in SEEDS] for c in CS}
        rows = {c: v for c, v in rows.items() if all(x is not None for x in v)}
        cand = [c for c in rows if c != 100]
        raw = {c: stats.ttest_1samp([100 * (rows[c][i] - rows[100][i]) / rows[100][i]
                                     for i in range(len(SEEDS))], 0)[1] for c in cand}
        adj = dict(zip(cand, _holm([raw[c] for c in cand])))
        cells[g] = (rows, adj)
    for c in CS:
        out = [f'{c}']
        for g in (7, 100):
            rows, adj = cells[g]
            if c not in rows:
                out += ['--', '--', '--']; continue
            m = np.mean(rows[c])
            if c == 100:
                out += [f'{m:.1f}', 'ref', '--']
            else:
                dd = np.mean([100 * (rows[c][i] - rows[100][i]) / rows[100][i]
                              for i in range(len(SEEDS))])
                out += [f'{m:.1f}', f'{dd:+.1f}\\%', f'{adj[c]:.3f}']
        L.append(' & '.join(out) + ' \\\\')
    L.append('\\hline')
    for g in (7, 100):
        rows, _ = cells[g]
        inter = [c for c in INTERIOR if c in rows]
        if 350 not in rows or not inter:
            continue
        d = [100 * (np.mean([rows[c][i] for c in inter]) - rows[350][i]) / rows[350][i]
             for i in range(len(SEEDS))]
        t, pv = stats.ttest_1samp(d, 0)
        L.append(f'\\multicolumn{{7}}{{l}}{{Time shift, gap {g} d: mean of '
                 f'$c\\in\\{{50,75,100,150,200\\}}$ against $c=350$ is '
                 f'{np.mean(d):+.2f}\\%, better in {sum(x < 0 for x in d)}/{len(d)} seeds, '
                 f'$t={t:+.2f}$, $p={pv:.4f}$}} \\\\')
    L += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return L


miss = {g: np.mean([r['achieved_missing_pct'] for k, r in R.items() if k[2] == g])
        for g in GAPS}
head = [f'% GENERATED by rerun_expts/make_tables.py from {RESDIR}/ -- do not edit by hand.',
        f'% net={NET} protocol={"+".join(PROTO)} seeds={SEEDS} gaps={GAPS}',
        '% achieved missingness: '
        + ', '.join(f'{g} d {v:.2f}%' for g, v in miss.items()), '']

body = head
body += metric_table(
    'conn', 'primary',
    'Imputation accuracy on the primary scope: the '
    f'{len(CONN)} Mahanadi gauges that take part in the connectivity graph '
    '(Section 4.3). Each entry is the mean over '
    f'{len(SEEDS)} gap seeds under the transductive protocol of Section 4.2. nRMSE and '
    'nMAE are each gauge\'s error as a percentage of its own mean discharge, averaged '
    'over gauges with equal weight; NSE and KGE are medians over gauges. Bold marks the '
    'best value in each block. PIMF is the full feature set $\\{A,R,N\\}$; PIMF-L omits '
    'the temporal block; PIMF$_{\\mathrm{naive}}$ carries the superseded block of '
    'Eq. (14). $\\varnothing$ carries the seasonal terms, historical-mean initialization '
    'and station ordering but none of the engineered features. PIMF, in bold, is the '
    'method; every other PIMF row is an ablation or a variant reported for robustness '
    'and is not proposed.', '4')
body += metric_table(
    'recv', 'recv',
    f'As Table~\\ref{{tab:4}}, restricted to the {len(recv)} receiver gauges '
    '($C_{i\\cdot}\\neq\\mathbf{0}$), the only gauges for which the routed predictor of '
    'Eq. (13) is defined.', '5')
body += contrast_table()
body += bias_table()
body += robustness_table()
body += routing_table()
body += cross_network_table()
body += factorial_table()
cel = celerity_table()
body += cel if len(cel) > 12 else [
    '% celerity sweep absent from res_celerity/ -- re-run drive_ablation.py with',
    '% --celerities 25,50,75,150,200,350 --out res_celerity, then regenerate.', '']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(body))
print(f'wrote {OUT}: {len(body)} lines, {len(R)} cells, '
      f'{len(CONN)} connected gauges, {len(recv)} receivers, seeds {SEEDS}')
print('achieved missingness: ' + ', '.join(f'{g}d {v:.2f}%' for g, v in miss.items()))
