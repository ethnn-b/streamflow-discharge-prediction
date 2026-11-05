"""Feature ablation under the fold protocol of run_fold.py.

Every arm sees the same folds, gaps and seeds, and differs only in which of
A / R / L it carries, so the differences are attributable to those features.
Skips cells whose JSON already exists, so a run can be resumed.
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ = sys.executable
FACTORIAL = ['base', 'A', 'R', 'L', 'AR', 'AL', 'RL', 'ARL']
# Kaveri has no connectivity matrix, so R is undefined there and the arms
# carrying it would silently duplicate their R-free counterparts.
CLASSICAL = ['vanilla', 'linear', 'histmean', 'kalman', 'knn']
# The L block rebuilt as anchors to the nearest observed reading (run_fold.decode).
ANCHOR = ['Lanc', 'ALanc', 'RLanc', 'ARLanc']
# Routing against a distance-, weight- and lag-matched NON-upstream set. Only on
# mahanadi: the US receivers have a single contributor each, so their routed term is
# already a rescaled copy of a raw column and there is nothing for a control to test.
PLACEBO = ['Rplac', 'ARplac']
ARMS = {'mahanadi': FACTORIAL + CLASSICAL + ANCHOR + PLACEBO,
        'us': FACTORIAL + CLASSICAL + ANCHOR,
        'kaveri': ['base', 'A', 'L', 'AL'] + CLASSICAL + ['Lanc', 'ALanc']}

p = argparse.ArgumentParser()
p.add_argument('nets', nargs='*', default=['mahanadi'])
p.add_argument('--protocol', default='fold', choices=['fold', 'transductive'])
p.add_argument('--arms', help='comma-separated; default is the full set for each net')
p.add_argument('--gaps', default='3,7,30,100')
p.add_argument('--seeds', default='42,1,2,3,4')
p.add_argument('--n-folds', type=int, default=3)
p.add_argument('--alphas', help='comma-separated distance/connectivity blend sweep; alpha '
                                "lands in the cell filename. Only the 5 receivers' weights move: "
                                'for a gauge with an all-zero connectivity row the blend reduces '
                                'to alpha*distance and renormalisation cancels alpha, so alpha is '
                                'inert there for any alpha > 0 and collapses the weights to zero '
                                'at alpha = 0.')
p.add_argument('--celerities', help='comma-separated wave-celerity sweep (km/day); the value '
                                    'lands in the cell filename. At c=350 all 28 Mahanadi '
                                    'upstream lags round to zero, which is the no-time-shift '
                                    'control holding columns and weights fixed.')
p.add_argument('--out', default='res_ablation', help='results folder under rerun_expts/')
p.add_argument('--workers', type=int, default=16)
a = p.parse_args()

nets = a.nets or ['mahanadi']
GAPS = [int(x) for x in a.gaps.split(',')]
SEEDS = [int(x) for x in a.seeds.split(',')]
# Transductive fits the whole record, so there is no block to hold out and one cell
# per (arm, seed, gap) is the whole replicate set.
FOLDS = [0] if a.protocol == 'transductive' else list(range(a.n_folds))
ALPHAS = [float(x) for x in a.alphas.split(',')] if a.alphas else [None]
CELS = [float(x) for x in a.celerities.split(',')] if a.celerities else [None]
jobs = [(net, arm, f, s, g, al, c)
        for net in nets for arm in (a.arms.split(',') if a.arms else ARMS[net])
        for g in GAPS for s in SEEDS for f in FOLDS for al in ALPHAS for c in CELS]


def run(j):
    net, arm, f, s, g, al, c = j
    tag = ('' if al is None else f'_a{al:g}') + ('' if c is None else f'_c{c:g}')
    out = os.path.join(HERE, a.out, f'{net}_{arm}_f{f}_s{s}_{g}{tag}.json')
    if os.path.exists(out):
        return
    cmd = [PY_, os.path.join(HERE, 'run_fold.py'), '--root', ROOT, '--net', net,
           '--arm', arm, '--fold', str(f), '--seed', str(s), '--gap', str(g),
           '--n-folds', str(a.n_folds), '--threads', '1', '--protocol', a.protocol,
           '--out', out]
    if al is not None:
        cmd += ['--alpha', str(al)]
    if c is not None:
        cmd += ['--celerity', str(c)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    pre = ('' if al is None else f'a={al:<4g} ') + ('' if c is None else f'c={c:<4g} ')
    print(pre + (r.stdout.strip().splitlines() or [f'FAIL {j}: {r.stderr[-400:]}'])[-1], flush=True)


print(f'{len(jobs)} cells over {nets} [{a.protocol}] -> {a.out}', flush=True)
# one single-threaded cell per core beats n_jobs=-1 per cell; see run_fold.py
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(run, jobs))
print('ABLATION DONE', flush=True)
