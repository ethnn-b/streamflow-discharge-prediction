"""Feature ablation under the fold protocol of run_fold.py.

Every arm sees the same folds, gaps and seeds, and differs only in which of
A / R / L it carries, so the differences are attributable to those features.
Skips cells whose JSON already exists, so a run can be resumed.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ = sys.executable
N_FOLDS = 3
SEEDS = [42, 1, 2, 3, 4]
GAPS = [3, 7, 30, 100]
FACTORIAL = ['base', 'A', 'R', 'L', 'AR', 'AL', 'RL', 'ARL']
# Kaveri has no connectivity matrix, so R is undefined there and the arms
# carrying it would silently duplicate their R-free counterparts.
CLASSICAL = ['vanilla', 'linear', 'histmean', 'kalman', 'knn']
ARMS = {'mahanadi': FACTORIAL + CLASSICAL,
        'us': FACTORIAL + CLASSICAL,
        'kaveri': ['base', 'A', 'L', 'AL'] + CLASSICAL}

nets = sys.argv[1:] or ['mahanadi']
jobs = [(net, arm, f, s, g)
        for net in nets for arm in ARMS[net]
        for g in GAPS for s in SEEDS for f in range(N_FOLDS)]


def run(j):
    net, arm, f, s, g = j
    out = os.path.join(HERE, 'res_ablation', f'{net}_{arm}_f{f}_s{s}_{g}.json')
    if os.path.exists(out):
        return
    r = subprocess.run([PY_, os.path.join(HERE, 'run_fold.py'), '--root', ROOT, '--net', net,
                        '--arm', arm, '--fold', str(f), '--seed', str(s), '--gap', str(g),
                        '--n-folds', str(N_FOLDS), '--threads', '1',
                        '--out', out], capture_output=True, text=True)
    print((r.stdout.strip().splitlines() or [f'FAIL {j}: {r.stderr[-400:]}'])[-1], flush=True)


print(f'{len(jobs)} cells over {nets}', flush=True)
# one single-threaded cell per core beats n_jobs=-1 per cell; see run_fold.py
with ThreadPoolExecutor(16) as ex:
    list(ex.map(run, jobs))
print('ABLATION DONE', flush=True)
