"""Celerity sweep under the fold protocol of run_fold.py.

The question is whether the *precision* of the travel-time lag matters, not
whether an upstream column helps. At c = 350 km/day every tau_ij = round(D_ij/c)
rounds to zero over all 28 Mahanadi upstream pairs, so the routed feature of
Eq. (13) degenerates to a same-day weighted sum over the same upstream subset:
same column count, same weights, no time shift. The contrast between the
interior celerities and c = 350 therefore isolates the shift itself.

Arms swept
  R   routing is the ONLY derived feature, so nothing else can supply temporal
      information. This is the cleanest test.
  AR  PIMF-L: routing plus the weighted aggregate.
L is deliberately excluded. It supplies its own short-range temporal signal and
masked the celerity effect in the previous draft's {R,L} sweep; adding it here
would only reproduce that null.

c = 100 is not re-run: those cells already exist in res_ablation/.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_ = sys.executable
N_FOLDS, SEEDS, GAPS = 3, [42, 1, 2, 3, 4], [7, 100]
CELERITIES = [25, 50, 75, 150, 200, 350]        # 100 comes from res_ablation
ARMS = ['R', 'AR']

jobs = [(arm, c, g, s, f) for arm in ARMS for c in CELERITIES
        for g in GAPS for s in SEEDS for f in range(N_FOLDS)]


def run(j):
    arm, c, g, s, f = j
    out = os.path.join(HERE, 'res_celerity', f'mahanadi_{arm}_f{f}_s{s}_{g}_c{c}.json')
    if os.path.exists(out):
        return
    r = subprocess.run([PY_, os.path.join(HERE, 'run_fold.py'), '--root', ROOT,
                        '--net', 'mahanadi', '--arm', arm, '--fold', str(f), '--seed', str(s),
                        '--gap', str(g), '--n-folds', str(N_FOLDS), '--celerity', str(float(c)),
                        '--threads', '1', '--out', out], capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [f'FAIL {j}: {r.stderr[-400:]}'])[-1]
    print(f'c={c:<4} {line}', flush=True)


print(f'{len(jobs)} cells: {ARMS} x {CELERITIES} x {GAPS}d x {len(SEEDS)} seeds x {N_FOLDS} folds',
      flush=True)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(run, jobs))
print('CELERITY SWEEP DONE', flush=True)
