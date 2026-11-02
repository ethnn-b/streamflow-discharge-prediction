import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
PY = sys.executable
SEEDS, GAPS = [42, 1, 2], [3, 7, 30, 100]

jobs = []
for arm in ['pimf', 'vanilla_single', 'vanilla_single_tf', 'vanilla_seedfit']:
    for g in GAPS:
        for s in SEEDS:
            jobs.append((arm, s, g, 100.0))
for c in [25.0, 50.0, 75.0, 150.0, 200.0, 350.0]:
    for g in [7, 100]:
        for s in SEEDS:
            jobs.append(('pimf', s, g, c))


def run(j):
    arm, s, g, c = j
    out = os.path.join(HERE, 'res', f'{arm}_{s}_{g}_{int(c)}.json')
    if os.path.exists(out):
        return f'skip {out}'
    r = subprocess.run([PY, os.path.join(HERE, 'run_net.py'), '--root', ROOT, '--net', 'mahanadi',
                        '--arm', arm, '--seed', str(s), '--gap', str(g), '--celerity', str(c), '--out', out],
                       capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [''])[-1]
    msg = line if r.returncode == 0 else f'FAIL {j}: {r.stderr[-400:]}'
    print(msg, flush=True)
    return msg


with ThreadPoolExecutor(6) as ex:
    list(ex.map(run, jobs))
print('ALL DONE', flush=True)
