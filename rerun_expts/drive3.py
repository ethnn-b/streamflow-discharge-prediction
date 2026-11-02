import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
PY = sys.executable
SEEDS, GAPS = [42, 1, 2], [3, 7, 30, 100]

jobs = []
for net in ['us']:
    for s in SEEDS:
        jobs.append((net, 'pimfL', s, 100))
        for g in GAPS:
            for arm in ['pimf', 'vanilla_single', 'linear']:
                jobs.append((net, arm, s, g))


def run(j):
    net, arm, s, g = j
    out = os.path.join(HERE, 'res3', f'{net}_{arm}_{s}_{g}.json')
    if os.path.exists(out):
        return f'skip {out}'
    r = subprocess.run([PY, os.path.join(HERE, 'run_net.py'), '--root', ROOT, '--net', net, '--arm', arm,
                        '--seed', str(s), '--gap', str(g), '--out', out],
                       capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [''])[-1]
    msg = line if r.returncode == 0 else f'FAIL {j}: {r.stderr[-400:]}'
    print(msg, flush=True)
    return msg


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, jobs))
print('ALL DONE', flush=True)
