"""Qualitative-validation figures, generated from the same code path as the tables.

The figures in earlier drafts came from src/experiments/station_comparison.py, which
built its own 22-gauge network and used a 3-year chaining protocol. Neither matches
the experiments the paper now reports, so a figure could disagree with the table it
sits next to. This script instead calls rerun_expts.common.load_network and fits the
one gapped record in place, which is run_fold.py's transductive protocol verbatim,
so a panel here is the same imputation that Table 2 scores.

Usage: python rerun_expts/make_figures.py --root $PWD [--net mahanadi] [--seed 42]
                                          [--stations basantpur,jondhra] [--out DIR]
"""
import argparse, contextlib, io, os, sys, warnings

ROOT = sys.argv[sys.argv.index('--root') + 1]
sys.path.insert(0, ROOT)
os.chdir(ROOT)

p = argparse.ArgumentParser()
p.add_argument('--root')
p.add_argument('--net', default='mahanadi')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--stations', default='basantpur')
p.add_argument('--gaps', default='3,7,30,100')
p.add_argument('--out', default='paper_latex_v6/figures_regen')
a = p.parse_args()

warnings.simplefilter('ignore')
import matplotlib
matplotlib.use('Agg')

from rerun_expts.common import load_network
from src.utils.data import create_contiguous_segment_gaps_by_percent
from src.imputers.missforest import MissForestImputer
from src.experiments.benchmark_1980_1990 import (
    plot_selected_gaps_subplots, plot_overall_results_subplots)

GAPS = [int(x) for x in a.gaps.split(',')]
os.makedirs(a.out, exist_ok=True)
df, stations, temporal, D, C, receivers = load_network(a.net)
print(f'{a.net}: {len(stations)} gauges, {len(receivers)} receivers, seed {a.seed}')

data_list = []
for gap in GAPS:
    with contextlib.redirect_stdout(io.StringIO()):
        # identical to run_fold.py under --protocol transductive
        gapped = df.copy()
        gapped.loc[:, stations] = create_contiguous_segment_gaps_by_percent(
            df, stations, gap, 10.0, random_seed=a.seed)[stations]
        m = MissForestImputer(distance_matrix=D, connectivity=C, max_iter=10,
                              n_estimators=100, random_state=a.seed,
                              distance_weighting_type='inverse',
                              temporal_feature_columns=temporal,
                              initialization_method='historical_mean',
                              ordering_method='most_full_first',
                              n_jobs=-1)
        m.fit(gapped)
        imp = m.transform(gapped)
    cut = gapped[stations].isna() & df[stations].notna()
    print(f'  {gap:>3}d gap: {int(cut.sum().sum())} cells withheld '
          f'({100 * cut.sum().sum() / df[stations].notna().sum().sum():.2f}%)')
    data_list.append((gap, gapped, imp))

for st in a.stations.split(','):
    if st not in stations:
        print(f'  skipping {st}: not in the {a.net} network'); continue
    print(f'>>> {st}')
    with contextlib.redirect_stdout(io.StringIO()):
        plot_selected_gaps_subplots(df, data_list, stations, a.out, f'PIMF_{st}', target_station=st)
        plot_overall_results_subplots(df, data_list, stations, a.out, f'PIMF_{st}', target_station=st)
    print(f'    zoomed_subplots_PIMF_{st}.png, overall_subplots_PIMF_{st}.png')
print(f'\nwrote {a.out}/')
