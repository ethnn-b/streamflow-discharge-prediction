"""
Robustness sweep for the contribution ablation: reruns the with/without
hydrological-contribution comparison across multiple random seeds (to check
whether the with/without gap is real signal or RF training noise) and across
both station-ordering strategies (to check sensitivity to that implementation
choice). Aggregates everything into one CSV for comparison.
"""
import warnings
import pandas as pd

from .contribution_ablation import run_contribution_ablation

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        all_runs = []

        for seed in [7, 123]:
            print(f"\n\n{'#' * 70}\n# SEED SWEEP: random_state={seed}, ordering=most_full_first\n{'#' * 70}")
            df = run_contribution_ablation(ordering='most_full_first', random_state=seed,
                                            make_plots=False, run_tag=f'seed{seed}_ordered')
            df['random_state'] = seed
            df['ordering'] = 'most_full_first'
            all_runs.append(df)

        print(f"\n\n{'#' * 70}\n# ORDERING SWEEP: random_state=42, ordering=none\n{'#' * 70}")
        df = run_contribution_ablation(ordering='none', random_state=42,
                                        make_plots=False, run_tag='seed42_natural')
        df['random_state'] = 42
        df['ordering'] = 'none'
        all_runs.append(df)

        combined = pd.concat(all_runs, ignore_index=True)
        combined.to_csv('contribution_ablation_sweep_combined.csv', index=False)
        print("\n\n" + "=" * 70)
        print("COMBINED SWEEP RESULTS")
        print(combined.round(4).to_string())
        print("=" * 70)
