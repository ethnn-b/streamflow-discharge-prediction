"""
Robustness sweep for the gap-initialization comparison: reruns the
Column Mean / Seasonal Mean / Historical Mean comparison across three random
seeds (matching the 3-seed averaging convention used for the contribution
ablation and Kaveri benchmark) and averages the results.
"""
import warnings
import pandas as pd

from .gap_init_comparison import run_gap_init_comparison

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        all_runs = []
        for seed in [42, 7, 123]:
            print(f"\n\n{'#' * 70}\n# SEED {seed}\n{'#' * 70}")
            df = run_gap_init_comparison(random_state=seed, make_plots=False, run_tag=f'seed{seed}')
            df['random_state'] = seed
            all_runs.append(df)

        combined = pd.concat(all_runs, ignore_index=True)
        combined.to_csv('gap_init_comparison_sweep_combined.csv', index=False)

        mean_df = combined.groupby(['Gap_Length', 'Initialization'], sort=False)[['RMSE', 'MAE', 'NSE', 'KGE']].mean().reset_index()
        mean_df.to_csv('gap_init_comparison_sweep_means.csv', index=False)

        print("\n\n" + "=" * 70)
        print("PER-SEED RESULTS")
        print(combined.round(4).to_string())
        print("\nSEED-AVERAGED RESULTS")
        print(mean_df.round(4).to_string())
        print("=" * 70)
