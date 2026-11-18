# Methodology for the paper runs

Decided 2026-11-03; the method was locked 2026-11-05.

The v5 audit that this folder started as -- `res/`, `res2/`, `res3/` and the scripts
that made them (`drive.py`, `drive2.py`, `drive3.py`, `run_net.py`, `summarise.py`) --
was deleted on 2026-11-05. It used the superseded seed-block protocol and the
uncorrected connectivity matrix, and nothing in the paper draws on it. Recover it from
commit 248d288 if the "vanilla was unfairly trained" finding ever needs its receipts.

## The method

**PIMF is `{A, R, N}`, the arm `ARLanc`.** Locked 2026-11-05, and it is the library
default: `MissForestImputer()` with no arguments gives `use_aggregate=True`,
`use_routing=True`, `use_temporal_lags=True`, `lag_mode='anchor'`, `anchor_bands=None`.
`tests/test_anchor_placebo.py` section 5 asserts exactly that, so the default cannot
drift away from the paper.

Chosen because it is the only arm significant on both networks where R is defined
(mahanadi -3.50%, p=0.0026; US -2.85%, p=0.0181) and negative at all four gap lengths
on both. Off the connectivity graph it reduces to `{A, N}` automatically -- R does not
exist and the blend collapses to inverse distance -- which is what runs on the Kaveri,
so `ALanc` is not a rival arm but the same method on a network with no river map.

Everything else is an ablation or a robustness variant and **is not proposed**:

| arm | what it is | why not the method |
|---|---|---|
| `ARLancd` (PIMF-D) | one forest per gap-length band | -0.55% vs `ARLanc`, p=0.47; needs ~72 seeds to separate, run on one network, adds a band-count hyperparameter |
| `AR` (PIMF-L) | no temporal block | +0.62%, p=0.33 on mahanadi; fails on the US (p=0.098) |
| `ARL` (PIMF-naive) | superseded shift(+-1) block | unstable across networks: p=0.57 mahanadi vs p=0.016 US |
| `Rplac`, `ARplac` | distance/weight/lag-matched non-upstream gauges | placebo controls |

## Files

- `common.py`: `load_network`, `connected`, `folds`. The USGS `-999999` sentinel is
  masked here (line 34), so every caller gets it.
- `run_fold.py`: one `(arm, fold, seed, gap)` cell. `--protocol fold|transductive`,
  `--alpha`, `--celerity`, `--max-features`. `decode()` maps an arm name to feature
  flags; the `anc` / `ancd` / `plac` suffixes select `lag_mode` / `anchor_bands` /
  `routing_mode`.
- `drive_ablation.py`: the parallel driver for every sweep (`--arms`, `--alphas`,
  `--celerities`, `--pct`, `--series`). It skips cells whose JSON already exists, so a
  run resumes.
- `make_tables.py`: writes `<paper_dir>/tables_main.tex` (argv[3]). The tables in
  `paper_latex_v10/tables/` are **hand-adapted copies** of its output (arm labels
  renamed, ablation rows dropped, the networks and scopes tables restructured), not
  generated files. After any sweep, regenerate and compare; the shared numbers must match.
  `headline_stats.py` recomputes the nRMSE/nMAE/|PBIAS| contrasts per network and the
  factorial main effects, including the US and Kaveri nMAE columns that `make_tables.py`
  does not emit.
- `check_invariance.py`: the Section 3.3.4 column-weighting check (1,200 trees).
- `summarise_ablation.py`, `summarise_celerity.py`: console readouts, `argv[2]` = resdir.

Result directories:

| dir | cells | used by |
|---|---|---|
| `res_main` | 900 (380 Mahanadi, 180 Kaveri, 340 US) | Tables 4-8, the factorial, cross-network |
| `res_celerity` | 120 | Section 4.4 |
| `res_alpha` | 200 | Section 3.9 prose (alpha sweep) |
| `res_invariance` | 20 | Section 3.3.4 (`wcol` arm vs the unscaled base; summarised by `check_invariance.py`) |
| `res_ladder_short`, `res_ladder_long` | 0 (empty) | the ladder sensitivity quoted here earlier (-0.12%, +1.39%) cannot be reproduced from these directories; the paper only says the ladder is fixed, not tuned |

## Two protocols, both reported

`run_fold.py --protocol ...`

- **`transductive` (primary).** Contiguous gaps to a 10% target cut across the whole
  record; the model is fitted on that one gapped frame and fills it in place. This is
  how the method is deployed - you have one gappy record - and how MissForest is
  normally evaluated, so it is the number comparable to the literature. A withheld
  cell is NaN in the frame `fit()` sees, so it is never a training target and never a
  predictor value. What this protocol does allow is overlap between fitted and scored
  rows, which is intrinsic to MissForest.
- **`fold` (confirmation).** The record is split into 3 contiguous blocks; gaps are
  cut in one block and the model is fitted only on the rows outside it, so no row
  containing a withheld cell is ever a training row. Reported to show the ranking does
  not depend on the overlap above.

No seed block in either. Holding the most complete 3-year window out ungapped, as v5
did, leaves only ~5-7% of the record masked and keeps the densest stretch clean.

## Scored set: the connectivity graph

`common.connected`. **Primary scope is the gauges that take part in the connectivity
graph** - 12 of the 22 gauges in the archive (the network is the graph; `load_network` drops the rest outright). The rule is structural and checkable from `C`
alone, with no reference to any result: for a gauge outside the graph
`_routed_upstream` returns `None`, so R does not exist, and its aggregate weights
reduce to exactly normalised inverse distance because the connectivity half of the
blend is all zeros. Neither feature under test can reach such a gauge.

The rule drops ten of the 22 archive gauges: six absent from the connectivity map
(`boudh, kantamal, kesinga, padampur, salebhata, tikarapara`), two with an all-zero row
and column in `C` (`kurubhata, sundergarh`), and two with no observation in
1980-1990 (`kelo, paramanpur`). The US keeps 6 of 10 and the Kaveri, which has no
matrix, all 12. Dropped gauges are neither scored nor predictors, so there is no
secondary all-gauge scope. The rule was written down after it was known that
tikarapara dominates the pooled error, so it is not pre-registered and must not be
presented as such; its justification stands on the structure of `C`, not on results.

## Metrics: never pooled RMSE alone

Pooled RMSE is variance-weighted, so it measures the largest gauge. Tikarapara is 61%
of total network variance and 48-78% of pooled squared error; removing it does not fix
this, because basantpur is then 47% of what remains. The reported scopes are therefore
`gauge_conn` (the primary scope) and `gauge`, which are now the same set: each gauge's RMSE as a percentage of its own mean discharge,
averaged over gauges with equal weight. RMSE, MAE, NSE, KGE and PBIAS are reported for
every scope. `pooled` and `recv` are kept for continuity with v5 only.

**Do not compare absolute error across gap lengths.** `floor(0.10 N / gap)` makes the
achieved mask ~9.9% at 3 d but ~7.5% at 100 d, so the arms differ in how many cells
are scored and where. Every cross-gap statement must be a ratio computed within a gap
length; `achieved_missing_pct` is recorded per cell and printed by the summariser.

## Open before the paper

- Statistics are blocked over gap length (df = 3). Gauges in one chain are not
  independent; a bootstrap over (seed, fold) would be the honest interval.
- Only fixed-length gaps are run. The realistic mixed-length case is the one place
  PIMF has not been shown to beat vanilla (e23 on the co-author's branch, p = 0.21).
- Mixed/native gap lengths are still not run under the transductive protocol. Only
  fixed-length gaps are reported, and mixed lengths are the one case where PIMF has
  not been shown to beat vanilla (p = 0.21 on the co-author's branch).
- Fold-protocol confirmation of the headline table has not been run.
- 5 seeds throughout. The 7-day celerity shift (p = 0.067) and the receiver-scope
  nRMSE contrast are the results that more seeds would settle.

Closed since: alpha is swept over its whole range (paper Section 3.9, immaterial); celerity
is fixed a priori at 100 km/day and swept only as sensitivity (Section 4.4).

## Additional analyses (response to reviewers)

`series_analysis.py` reads `res_series/` (PIMF, vanilla, linear, histmean, kNN, Kalman and Kalman with annual Fourier terms, 20 cells each, imputed series saved as `series/*.npz`) and `res_missing/` (5% and 20% missingness) and prints seasonal and flow-regime contrasts, peak magnitude and timing, per-gauge results, the upstream-downstream consistency check and the Kalman variant. They back Supplementary Tables S.T.4-S.T.7. The cells were produced with

    python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms ARLanc,vanilla,linear,histmean,knn,kalman,kalmanfour --out res_series --series
    python rerun_expts/drive_ablation.py mahanadi --protocol transductive --arms ARLanc,vanilla,linear,knn --out res_missing --pct 20   # and 5
