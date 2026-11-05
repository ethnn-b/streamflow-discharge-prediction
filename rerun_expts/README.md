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
- `drive_ablation.py`, `drive_celerity.py`: parallel drivers. Both skip cells whose
  JSON already exists, so a run resumes.
- `make_tables.py`: writes `paper_latex_v4/tables_main.tex`. **The paper transcribes no
  number by hand** -- re-run this after any sweep and every table follows.
- `summarise_ablation.py`, `summarise_celerity.py`: console readouts, `argv[2]` = resdir.

Result directories, all still live and all cited by the paper:

| dir | cells | used by |
|---|---|---|
| `res_main` | 820 | Tables 4-8, the factorial, cross-network |
| `res_celerity` | 120 | Section 4.4 |
| `res_alpha` | 200 | Section 4.5 prose |
| `res_ladder_short`, `res_ladder_long` | 20 each | the anchor ladder is fixed, not tuned: a shorter ladder is -0.12% (p=0.91) and a longer one +1.39% (p=0.36) on the primary scope, so the choice is immaterial |

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
graph** - 12 of 18 live Mahanadi gauges. The rule is structural and checkable from `C`
alone, with no reference to any result: for a gauge outside the graph
`_routed_upstream` returns `None`, so R does not exist, and its aggregate weights
reduce to exactly normalised inverse distance because the connectivity half of the
blend is all zeros. Neither feature under test can reach such a gauge.

It excludes six gauges, not one: `kantamal, kesinga, kurubhata, salebhata, sundergarh,
tikarapara`. **All 18 are still reported as the secondary scope** and the direction of
the result must be stated for both. This rule was written down after it was known that
tikarapara dominates the pooled error, so it is not pre-registered and must not be
presented as such; its justification stands on the structure of `C`, not on results.

All 18 live gauges remain predictors and remain imputation targets in every arm. Only
the four columns with no observation at all in the window are excluded, by
`_dead_columns`, which already zero-weights them.

## Metrics: never pooled RMSE alone

Pooled RMSE is variance-weighted, so it measures the largest gauge. Tikarapara is 61%
of total network variance and 48-78% of pooled squared error; removing it does not fix
this, because basantpur is then 47% of what remains. The reported scopes are therefore
`gauge_conn` and `gauge`: each gauge's RMSE as a percentage of its own mean discharge,
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
- 5 seeds throughout. The 7-day celerity shift (p = 0.083) and `{N}` (p = 0.068) are
  the results that more seeds would settle; `ARLanc` vs `ARLancd` they would not.

Closed since: alpha is swept over its whole range (Section 4.5, immaterial); celerity
is fixed a priori at 100 km/day and swept only as sensitivity (Section 4.4).
