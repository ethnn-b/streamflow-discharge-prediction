# Reruns against a fairly-trained vanilla MissForest

Reruns of the v5 paper experiments using the code on `new-pimf` (85826ae), unchanged.
Only this folder is new.

## What was rerun and why

| v5 location | Problem | Rerun |
|---|---|---|
| Table 2 vanilla row, Table 4, vanilla rows of Tables 8/9 | `benchmark_1980_1990.py` fits vanilla on the 3-year seed block only, while PIMF is fitted on the whole gapped record | `vanilla_single`: vanilla fitted once on the whole gapped record, same as PIMF. The as-published version is kept as `vanilla_seedfit` |
| Table 2 PIMF row | Committed code does not reproduce v5 (100 d: 323.86 here vs 363.45 in v5) | PIMF from committed code |
| Table 4 | Recomputed as PIMF vs vanilla paired tests | 12 seed x gap cells |
| Table 7 (celerity) | Run on reduced feature sets, not on PIMF | Full PIMF, c = 25-350, 7 d and 100 d |
| US network (Table 9, Table 10) | Two `-999999` USGS sentinels at `usgs_01010000` (2024-11-29/30) were treated as discharge | Masked in `run_net.py`, results in `res3/` |
| Kaveri (Table 8) | Vanilla baseline as above | All arms rerun |
| Table 10 (PIMF-L) | Depended on old PIMF numbers and contaminated US data | PIMF-L at 100 d, all networks |

Tables 5 and 6 were not rerun: the scripts that produced them are not in the repo.

## Protocol

Section 4.1 of v4: 1980-1990 (US 2013-2024), 3-year seed block kept ungapped,
contiguous gaps of 3/7/30/100 days targeting 10% missingness, seeds 42/1/2, single fit
over the whole gapped record.
Linear interpolation matches v4 Table 2 exactly, so the gaps are identical to v4.

Two things the earlier version of this file got wrong about the protocol:

- The seed block (1985-12-09 to 1988-12-07) is the *most complete* three-year window,
  not a gap-free one. Eleven of the 22 Mahanadi gauges have native gaps inside it and
  five have no data in it at all.
- The 10% target is not achieved at every gap length. The gap count per column is
  `floor(0.10 * n_days / gap_length)`, so the achieved missingness is 9.96 / 9.82 /
  9.24 / **6.84**% at 3 / 7 / 30 / 100 days. The 100-day column is a lower-missingness
  test on about a third fewer withheld cells, not purely a longer-gap test.

## Files

- `run_net.py`: one cell, for `--net mahanadi|kaveri|us` and `--celerity`. Arms:
  `pimf`, `pimfL`, `vanilla_single`, `vanilla_single_tf`, `vanilla_seedfit`, `linear`.
  (`run_cell.py` was a Mahanadi-only subset of this and has been removed; `drive.py`
  now calls `run_net.py --net mahanadi`, so every cell comes from one script.)
- `drive.py`: Mahanadi main runs and celerity sweep -> `res/`
- `drive2.py`: Kaveri, Mahanadi PIMF-L and linear -> `res2/`
- `drive3.py`: US with sentinels masked -> `res3/`
- `summarise.py`: seed-mean tables, paired t-tests, per-station changes, celerity ranks

Each JSON holds pooled, receiver and per-station RMSE/MAE/NSE/KGE/PBIAS on the gap cells.

```
python rerun_expts/summarise.py     # from the repo root; no re-running needed
python rerun_expts/drive.py         # re-runs (skips cells whose JSON exists)
```

## Headline (PIMF vs `vanilla_single`, mean % RMSE change over 12 cells)

`p` is blocked over the four gap lengths (df = 3). The earlier version of this table
reported a one-sample t-test over all 12 seed x gap cells, which assumes the cells are
exchangeable; they are not, because the effect varies with gap length and changes sign
between 30 and 100 days. That test gave p = 0.0001 where the blocked test gives 0.002.

| Network | Scope | Change | p (blocked) | Wins |
|---|---|---|---|---|
| Mahanadi | pooled (18 of 22 gauges scored) | -3.8% | 0.28 | 9/12 |
| Mahanadi | 5 receivers | **-14.3%** | **0.002** | **12/12** |
| Kaveri | 12 gauges | +5.3% | 0.65 | 6/12 |
| US | pooled (10) | -13.4% | 0.30 | 8/12 |
| US | 3 receivers | -7.3% | 0.054 | 10/12 |

`vanilla_single_tf` is a control added during the audit: vanilla fitted once on the
whole gapped record but keeping the seasonal columns that the committed benchmark's
vanilla also received. It separates the fitting schedule from the feature set, and it
lands on `vanilla_single` (-3.0% pooled, -15.7% at the receivers), so the collapse of
the published -15.4% gain is the schedule, not the features.

## Effect of the code fixes in this branch

Two defects in `src/imputers/missforest.py` were fixed after the first round of reruns,
and the Mahanadi cells were regenerated (Kaveri and the US are unaffected: neither has
an all-empty column, and neither ran the `vanilla_seedfit` arm).

- Columns with no observations in the window took weight mass in the Eq. (11)
  aggregate - 15% on average over Mahanadi targets, 36% at the worst. Excluding them
  improved PIMF most where the dilution was worst, the pooled scope: 100-day pooled
  RMSE 323.86 -> 314.47, and PIMF vs `vanilla_single` at 100 d from +6.8% to +3.9%.
  At the receivers PIMF went from 11/12 to **12/12** cells.
- `col_means.get(col, 0)` returned a NaN rather than the default for a column that was
  empty in the fitted frame, so `vanilla_seedfit` left cells unimputed and the metrics
  silently dropped them while still reporting the full cell count. The fallback is now
  training mean -> frame mean -> 0, and `run_net.py` asserts no NaN predictions escape.

**The celerity sweep did not survive this.** Before the fix, c = 350 (every lag rounded
to zero, i.e. no time shift) was the worst of the seven celerities in all three seeds at
the 7-day receivers, +10.5%. After it, c = 350 ranks 4th, 6th and 1st (+2.9%,
p_holm = 1.00) and the slowest celerity c = 25 is instead worst in all three seeds;
at 7 d pooled, c = 350 is *better* than c = 100 (-3.6%, p_holm = 0.05). The ranking
therefore flips under a fix that touches no routing code, which means that with three
seeds the celerity is not identified on this arm. Note this sweep is run on full PIMF,
whereas the paper's Table 7 sweeps `{A,R}` and `{R,L}`, so it is not a direct
reproduction of that table - but it does mean the full-PIMF version of the argument
cannot be relied on.
