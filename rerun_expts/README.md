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

Section 4.1 of v5: 1980-1990 (US 2013-2024), 3-year gap-free seed block kept ungapped,
10% contiguous gaps of 3/7/30/100 days, seeds 42/1/2, single fit over the whole gapped record.
Linear interpolation matches v5 Table 2 exactly, so the gaps are identical to v5.

## Files

- `run_cell.py`: one Mahanadi cell (arms `pimf`, `pimfL`, `vanilla_single`, `vanilla_seedfit`; `--celerity`)
- `run_net.py`: same for `--net mahanadi|kaveri|us`, plus a `linear` arm
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

| Network | Scope | Change | p | Wins |
|---|---|---|---|---|
| Mahanadi | pooled (18 gauges) | -3.5% | 0.19 | 9/12 |
| Mahanadi | 5 receivers | -15.1% | 0.0001 | 11/12 |
| Kaveri | 12 gauges | +5.3% | 0.42 | 6/12 |
| US | pooled (10) | -13.4% | 0.050 | 8/12 |
| US | 3 receivers | -7.3% | 0.012 | 10/12 |
