# E21 - Why PIMF loses to vanilla MissForest on long gaps

Diagnosis of the v6 Table 2 result (PIMF 323.9 vs vanilla 293.0 RMSE at 100-day gaps).
Paper protocol: 22 Mahanadi gauges, 1980-1990, 3-year seed block held out, 10% gaps, seeds 42/1/2, 30 and 100 days.

Arms: `vanilla`, `pimf`, `pimfL` (no N), `pimf_warm` (PIMF initialised from vanilla's output, a probe only).
Here `pimf` is the v5/v6 paper's PIMF, now `n_features='neighbour_day'` (`pimf_old` elsewhere). This diagnosis is what led to the final PIMF in e22.

## Run
Needs the PIMF code with routing and neighbour-day features (branch `review/rerun-vs-vanilla` or `new-pimf`); `--root` points at a checkout of it.
```
python run_cell.py --root <new-pimf checkout> --seed 42 --gap 100 --out results/cells/cell_42_100.pkl
python analyse.py results/cells > results/summary.txt
```

## Findings
- The pooled 100-day loss is mostly one gauge: in seed 42, tikarapara makes up 87-93% of the squared error. Without it, PIMF 184 vs vanilla 217.
- N (lag_prev/lag_next) is read from the model's own estimates inside a gap. PIMF's bias grows with depth into the gap, reaching about +20% PBIAS. A warm start from vanilla does not remove it, so the feedback loop causes it, not the initialisation.
- gap_dist_prev/next have feature importance exactly 0: they are always 0 on training (observed) rows.
- Per station, PIMF already loses to vanilla at 30 days (wins 43% of station-seeds) and at 100 days (41%).
