# E22 - Final PIMF (nearest-observed N) and placebo routing

The committed results used a seed block (pass `--seed-block` to reproduce them). By default the script now gaps the whole 1980-1990 record.

Network: the 16 stations of the connectivity matrix (V1-V16), with the paramanpur -> basantpur link removed from `mahanadi_data/mahanadi_connectivity_matrix.csv`. kelo and paramanpur have no 1980-1990 data and patharidih has none outside the seed block, so **13 stations are scored**. Seed block 1988-01-02 to 1990-12-31; 10% gaps of 3/7/30/100 days in the rest; seeds 42/1/2/3/4.

Arms
- `vanilla`: MissForest on discharge only.
- `pimf`: the final PIMF (`MissForestImputer`, default `n_features='nearest_observed'`). N is the last and next observed value of the target and the distance to each. Training rows get distances drawn from the station's own missing cells.
- `pimf_old`: the v5/v6 paper's PIMF (`n_features='neighbour_day'`). N is the day before and after, read from the live iterate.
- `pimfL`: PIMF without N.
- `pimfL_placebo`: `pimfL` with R built from distance-matched non-connected gauges, at the same lags and weights.

The pickles predate the rename and call the final PIMF `pimfA` and the old one `pimf`; `analyse.py` maps them. The `src/` implementation reproduces the committed `pimfA` and `pimf` columns to within 2e-12 (checked on seed 42 at 100 d and seed 1 at 3 d).

## Run
```
python run_cell.py --root <repo> --data <repo>/mahanadi_data --seed 42 --gap 30 --njobs 4 --seed-block --out results/cells_16stations/cell_42_30.pkl
python analyse.py results/cells_16stations > results/summary_16stations.txt
```
`results/cells_18stations_100d_only` is an earlier, stopped run on 18 stations (22 minus the 4 with no data); only its 100-day cells finished.

## Findings (16-station network, 13 scored, 5 seeds)
The statistics are station-level: each station is averaged over seeds, then a Wilcoxon test over the 13 stations, Holm-corrected across the 4 gap lengths.

- **PIMF vs vanilla:**
  - It wins at 10-11 of 13 stations at every gap length, with a geometric-mean RMSE ratio of 0.92 / 0.94 / 0.93 / 0.92 at 3 / 7 / 30 / 100 d.
  - Uncorrected p = 0.033 / 0.003 / 0.017 / 0.040. After Holm correction only 7 d stays below 0.05 (0.014); the others are 0.051-0.065.
  - Pooled RMSE is 7-10% lower than vanilla in every seed at every gap length.
- **Old PIMF vs vanilla:** it wins at 3 d (11/13) but not beyond, and loses at 100 d (ratio 1.28, 4/13 stations). Its pooled RMSE is 4.2% below vanilla at 100 d in 4 of 5 seeds.
- **PIMF vs old PIMF:** level at 3 d. PIMF is better at 7 / 30 / 100 d (10/13, 13/13, 12/13 stations; Holm p = 0.012 / 0.001 / 0.001; ratio 0.72 at 100 d).
- **PIMF vs PIMF-L (does N add anything?):** PIMF is 3 / 3 / 2 / 1% better (9-10/13 stations), but not significantly after correction. Pooled RMSE favours PIMF-L at 30 and 100 d. The gain over old PIMF comes mostly from removing the drifting N, not from information the nearest-observed values add.
- **Volume bias:**
  - Old PIMF overestimates volume: median |PBIAS| per station is 20-32% at 7-100 d.
  - PIMF is level with vanilla (6.6 / 8.0 / 9.5 / 11.3% against vanilla's 6.2 / 8.2 / 10.3 / 14.7%).
  - Pooled PBIAS: PIMF +1.3 to +2.3%; vanilla +1.1 to +3.7%; old PIMF +5.9 to +13.4%.
- **Placebo:**
  - Real R beats placebo R at all 5 of 5 receivers: basantpur 0.86 (20/20 cells), jondhra 0.90, seorinarayan 0.91, rajim 0.96, simga 0.97. With 5 receivers the smallest possible two-sided p is 0.062, and that is what it reaches.
  - This shows that **which gauges are summed matters**. It does not isolate travel-time routing: the lags are only 0-2 days.
  - The placebo pool for basantpur and seorinarayan is just kurubhata and sundergarh, and rajim's placebo (patharidih) is entirely imputed outside the seed block.
