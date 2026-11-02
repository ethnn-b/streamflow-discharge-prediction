# E22 - Anchor features for N, and placebo routing

Network: the 16 connectivity-matrix stations (V1-V16), with the paramanpur -> basantpur link removed from
`mahanadi_data/mahanadi_connectivity_matrix.csv`. kelo and paramanpur have no 1980-1990 data: they stay as
(constant) predictor columns but are never fitted or scored, so 14 gauges are scored. Seed block 1988-01-02 to
1990-12-31, 10% gaps of 3/7/30/100 days, seeds 42/1/2/3/4.

Arms
- `vanilla` - MissForest on discharge only
- `pimf` - paper PIMF (A, R, N from the live iterate, S)
- `pimfL` - PIMF without N
- `pimfA` - N replaced by anchor features: the last and next observed value and the distance to each. Training rows get synthetic distances drawn from the station's own missing-cell (g-, g+) pairs.
- `pimfL_placebo` - pimfL with R built from distance-matched non-connected gauges, with the same lags and weights

## Run
```
python run_cell.py --root <new-pimf checkout> --data <repo>/mahanadi_data --seed 42 --gap 30 --njobs 4 --out results/cells_16stations/cell_42_30.pkl
python analyse.py results/cells_16stations > results/summary_16stations.txt
```
`results/cells_18stations_100d_only` is an earlier, stopped run on 18 gauges (22 minus the 4 with no data); only its 100-day cells finished.

## Findings (16 stations, 5 seeds)
- Every PIMF variant beats vanilla on pooled RMSE at every gap length, about 8-10%. The 100-day loss disappears once tikarapara is out.
- Per station, PIMF-A beats vanilla in 80/71/65/62% of station-seeds at 3/7/30/100 days (all p <= 0.022). Old PIMF falls to 38% at 100 days (worse, p = 0.002).
- PBIAS: old PIMF +6 to +13%; PIMF-A +1.3 to +2.3%, the lowest of all arms.
- Placebo: real R beats placebo R by 7.4% at the receivers (p = 4e-9); basantpur 20/20 cells, jondhra and seorinarayan p < 0.01, rajim and simga not significant.
