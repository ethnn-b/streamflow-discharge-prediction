# PIMF: final method and results

**PIMF is now the version with nearest-observed N** (developed as "PIMF-A"). The v5/v6 paper's version is kept only as a legacy option and called `pimf_old` in the results.

## The method
`src/imputers/missforest.py`, `MissForestImputer`. Iterative MissForest. Each station's random forest predicts its missing days from:

| Group | Features |
|---|---|
| Other stations | their same-day discharge |
| A | weighted basin discharge: distance + connectivity weights, same day |
| R | travel-time-routed upstream discharge: upstream gauges read at round(D / 100 km per day) days earlier |
| N | **nearest observed**: last and next observed value of the target station, and the days to each |
| S | sin/cos of day of year |

How N works (`n_features='nearest_observed'`, the default):
- **Inside a gap:** the real nearest observations. They are never the model's own estimates, so nothing is fed back and nothing drifts.
- **On training (observed) days:** a pretend gap. The (days before, days after) are drawn from the station's own missing cells, and the features read the real observations at or beyond those distances. The model therefore learns how much a distant observation is worth.
- **At the record edge**, where one side has no observation: the other side's value is copied, and the distance is reported as 10,000 days.
- **The legacy N** (`n_features='neighbour_day'`) reads the day before and after from the live iterate. Inside long gaps it feeds the model its own estimates, and the result drifts upward ([e21](e21_long_gap_diagnosis/README.md)).

| PIMF configuration | Value |
|---|---|
| Imputer options | `initialization_method='historical_mean'`, `ordering_method='most_full_first'`, `use_routing=True`, `celerity_km_per_day=100` |
| Random forest | 100 trees, `max_features='sqrt'`, 10 sweeps |
| Stations | every station with at least one observation is filled; there is no completeness threshold |

## Why the old PIMF lost on long gaps ([e21](e21_long_gap_diagnosis/README.md))
- **One gauge dominates the pooled score.** In the paper's v6 setup, tikarapara (dam-regulated, not in the connectivity matrix) makes up 87-93% of the pooled squared error in seed 42 at 100 days.
- **The old N drifts upward.** It feeds the model its own estimates; bias grows with depth into the gap, up to about +20% PBIAS. A warm start does not remove it.
- **The gap-distance features never worked.** They are always 0 on training rows, so their importance is exactly 0.

## Main results ([e22](e22_anchor_n_placebo/README.md))
Setup:
- Mahanadi 1980-1990, the 16 connectivity-matrix stations, 13 of them scored.
- Seed block 1988-01-02 to 1990-12-31 held out ungapped; 10% contiguous gaps elsewhere; 5 seeds.

The statistics are station-level: each station is averaged over seeds, Wilcoxon over the 13 stations, Holm-corrected across gap lengths.

**Station level: geometric-mean RMSE ratio to vanilla (stations better than vanilla)**

| Gap | PIMF | PIMF old | PIMF without N |
|---|---|---|---|
| 3 d | **0.92** (10/13) | 0.92 (11/13) | 0.95 (10/13) |
| 7 d | **0.94** (11/13) | 1.00 (6/13) | 0.96 (10/13) |
| 30 d | **0.93** (10/13) | 1.01 (6/13) | 0.95 (9/13) |
| 100 d | **0.92** (10/13) | 1.28 (4/13) | 0.93 (9/13) |

**PIMF vs vanilla, significance per gap**

| Gap | p (uncorrected) | Holm p |
|---|---|---|
| 3 d | 0.033 | 0.065 |
| 7 d | 0.003 | 0.014 |
| 30 d | 0.017 | 0.051 |
| 100 d | 0.040 | 0.065 |

**PIMF vs old PIMF:** level at 3 d; PIMF better at 7 / 30 / 100 d (Holm p = 0.012 / 0.001 / 0.001).

**PIMF vs PIMF without N:** PIMF is 1-3% better, which is not significant after correction.

**Pooled RMSE (cumec, mean of 5 seeds)**

| Gap | vanilla | PIMF | PIMF old | PIMF without N |
|---|---|---|---|---|
| 3 d | 181.8 | 166.2 | 165.0 | 165.3 |
| 7 d | 179.5 | 161.3 | 162.5 | 164.0 |
| 30 d | 221.5 | 202.4 | 204.4 | 201.0 |
| 100 d | 198.7 | 186.0 | 190.2 | 182.1 |

PIMF is 7-10% below vanilla in every seed at every gap length.

**Volume bias (median |PBIAS| per station-seed)**

| Gap | vanilla | PIMF | PIMF old | PIMF without N |
|---|---|---|---|---|
| 3 d | 6.2% | 6.6% | 8.7% | 7.5% |
| 7 d | 8.2% | 8.0% | 20.2% | 8.3% |
| 30 d | 10.3% | 9.5% | 21.5% | 9.6% |
| 100 d | 14.7% | 11.3% | 31.8% | 11.3% |

**Placebo routing (real R vs R from distance-matched non-connected gauges, same lags and weights)**

| Receiver | RMSE ratio real / placebo | Cells won |
|---|---|---|
| basantpur | 0.86 | 20/20 |
| jondhra | 0.90 | 15/20 |
| seorinarayan | 0.91 | 12/18 |
| rajim | 0.96 | 14/20 |
| simga | 0.97 | 13/20 |

All 5 receivers favour real R. That is the most a 5-receiver test can show (p = 0.062).

## Mixed gap sizes ([e23](e23_mixed_gap_sizes/README.md))
Gap lengths are drawn from the record's own native missing runs (2-365 d); 2 seeds; 92% of scored days lie in gaps over 30 d.
- **Bias:** PIMF +1 to +3% and vanilla -0.2 to +2.2%, against old PIMF's +10 to +14%.
- **PIMF vs old PIMF:** PIMF is better in 69% of station-seed cases (p = 0.006).
- **PIMF vs vanilla:** not shown to be better (65%, p = 0.21).

## Connectivity matrix ([e24](e24_connectivity_check/README.md))
- All 29 listed pairs peak at a lag of 0-2 days.
- The paramanpur -> basantpur link was wrong and has been removed.
- kesinga -> kantamal is the strongest real pair in the data, but both gauges lie outside the 16-station network.

## What the evidence supports, and what it does not
- **Supported:**
  - PIMF beats vanilla at most stations at every gap length (consistent, about 6-8%; individually significant after correction only at 7 d).
  - PIMF removes the old version's long-gap drift and volume bias.
  - The choice of upstream gauges in R carries information beyond nearby gauges.
- **Not supported:**
  - That the nearest-observed features add much beyond dropping N altogether.
  - That PIMF beats vanilla on realistic mixed gaps (2 seeds).
  - That the R gain is due to travel-time lags specifically.
- **Caveats:**
  - 5 seeds and 13 stations.
  - The results use the seed-block protocol. New runs default to gapping the whole record (`--seed-block` off).
  - The pretend-gap distances follow all missing cells, native ones included, so at heavily gapped stations (kotni, rampur, seorinarayan, baronda) training leans to long distances. The N benefit there is smaller (0-2% vs 1-4%).
