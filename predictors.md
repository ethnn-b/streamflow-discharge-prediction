# What exactly are the predictors

Precise specification of every predictor the model sees, for the methods section of
the revision. The ablation evidence for which of these earn their place is in
`feature_sets.md`; this file only defines them.

## Notation

| symbol | meaning |
|---|---|
| `i` | the target station being imputed |
| `j` | any other station |
| `t` | a daily timestamp |
| `Q_j(t)` | discharge at station `j` on day `t`, in cumec |
| `X` | the **working frame** during imputation: observed values where observed, the model's current estimate where missing. MissForest sweeps the stations repeatedly and `X` is updated in place, so `X` improves with each sweep |
| `M_i(t)` | 1 if `Q_i(t)` is missing in the record being imputed, 0 otherwise |
| `C[i,j]` | 1 if station `j` is a listed upstream contributor to `i`, else 0 |
| `d_ij` | geodesic distance between `i` and `j` in km |

The record is 22 Mahanadi gauges at daily resolution. When imputing station `i`, the
model is fitted on the days where `Q_i` is **observed** and applied to the days where
it is missing.

---

## The predictor vector

For target `i` at time `t`, 29 predictors:

```
  21  Q_j(t)                          every other station, same day
   1  weighted_upstream_discharge(t)  weighted sum of them, same day
   1  routed_upstream_discharge(t)    weighted sum of upstream gauges, time-shifted
   4  lag_prev, lag_next,             the target's own temporal neighbourhood
      gap_dist_prev, gap_dist_next
   2  day_of_year_sin, day_of_year_cos
```

Of these, the 21 station columns and the 2 seasonal terms are the published method's
entire predictor set. The other 6 are new.

---

## 1. Station columns — `Q_j(t)` for all `j != i`

The other 21 gauges on the same day, read from `X`. Where station `j` is itself
missing on day `t`, this is `j`'s current estimate, not a gap — MissForest never has a
NaN in its predictor matrix after initialisation.

**These are plain, unweighted values.** The published method multiplies each column by
an inverse-distance weight, which has no effect: a decision tree splits one feature at
a time on the rank order of its values, and multiplying a column by a positive
constant leaves that order unchanged. Verified directly — identical tree node counts,
identical split features, feature importances agreeing to 1e-12, maximum prediction
difference 1.4e-12, and no weight ever exactly zero across all 462 station pairs.
Weighting a predictor column is therefore not a way to inform a forest; the two
features below are.

## 2. The weights — `w_ij`

Both engineered spatial predictors need a weight vector over the 21 predictors:

```
  dist_ij = 1 / (d_ij + 1e-9)      normalised to sum to 1 over the predictors of i
  conn_ij = C[i,j]                 normalised to sum to 1 over the predictors of i
  w_ij    = 0.5 * dist_ij + 0.5 * conn_ij,   then renormalised to sum to 1
```

`alpha = 0.5`, the inverse-distance power `p = 1` and `eps = 1e-9` are hardcoded.

Worked values for `jondhra` — note that the distance term is non-zero for every pair,
so stations with `C[i,j] = 0` still receive weight:

| `j` | `w_ij` | `d_ij` km | upstream of jondhra? |
|---|---|---|---|
| ghatora | 0.14379 | 39.0 | yes |
| simga | 0.12473 | 69.0 | yes |
| andhiyarkhore | 0.12199 | 77.6 | yes |
| patharidih | 0.11921 | 88.9 | yes |
| kotni | 0.11353 | 126.2 | yes |
| rampur | 0.08715 | 19.6 | **no** |
| seorinarayan | 0.06609 | 25.8 | **no — it is downstream of jondhra** |
| bamnidhi | 0.03986 | 42.8 | **no** |

## 3. `weighted_upstream_discharge` — the aggregate

```
  weighted_upstream_discharge_i(t) = SUM over all j != i of  w_ij * Q_j(t)
```

One number per day: a weighted average of every other gauge on the same day. Unlike
rescaling the columns individually, collapsing them into a single sum produces values
whose ordering depends on the weights, so a tree can split on it.

**The name is wrong and should change in the revision.** Because the distance term
gives weight to every station, the sum runs over all 21 — not the upstream set. The
share of weight landing on non-upstream stations:

| target | upstream gauges | weight on upstream | weight elsewhere |
|---|---|---|---|
| basantpur | 12 | 0.863 | 0.137 |
| seorinarayan | 9 | 0.772 | 0.228 |
| jondhra | 5 | 0.623 | **0.377** |
| simga | 2 | 0.600 | **0.400** |
| rajim | 1 | 0.739 | 0.261 |

It is a distance-and-topology weighted basin average, and for `jondhra` it includes
a downstream gauge at weight 0.066. `weighted_basin_discharge` would be accurate.

It is defined for all 22 targets.

## 4. `routed_upstream_discharge` — the routing predictor

```
  tau_ij = round(d_ij / celerity),     celerity = 100 km/day
  routed_upstream_discharge_i(t) = SUM over j with C[i,j] = 1 of  w_ij * Q_j(t - tau_ij)
```

Series edges are filled by back-fill then forward-fill, then 0.

This is the only predictor that uses the connectivity matrix non-redundantly.
Topology selects which gauges are upstream; distance divided by a wave celerity says
how long their water takes to arrive; so each term reads a **different day** from the
one the raw column already supplies. Rescaling cannot do this and neither can the
aggregate.

Restricted to the upstream set, so it exists only for the five stations that have one
— `basantpur`, `seorinarayan`, `jondhra`, `simga`, `rajim`. For the other 17 targets
the feature is absent, and those models are identical with and without routing
enabled.

For `jondhra`, `tau` at 100 km/day: `ghatora` 0, `andhiyarkhore` 1, `kotni` 1,
`patharidih` 1, `simga` 1.

### The celerity is too high for daily data

Across all 29 real upstream pairs, `tau = round(d_ij / 100)` is:

| `tau` (days) | pairs |
|---|---|
| 0 | **9** |
| 1 | 19 |
| 2 | 1 |

A third of the pairs get `tau = 0`, where the routed term reduces to the same-instant
weighted sum and adds nothing the aggregate does not already carry. The remainder are
a flat one-day shift. So the feature as configured is a coarse instrument, and the
celerity sweep is now a requirement rather than a nicety — a lower celerity (30-50
km/day) would spread `tau` over 1-3 days and let the predictor express the routing it
is named for.

### This predicts its own null case, and the prediction holds

`rajim` has exactly one upstream gauge, `baronda`, at `tau = 0`. Its routing predictor
is therefore `w * Q_baronda(t)` — a single column multiplied by a positive constant,
which is inert against a tree for exactly the reason the published weighting is inert.
`rajim` should gain nothing from routing, and it is the one receiver that does not:

| target | upstream pairs with `tau >= 1` | routing effect on RMSE |
|---|---|---|
| jondhra | 4 | -17.19% |
| seorinarayan | 7 | -15.73% |
| basantpur | 8 | -13.35% |
| simga | 1 | -7.38% |
| **rajim** | **0** | **-0.91%** |
| 17 headwater stations | none (no upstream at all) | -0.72% |

The mechanism therefore predicts, before the fact, both which stations benefit and
which single receiver cannot — and the measurement agrees.

## 5. The lagged predictors

Four features, from the target's own series.

```
  lag_prev_i(t)      = X_i(t - 1)
  lag_next_i(t)      = X_i(t + 1)
  gap_dist_prev_i(t) = number of consecutive missing days ending at t      (0 if observed)
  gap_dist_next_i(t) = number of consecutive missing days starting at t    (0 if observed)
```

Series edges are back-filled then forward-filled, then 0.

**`lag_prev` and `lag_next` are read from `X`, the working frame, not from the record.**
So at an observed day they are the neighbouring observed values; inside a gap they are
the model's own current estimates for the adjacent days, refined on every MissForest
sweep. This is what lets information propagate inward from both gap edges, and it is
also the source of the volume bias documented in `feature_sets.md` section 3 — there is
no observed anchor deep inside a long gap, so estimates can drift. Replacing them with
frozen observed anchors removes the bias and costs 57% RMSE at 30-day gaps, so the
feedback is doing the work.

**The two distance features are the model's trust signal.** They tell it how far the
row sits from real data, so it can discount the anchors when they are stale. They are
computed from the gap mask `M_i`, never from values.

**On leakage.** Knowing *where* the gaps are is legitimate — a missingness pattern is
given at inference time. The withheld **values** are never read: everything comes from
`X`, which contains the initialisation and the model's own estimates in the missing
cells, never the held-out truth. This was tested before any accuracy was measured.
These features are also disabled in `vanilla_mode`, so the conventional-MissForest
ablation baseline cannot silently acquire them.

## 6. `day_of_year_sin`, `day_of_year_cos`

```
  day_of_year_sin(t) = sin(2*pi * doy(t) / 366)
  day_of_year_cos(t) = cos(2*pi * doy(t) / 366)
```

A smooth annual cycle with no January discontinuity. The pair together identifies a
position in the year; either alone is ambiguous between two dates. These are the only
temporal features in the published method, and they are shared across all targets
rather than computed per station.

---

## Worked example

Target `jondhra`, a five-day gap over 1985-08-10 to 1985-08-14 in the monsoon, after
historical-mean initialisation and before any sweep. `Q_true` is the held-out truth,
shown only for the reader — the model never sees it. `Q_in_X` is what the working
frame holds.

| date | `Q_true` | `Q_in_X` | missing | aggregate | routed | `lag_prev` | `lag_next` | `gap_dist_prev` | `gap_dist_next` |
|---|---|---|---|---|---|---|---|---|---|
| 1985-08-08 | 1946.9 | 1946.9 | no | 1095.63 | 166.99 | 406.8 | 3543.5 | 0 | 0 |
| 1985-08-09 | 3543.5 | 3543.5 | no | 896.90 | 278.97 | 1946.9 | 323.9 | 0 | 0 |
| 1985-08-10 | 2501.0 | 323.9 | **yes** | 470.79 | 248.92 | 3543.5 | 323.9 | 1 | 5 |
| 1985-08-11 | 1350.0 | 323.9 | **yes** | 350.93 | 128.95 | 323.9 | 323.9 | 2 | 4 |
| 1985-08-12 | 918.7 | 323.9 | **yes** | 227.65 | 95.27 | 323.9 | 323.9 | 3 | 3 |
| 1985-08-13 | 691.9 | 323.9 | **yes** | 192.61 | 62.30 | 323.9 | 323.9 | 4 | 2 |
| 1985-08-14 | 461.9 | 323.9 | **yes** | 175.07 | 55.90 | 323.9 | 640.0 | 5 | 1 |
| 1985-08-15 | 640.0 | 640.0 | no | 274.32 | 62.57 | 323.9 | 1372.3 | 0 | 0 |
| 1985-08-16 | 1372.3 | 1372.3 | no | 581.72 | 107.59 | 640.0 | 2499.8 | 0 | 0 |

What the table shows:

- `gap_dist_prev` counts 1 to 5 into the gap and `gap_dist_next` counts 5 down to 1,
  so the model can locate every row relative to both edges. Both are 0 on observed
  days.
- On 1985-08-10, `lag_prev` is 3543.5 — the real observed value from the day before
  the gap. Deeper in, both lags sit at 323.9, the historical-mean initialisation,
  because no observed day is adjacent. Those values are replaced by better estimates
  on each sweep; this is the first pass.
- The initialisation of 323.9 against a true 2501.0 shows why iteration matters at a
  monsoon peak, and why the cross-sectional predictors carry the early sweeps: the
  aggregate still reads 470.79 on 08-10 and falls through the gap, tracking the
  recession in the other gauges while the target's own lags know nothing yet.
- The routed term leads the aggregate — it peaks on 08-09 at 278.97 and is still
  248.92 on 08-10 — because it is reading upstream gauges a day earlier.

---

## Configuration constants

| constant | value | status |
|---|---|---|
| `alpha`, the distance/connectivity blend | 0.5 | hardcoded, not a constructor argument |
| inverse-distance power `p` | 1 | hardcoded as `1 / (d + 1e-9)` |
| `eps` | 1e-9 | hardcoded |
| `celerity` | 100 km/day | constructor default, never swept |
| `max_iter` | 10 (5 in the ablations) | constructor argument |
| `n_estimators` | 100 | constructor argument |
| `max_features` | `'sqrt'` | hardcoded in the forest call |
| tree depth | unlimited | sklearn default |

No out-of-bag or k-fold tuning is performed anywhere, contrary to section 3.8 of the
current manuscript. Distances come from `geopy`, not `geographiclib` or `pyproj`.
