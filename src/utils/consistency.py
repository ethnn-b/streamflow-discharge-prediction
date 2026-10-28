import numpy as np
import pandas as pd


def upstream_sets(connectivity):
    """{station: [contributors]} from the connectivity matrix.

    C[i, j] == 1 means j drains into i, so a station's contributors are the
    nonzero entries of its own row. Diagonal self-entries are dropped.
    """
    sets = {}
    for i in connectivity.index:
        row = connectivity.loc[i]
        ups = [j for j in connectivity.columns if j != i and row.get(j, 0) > 0]
        if ups:
            sets[i] = ups
    return sets


def network_consistency(df, connectivity, gap_mask=None, tol=0.0):
    """How often a discharge series breaks upstream-downstream ordering.

    The connectivity matrix lists every ancestor of a station rather than only its
    immediate neighbours, so contributors overlap and their discharges cannot be
    summed without heavy double counting (basantpur's contributors include
    seorinarayan, which itself contains most of them). The constraint that is valid
    for ancestors is Q_i(t) >= max_j Q_j(t): a downstream gauge must carry at least
    as much as any single gauge above it.

    `tol` is a relative slack, so a violation needs Q_i < (1 - tol) * max_j Q_j.

    With `gap_mask`, rates are split by whether a row involved an imputed value.
    That split is the point: the OBSERVED record violates this too, through
    measurement error, abstraction and regulation, so the observed rate is the only
    meaningful yardstick for an imputed one. A claim of hydrological consistency
    needs the imputed rate to be no worse than the observed rate, not zero.
    """
    sets = upstream_sets(connectivity)
    per_station, totals = {}, {"checked": 0, "violations": 0,
                               "checked_imputed": 0, "violations_imputed": 0,
                               "checked_observed": 0, "violations_observed": 0}
    shortfalls = []

    for i, ups in sets.items():
        if i not in df.columns:
            continue
        row = {"n_upstream": 0, "checked": 0, "violations": 0,
               "checked_imputed": 0, "violations_imputed": 0,
               "checked_observed": 0, "violations_observed": 0}

        # Each ancestor is checked against the receiver INDEPENDENTLY. Requiring
        # every contributor to be present at once collapses coverage - basantpur
        # has 12 nested ancestors and almost never has them all on the same day -
        # and the constraint holds pairwise anyway.
        for j in ups:
            if j not in df.columns:
                continue
            row["n_upstream"] += 1
            pair = df[[i, j]]
            valid = pair.notna().all(axis=1)
            if not valid.any():
                continue
            down, up = df.loc[valid, i], df.loc[valid, j]
            viol = down < (1.0 - tol) * up
            shortfalls.extend((((up - down) / up.replace(0, np.nan))[viol]).dropna().tolist())

            row["checked"] += int(valid.sum())
            row["violations"] += int(viol.sum())

            if gap_mask is not None:
                touched = gap_mask.reindex(columns=[i, j]).fillna(False).any(axis=1)[valid]
                for label, sel in (("imputed", touched), ("observed", ~touched)):
                    row[f"checked_{label}"] += int(sel.sum())
                    row[f"violations_{label}"] += int((viol & sel).sum())

        if not row["checked"]:
            continue
        row["rate"] = row["violations"] / row["checked"]
        if gap_mask is not None:
            for label in ("imputed", "observed"):
                c = row[f"checked_{label}"]
                row[f"rate_{label}"] = row[f"violations_{label}"] / c if c else np.nan
        for k in ("checked", "violations", "checked_imputed", "violations_imputed",
                  "checked_observed", "violations_observed"):
            totals[k] += row[k]
        per_station[i] = row

    for label in ("", "_imputed", "_observed"):
        c, v = totals[f"checked{label}"], totals[f"violations{label}"]
        totals[f"rate{label}"] = v / c if c else np.nan
    totals["mean_relative_shortfall"] = float(np.mean(shortfalls)) if shortfalls else np.nan

    return {"overall": totals, "per_station": pd.DataFrame(per_station).T}
