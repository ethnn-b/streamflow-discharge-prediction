"""Empirical check of the Mahanadi connectivity matrix: for each (target, contributor)
pair, Spearman correlation of Q_target(t) with Q_contrib(t - k), k = -3..6, over the
full record. A true upstream contributor should peak at k >= 0 with high correlation."""
import contextlib, io, sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.utils.data import load_and_preprocess_data
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix

with contextlib.redirect_stdout(io.StringIO()):
    df, contrib, coords, _, s2v = load_and_preprocess_data(
        'mahanadi_data/mahanadi_discharge_data.csv',
        'mahanadi_data/mahanadi_station_coordinates.csv',
        'mahanadi_data/mahanadi_connectivity_matrix.csv')
    st = sorted(c for c in df.columns if not c.startswith(('day_of', 'month_', 'week_')))
    D = build_distance_matrix(coords, st)
    C = build_connectivity_matrix(contrib, st, s2v)
R = df[st].rank(pct=True)  # Spearman via ranks, ignoring NaN

def prof(i, j):
    out = {}
    for k in range(-3, 7):
        a, b = R[i], R[j].shift(k)
        m = a.notna() & b.notna()
        out[k] = a[m].corr(b[m]) if m.sum() > 200 else np.nan
    s = pd.Series(out)
    n = int((df[i].notna() & df[j].notna()).sum())
    return s.idxmax() if s.notna().any() else None, s.max(), s.get(0), n

listed = [(i, j) for i in st for j in st if i != j and C.loc[i, j] > 0]
cand = [('kantamal', 'kesinga'), ('boudh', 'kantamal'), ('boudh', 'salebhata'), ('tikarapara', 'boudh'),
        ('tikarapara', 'kantamal'), ('tikarapara', 'salebhata'), ('tikarapara', 'kesinga'),
        ('salebhata', 'padampur'), ('paramanpur', 'sundergarh'), ('tikarapara', 'basantpur'),
        ('boudh', 'basantpur'), ('seorinarayan', 'bamnidhi'), ('basantpur', 'kurubhata')]
pd.set_option('display.width', 200)
for name, pairs in [('LISTED in C', listed), ('CANDIDATES (not in C)', cand)]:
    rows = []
    for i, j in pairs:
        k, r, r0, n = prof(i, j)
        rows.append(dict(target=i, contributor=j, km=round(D.loc[i, j]), tau_c100=int(round(D.loc[i, j] / 100)),
                         best_lag=k, r_best=round(r, 3) if r == r else None, r_lag0=round(r0, 3) if r0 == r0 else None,
                         overlap_days=n))
    print(f'\n== {name} ==')
    print(pd.DataFrame(rows).to_string(index=False))
print('\nrecord span per station (first/last valid, % valid 1980-1990):')
for s in st:
    q = df[s]
    print(f'  {s:14s} {q.first_valid_index().date()} {q.last_valid_index().date()}  {100 * q.loc["1980":"1990"].notna().mean():5.1f}%')
