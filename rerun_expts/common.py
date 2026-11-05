"""Network loading shared by the experiment drivers."""
import contextlib
import io
import warnings

import pandas as pd

from src.utils.data import load_and_preprocess_data, add_temporal_features
from src.utils.spatial import build_distance_matrix, build_connectivity_matrix

NETS = {
    'mahanadi': ('mahanadi_data/mahanadi_discharge_data.csv',
                 'mahanadi_data/mahanadi_station_coordinates.csv',
                 'mahanadi_data/mahanadi_connectivity_matrix.csv', '1980-01-01', '1990-12-31'),
    'kaveri': ('kaveri_data/cauv_discharge.csv',
               'kaveri_data/lat_long_cauv.csv', None, '1980-01-01', '1990-12-31'),
    'us': ('us_data/us_top10_discharge_data_demo.csv',
           'us_data/us_top10_station_coordinates.csv',
           'us_data/us_top10_connectivity_matrix.csv', '2013-01-01', '2024-12-31'),
}

TEMPORAL_PREFIXES = ('day_of_year_', 'month_', 'week_of_year_')


def load_network(net):
    """Returns (df, stations, temporal_cols, D, C, receivers) for one network."""
    warnings.simplefilter('ignore')
    dpath, cpath, kpath, t0, t1 = NETS[net]
    with contextlib.redirect_stdout(io.StringIO()):
        df_all, df_contrib, df_coords, _, s2v = load_and_preprocess_data(dpath, cpath, kpath)
        df = add_temporal_features(df_all).loc[t0:t1].copy()
        st = [c for c in df.columns if not c.startswith(TEMPORAL_PREFIXES)]
        # USGS no-data sentinel; a real number downstream if left in place.
        df[st] = df[st].mask(df[st] <= -999999)
        temporal = [c for c in df.columns if c.startswith(TEMPORAL_PREFIXES)]
        stations = sorted(st)
        D = build_distance_matrix(df_coords, stations).loc[stations, stations]
        C = build_connectivity_matrix(df_contrib, stations, s2v).loc[stations, stations]
        # THE NETWORK IS THE GRAPH. A gauge that neither receives from nor contributes
        # to another, or that has no observation in the window, is dropped outright -
        # not merely left out of the scored set. Neither A nor R can reach such a gauge
        # (_routed_upstream returns None and the connectivity half of the blend is all
        # zeros), so it can only enter as a generic correlated column, and on Mahanadi
        # the largest of them, tikarapara, is 61% of network variance, is regulated by
        # Hirakud releases, and is absent from the basin connectivity map. Keeping the
        # scored set and the predictor set identical is what makes the comparison a
        # statement about the graph rather than about which extra columns were on hand.
        stations = connected(stations, C, df)
        df = df[stations + temporal]
        D, C = D.loc[stations, stations], C.loc[stations, stations]
    receivers = [s for s in stations if C.loc[s].drop(s, errors='ignore').sum() > 0]
    return df, stations, temporal, D, C, receivers


def connected(stations, C, df=None):
    """Gauges that take part in the connectivity graph, as a contributor or a receiver.

    This is the scored set for the headline tables, and the rule is structural, not
    results-driven: for a gauge outside the graph `_routed_upstream` returns None, so
    R does not exist, and its aggregate weights reduce to pure inverse distance
    because the connectivity half of the blend is all zeros. Neither feature under
    test can reach such a gauge, so scoring it measures the ablation base rather than
    the method. Falls back to every gauge on a network with no connectivity matrix
    (kaveri), where the distinction does not exist.
    """
    out = [s for s in stations
           if C.loc[s].drop(s, errors='ignore').sum() > 0
           or C[s].drop(s, errors='ignore').sum() > 0]
    if df is not None:
        out = [s for s in out if df[s].notna().any()]
    return out or list(stations)


def folds(index, n_folds):
    """n_folds contiguous, equal-length row blocks covering the record."""
    size = len(index) // n_folds
    out = []
    for k in range(n_folds):
        lo = k * size
        hi = len(index) if k == n_folds - 1 else (k + 1) * size
        out.append(index[lo:hi])
    return out
