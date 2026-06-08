import pandas as pd
import numpy as np
from geopy.distance import geodesic

def build_distance_matrix(df_coords, discharge_cols):
    """Build distance matrix between all stations."""
    print("Building distance matrix...")
    stations = discharge_cols
    distance_matrix = pd.DataFrame(np.inf, index=stations, columns=stations)
    
    for station_i in stations:
        for station_j in stations:
            if station_i == station_j:
                distance_matrix.loc[station_i, station_j] = 0
                continue
            if station_i in df_coords.index and station_j in df_coords.index:
                lat1, lon1 = df_coords.loc[station_i, ['Latitude', 'Longitude']]
                lat2, lon2 = df_coords.loc[station_j, ['Latitude', 'Longitude']]
                dist = geodesic((lat1, lon1), (lat2, lon2)).km
                distance_matrix.loc[station_i, station_j] = dist
    return distance_matrix

def build_connectivity_matrix(df_contrib, discharge_cols, station_name_to_vcode):
    """Build directed connectivity matrix from contributor data."""
    print("Building connectivity matrix...")
    stations = discharge_cols
    connectivity_matrix = pd.DataFrame(0.0, index=stations, columns=stations)
    
    if df_contrib is None or df_contrib.empty:
        return connectivity_matrix
        
    for _, row in df_contrib.iterrows():
        station = row.get('station')
        contributor = row.get('contributor')
        if station in connectivity_matrix.index and contributor in connectivity_matrix.columns:
            connectivity_matrix.loc[station, contributor] = 1.0
            
    return connectivity_matrix
