import numpy as np
import pandas as pd

def calculate_distance_by_poi(poi_type, data, distance_rings):
    counts_by_distance = []
    for i in range(len(distance_rings) - 1):
        counts_by_distance.append(data[(data['nearest_poi_type'] == poi_type) & (data['distance_to_poi'] >= distance_rings[i]) & (data['distance_to_poi'] < distance_rings[i+1])].shape[0])
    return counts_by_distance

def extract_lat_lon(centroid_str):
    lon, lat = centroid_str.replace('POINT(', '').replace(')', '').split()
    return float(lat), float(lon)

# Calculate distance between two geographic points (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) # Convert from degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371 # Radius of Earth (km)
    return r * c
    
def count_within_band(distance_rings, data):
    band_columns = [f"<{float(distance_rings[i+1])}km" for i in range(len(distance_rings)-1)]
    poi_types_list = data['nearest_poi_type'].unique()
    distance_df = pd.DataFrame(index=poi_types_list, columns=band_columns)
    for poi_type in poi_types_list:
        distance_df.loc[poi_type] = calculate_distance_by_poi(poi_type, data, distance_rings)
    distance_df.fillna(0, inplace=True)
    return distance_df