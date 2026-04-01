# helper geometry functions
import geopandas as gpd

import xmin


def to_centroids(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Convierte las geometrías de un GeoDataFrame que no sean puntos,
    cambiándolas por sus centroides.

    Parameters
    ---
    gdf : GeoDataFrame
        GeoDataFrame para el cual cambiar su geometría.
    
    Returns
    ---
    Un nuevo GeoDataFrame con las geometrías modificadas.
    """
    original_crs = gdf.crs
    return gdf.assign(
        geometry=gdf.geometry.to_crs(xmin.projected_crs).centroid.to_crs(
            original_crs
        )
    )
