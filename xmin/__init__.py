import geopandas as gpd

# default projected CRS (Chile)
projected_crs = 5361


def _convert_geometries_to_centroids(gdf: gpd.GeoDataFrame):
    """
    Convierte las geometrías de un GeoDataFrame que no sean puntos,
    cambiándolas por sus centroides.
    
    Parameters
    ---
    gdf : GeoDataFrame
    """
    original_crs = gdf.crs
    gdf.geometry = gdf.geometry.to_crs(
        projected_crs
    ).centroid.to_crs(original_crs)