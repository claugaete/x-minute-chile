# helper geometry functions
import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, Point, Polygon

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


def overlay_column(
    main_gdf: gpd.GeoDataFrame, info_gdf: gpd.GeoDataFrame, column_name: str
) -> pd.Series:
    """
    Divide los valores de `column_name` (p. ej. la población) de un
    GeoDataFrame `info_gdf` entre las geometrías de otro GeoDataFrame
    `main_gdf`. Divide aquellas zonas que caen en múltiples geometrías, y
    reparte el valor de `column_name` de forma proporcional al área de la zona
    que cae en cada geometría.

    Parameters
    ---
    main_gdf : GeoDataFrame
        GeoDataFrame al cual se le agregará la nueva columna.
    info_gdf : GeoDataFrame
        GeoDataFrame que contiene la información que se desea agregar. Las
        geometrías pueden ser de tipo `Point`, `Polygon` o `MultiPolygon`.
    column_name : str
        Nombre de la columna de `info_gdf` que se desea traspasar a `main_gdf`.
        
    Returns
    ---
    Una serie con los nuevos valores de `column_name` para `main_gdf`,
    manteniendo el mismo índice.
    """

    def size(shape: Point | Polygon | MultiPolygon) -> float:
        """
        Obtiene el "tamaño" de un objeto (1 si es un punto porque todos los
        puntos son de igual tamaño; el área del polígono si es un
        polígono).
        """

        return 1 if isinstance(shape, Point) else shape.area

    # add column with area of each zone
    info_gdf = info_gdf.to_crs(4326).assign(
        size=info_gdf.to_crs(xmin.projected_crs).geometry.map(size)
    )

    index_name = main_gdf.index.name

    # overlay zones with main GDF, splitting zones that fall between
    # multiple cells into multiple "fragments"
    population_gdf_split = info_gdf.overlay(
        main_gdf.rename_axis("main_gdf_index").reset_index(),
        keep_geom_type=True,
    )

    # assign column value to each zone fragment, proportional to the area of
    # the original zone that the split zone has
    population_gdf_split["column_split"] = (
        population_gdf_split[column_name]
        * population_gdf_split.to_crs(xmin.projected_crs).geometry.map(size)
        / population_gdf_split["size"]
    )

    # group by main GDF to get the total population in each cell
    return (
        population_gdf_split.groupby("main_gdf_index")["column_split"]
        .sum()
        .rename_axis(index_name)
    )
