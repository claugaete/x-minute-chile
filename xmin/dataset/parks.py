import geopandas as gpd
import pandas as pd
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from ..config import config
from ..geometry import convert_polygon_to_representative_points


def clean_parks(
    parks_gdf: gpd.GeoDataFrame,
    pedestrian_network: gpd.GeoDataFrame,
    is_fenced_column: str,
    index_column: str | None = None,
    min_dist: float | None = None,
    max_dist: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Dado un GeoDataFrame con parques (o cualquier otro polígono que se desee
    convertir a puntos), se convierte cada geometría a una serie de puntos
    representativos. Estos puntos se obtienen mediante la función
    `geometry.convert_polygon_to_representative_points`, considerando como
    puntos de entrada la intersección entre el borde del polígono y
    `pedestrian_network`, y solamente agregando puntos intermedios si
    `is_fenced_column` es falso para el polígono en cuestión, o si no hay
    intersección entre el polígono y `pedestrian_network`.
    
    El peso asignado a cada punto es la razón entre el área del polígono que
    representa, y la cantidad de puntos que representan a ese polígono. Es
    decir, si a partir de un polígono de área 5000 se obtuvieron 10 puntos,
    cada punto tendrá un peso 5000/10 = 500.

    Parameters
    ---
    parks_gdf : GeoDataFrame
        GeoDataFrame cuyas geometrías se convertirán a puntos.
    pedestrian_network : GeoDataFrame
        GeoDataFrame con la red peatonal.
    is_fenced_column : str
        Columna de `parks_gdf` donde se guarda si el parque tiene una reja
        (implicando que solo se puede entrar por caminos designados) o no
        (implicando que se puede entrar por cualquier parte).
    index_column : str or None, default: None
        Columna a utilizar como índice para agrupar polígonos como parte del
        mismo parque. Si no se entrega una columna, se utiliza el índice de
        `parks_gdf`.
    min_dist : float or None, default: None
        Distancia mínima que debe haber entre dos puntos representativos
        consecutivos. Debe cumplirse que `min_dist <= max_dist/2`. Si es None,
        se asignará `min_dist = max_dist/2`.
    max_dist : float or None, default: None
        Distancia máxima que debe haber entre dos puntos representativos
        consecutivos. Debe cumplirse que `max_dist >= 2*min_dist`. Si es None,
        se asignará `max_dist = 2*min_dist`.
    """

    parks_gdf = parks_gdf.to_crs(config.projected_crs).assign(
        park_id=(
            parks_gdf.index
            if index_column is None
            else parks_gdf[index_column]
        ),
        area=lambda gdf: gdf.area
    )

    pedestrian_network = pedestrian_network.to_crs(config.projected_crs)

    # some ways in the network are polygons, they need to be converted into
    # lines
    polygons = pedestrian_network.geom_type.isin(["Polygon", "MultiPolygon"])
    # convert polygons to their boundary (returns LinearRing / MultiLineString)
    pedestrian_network.loc[polygons, "geometry"] = pedestrian_network.loc[
        polygons, "geometry"
    ].boundary
    pedestrian_network_union = pedestrian_network.union_all()

    results = []
    for _, row in tqdm(parks_gdf.iterrows(), total=len(parks_gdf)):
        base_geom: BaseGeometry = row["geometry"]
        if isinstance(base_geom, Point):
            results.append(row)
            continue
        elif isinstance(base_geom, Polygon):
            geoms = MultiPolygon([base_geom])
        elif isinstance(base_geom, MultiPolygon):
            geoms = base_geom
        else:
            raise RuntimeError(
                f"Entrada con geometría de tipo {geoms.geom_type}"
            )

        for geom in geoms.geoms:
            
            # generate entry points
            inter = geom.exterior.intersection(pedestrian_network_union)
            if inter.is_empty:
                entry_points = []
            elif isinstance(inter, Point):
                entry_points = [inter]
            elif isinstance(inter, MultiPoint):
                entry_points = list(inter.geoms)
            else:
                raise RuntimeError(
                    f"Intersección con geometría de tipo {inter.geom_type}"
                )
                
            # only add intermediate points if the park is not fenced, or if
            # there are no entry points
            add_extra_points = not (row[is_fenced_column] and entry_points)
            
            # get all representative points and add to results
            representative_points = convert_polygon_to_representative_points(
                polygon=geom,
                entry_points=entry_points,
                add_extra_points=add_extra_points,
                min_dist=min_dist,
                max_dist=max_dist,
            )
            result = gpd.GeoDataFrame(
                [row] * len(representative_points), crs=config.projected_crs
            )
            result = result.set_geometry(representative_points)
            result["weight"] = result["area"] / len(result)
            results.append(result)

    points_gdf = gpd.GeoDataFrame(
        pd.concat(results, ignore_index=True), crs=config.projected_crs
    )
    points_gdf["n_points"] = points_gdf["park_id"].map(
        points_gdf["park_id"].value_counts()
    )

    return points_gdf.to_crs(4326)
