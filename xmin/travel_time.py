from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import r5py

import xmin
from xmin.amenities import Amenity
from xmin.origins import Origins


def try_snap_to_network(
    transport_network: r5py.TransportNetwork,
    street_mode: r5py.TransportMode | str,
    point_gdf: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Intenta hacer "snapping" de un conjunto de puntos, modificando sus
    posiciones para que toquen la red de transporte.

    Parameters
    ---
    transport_network : r5py.TransportNetwork
        Red de transporte de r5py de la cual extraer los caminos disponibles
        para el "snapping".
    street_mode : str or r5py.TransportMode
        Modo de transporte que deben aceptar los caminos disponibles para el
        "snapping".
    point_gdf : GeoDataFrame
        Puntos a los cuales realizar el "snapping".
    area_gdf : GeoDataFrame, default: None
        Áreas de las cuales no salirse al hacer el "snapping" de los puntos de
        `point_gdf`. Este GeoDataFrame debe compartir índice con `point_gdf`.
        Si, luego del "snapping", un punto de `point_gdf` queda fuera del área
        entregada por su equivalente en `area_gdf`, entonces se cancelará el
        "snapping" para ese punto.

    Returns
    ---
    GeoDataFrame con los puntos modificados.
    """

    snapped_points: gpd.GeoSeries = transport_network.snap_to_network(
        point_gdf.geometry, street_mode=street_mode
    )

    # if point is empty, or falls outside its designated area, undo snapping
    # and return original point
    return point_gdf.assign(
        geometry=np.where(
            snapped_points.is_empty
            | (
                area_gdf is not None
                and ~area_gdf.contains(snapped_points.geometry)
            ),
            point_gdf.geometry,
            snapped_points.geometry,
        )
    )


def calculate_travel_time_matrices(
    origins: Origins,
    amenities: list[Amenity],
    gtfs_paths: str | Path | list[str] | list[Path],
    osm_path: str | Path,
    snap_to_network: str | bool = False,
    snap_street_mode: r5py.TransportMode | str = r5py.TransportMode.CAR,
    **kwargs,
) -> dict[Amenity, gpd.GeoDataFrame]:
    """
    Calcula los tiempos de viaje desde un conjunto de orígenes hacia varios
    conjuntos de necesidades, utilizando una sola operación de cálculo de TTM
    (Time Travel Matrix).

    Parameters --- origins : Origins
        Orígenes desde los cuales se desean calcular tiempos de viaje.
    amenities : list[Amenity]
        Conjuntos de necesidades que se desean cubrir.
    gtfs_paths: str | Path | list[str] | list[Path]
        Ruta(s) a el/los archivo(s) GTFS a utilizar para obtener viajes en
        transporte público. Puede dejarse vacío si no se utilizará transporte
        público en el cálculo.
    osm_path : str | Path
        Ruta al archivo OSM desde el cual se extraerá la red de transporte
        (para caminata/bicicleta).
    snap_to_network: str or bool, default: False
        Si se desea hacer "snapping" de los orígenes/destinos a la red de
        transporte. Existen cuatro opciones posibles: - `False` o `"none"`: no
        hacer "snapping". - `"origins"`: solo hacer "snapping" a los orígenes.
        - `"amenities"`: solo hacer "snapping" a las necesidades/destinos. -
        `True` o `"all"`: hacer "snapping" tanto a orígenes como a
        necesidades/destinos.
    snap_street_mode : str or r5py.TransportMode, default: TransportMode.CAR
        Modo de transporte que deben aceptar los caminos disponibles para el
        "snapping". Irrelevante si no se aplica "snapping".
    **kwargs
        Argumentos que serán pasados al cálculo de la TTM. Puede ser cualquier
        argumento que se pueda pasar a `r5py.RegionalTask`.
    """

    # load r5py and pois
    transport_network = r5py.TransportNetwork(osm_path, gtfs_paths)

    # convert origins from areas to points
    origin_points_gdf = origins.h3_grid.copy()
    xmin._convert_geometries_to_centroids(origin_points_gdf)

    # assign each amenity to its category and concatenate
    all_amenities = pd.concat(
        [
            amenity.amenity_gdf.assign(_amenity_id=id(amenity))
            for amenity in amenities
        ]
    )

    # snap to network if necessary
    if snap_to_network in ("origins", "all", True):
        origin_points_gdf = try_snap_to_network(
            transport_network,
            snap_street_mode,
            origin_points_gdf,
            origins.h3_grid,
        )
    if snap_to_network in ("amenities", "all", True):
        all_amenities = try_snap_to_network(
            transport_network, snap_street_mode, all_amenities
        )

    # calculate matrix
    travel_time_matrix = r5py.TravelTimeMatrix(
        transport_network,
        origins=origin_points_gdf,
        destinations=all_amenities,
        **kwargs,
    )
    travel_time_matrix["_amenity_id"] = travel_time_matrix.merge(
        all_amenities, left_on="to_id", right_on="id"
    )["_amenity_id"]

    # split matrix by amenities and return
    return {
        amenity: travel_time_matrix[
            travel_time_matrix["_amenity_id"] == id(amenity)
        ].drop(columns="_amenity_id")
        for amenity in amenities
    }
