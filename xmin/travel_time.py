from collections import Counter
from functools import reduce
import operator
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import r5py
from tqdm.auto import tqdm

from .amenities import Amenity
from .config import config
from .geometry import to_centroids
from .origins import Origins


def _check_duplicate_amenities(amenities: list[Amenity]):
    """
    Función helper para revisar si un conjunto de Amenities tiene nombres
    duplicados.

    Parameters
    ---
    amenities : list[Amenity]
        Lista de necesidades a revisar.

    Returns
    ---
    None; genera un warning si encuentra duplicados.
    """

    counts = Counter([amenity.name for amenity in amenities])
    duplicates = [item for item, count in counts.items() if count > 1]
    if duplicates:
        warnings.warn(
            "Los siguientes nombres de Amenities están duplicados: "
            + ", ".join(duplicates)
            + ". Ambas necesidades serán consideradas como una sola, lo "
            "cual podría generar resultados inesperados; se recomienda "
            "cambiar los nombres de las necesidades para que sean "
            "distintos."
        )


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


def chunk_gdf(gdf: gpd.GeoDataFrame, chunk_size: int):
    """Itera sobre un GeoDataFrame en chunks de tamaño `chunk_size`."""
    for start in range(0, len(gdf), chunk_size):
        yield gdf.iloc[start : start + chunk_size]


def compute_matrix(
    transport_network: r5py.TransportNetwork,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    chunk_size: int | None,
    **kwargs,
) -> pd.DataFrame:
    """
    Calcula una matriz de tiempo de viaje (TTM) con los parámetros dados,
    usando `r5py.TravelTimeMatrix`. La única diferencia consiste en el
    parámetro `chunk_size`; si este no es nulo, entonces el cálculo de la
    matriz se divide en *chunks* con `chunk_size` orígenes cada uno,
    permitiendo agregar una barra de progreso.
    """

    print("Calculando tiempos de viaje desde cada origen...")

    if chunk_size is None:
        return r5py.TravelTimeMatrix(
            transport_network,
            origins,
            destinations,
            **kwargs,
        )

    else:
        progress_bar = tqdm(total=len(origins))
        result = []
        for origin_chunk in chunk_gdf(origins, chunk_size):
            result.append(
                r5py.TravelTimeMatrix(
                    transport_network, origin_chunk, destinations, **kwargs
                )
            )
            progress_bar.update(len(origin_chunk))
        progress_bar.close()
        return pd.concat(result, ignore_index=True)


class TravelTimeMatrices:
    """
    Guarda los tiempos de viaje desde un conjunto de orígenes hacia varios
    conjuntos de necesidades, utilizando una sola operación de cálculo de TTM
    (Travel Time Matrix).

    Parameters
    ---
    origins: Origins
        Orígenes desde los cuales se calcularon los tiempos de viaje.
    matrices: dict[Amenity, DataFrame]
        Un diccionario cuyas llaves son las distintas `Amenities`, y los
        valores son las matrices de tiempo de viaje desde cada origen (columna
        `from_id`) a cada destino que posee la `Amenity` (columna `to_id`). El
        tiempo de viaje `travel_time` es un valor entero representando el
        tiempo de viaje en minutos, y es `None` si el tiempo de viaje es mayor
        al tiempo máximo permitido.
    """

    def __init__(
        self, origins: Origins, matrices: dict[Amenity, pd.DataFrame]
    ):

        self._origins = origins
        self._matrices = matrices

    @classmethod
    def compute(
        cls,
        origins: Origins,
        amenities: list[Amenity],
        gtfs_paths: str | Path | list[str] | list[Path],
        osm_path: str | Path,
        snap_to_network: str | bool = False,
        snap_street_mode: r5py.TransportMode | str = r5py.TransportMode.CAR,
        chunk_size: int | None = 50,
        **kwargs,
    ) -> "TravelTimeMatrices":
        """
        Calcula los tiempos de viaje y retorna una instancia de
        TravelTimeMatrices.

        Parameters
        ---
        origins : Origins
            Orígenes desde los cuales se desean calcular tiempos de viaje.
        amenities : list[Amenity]
            Conjuntos de necesidades que se desean cubrir.
        gtfs_paths: str | Path | list[str] | list[Path]
            Ruta(s) a el/los archivo(s) GTFS a utilizar para obtener viajes en
            transporte público. Puede entregarse una lista vacía si no se
            utilizará transporte público en el cálculo.
        osm_path : str or Path
            Ruta al archivo OSM desde el cual se extraerá la red de transporte
            (para caminata/bicicleta).
        snap_to_network: str or bool, default: False
            Si se desea hacer "snapping" de los orígenes/destinos a la red de
            transporte. Existen cuatro opciones posibles:
                - `False` o `"none"`: no hacer "snapping".
                - `"origins"`: solo hacer "snapping" a los orígenes.
                - `"amenities"`: solo hacer "snapping" a las
                  necesidades/destinos.
                - `True` o `"all"`: hacer "snapping" tanto a orígenes como a
                necesidades/destinos.
        snap_street_mode : str or r5py.TransportMode, default: TransportMode.CAR
            Modo de transporte que deben aceptar los caminos disponibles para
            el "snapping". Irrelevante si no se aplica "snapping".
        chunk_size : int or None, default: None
            Cantidad de orígenes que se pasarán a `r5py.TravelTimeMatrix` en
            cada *chunk*. Cada *chunk* es una ejecución nueva de
            `r5py.TravelTimeMatrix`, por lo que se pierde un poco de eficiencia
            en comparación a realizar una única llamada, pero con la ventaja de
            poder mostrar una barra de progreso. Si se desea obtener la mayor
            eficiencia (aunque sin barra de progreso), utilizar
            `chunk_size=None`.
        **kwargs
            Argumentos que serán pasados al cálculo de la TTM. Puede ser
            cualquier argumento que se pueda pasar a `r5py.RegionalTask`
            exceptuando `snap_to_network` (que se detalla más arriba).
        """

        _check_duplicate_amenities(amenities)

        # load r5py and pois
        transport_network = r5py.TransportNetwork(osm_path, gtfs_paths)

        # convert origins from areas to points
        origin_points_gdf = to_centroids(origins.h3_grid)

        # assign each amenity to its category and concatenate
        all_amenities: gpd.GeoDataFrame = pd.concat(
            [
                amenity.amenity_gdf.assign(_amenity_id=amenity.name)
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
        travel_time_matrix = compute_matrix(
            transport_network,
            origins=origin_points_gdf,
            destinations=all_amenities,
            chunk_size=chunk_size,
            **kwargs,
        )
        travel_time_matrix["_amenity_id"] = travel_time_matrix.merge(
            all_amenities[["id", "_amenity_id"]],
            left_on="to_id",
            right_on="id",
        )["_amenity_id"]

        # split matrix by amenities
        matrices = {
            amenity: travel_time_matrix[
                travel_time_matrix["_amenity_id"] == amenity.name
            ].drop(columns="_amenity_id")
            for amenity in amenities
        }

        return cls(origins, matrices)

    def group_amenity_destinations(
        self, amenity: Amenity, group_col: str
    ) -> Amenity:
        """
        Agrupa destinos dentro de una necesidad según la columna `group_col`,
        considerando el menor tiempo de viaje dentro de los destinos
        individuales como el tiempo de viaje al destino agrupado (para cada
        origen). Modifica el objeto `TravelTimeMatrices`, quitando la matriz de
        `amenity` y reemplazándola por la nueva matriz agrupada, la cual asigna
        a un nuevo objeto `Amenity`.

        Retorna el nuevo objeto `Amenity`, que es utilizado como llave para
        acceder a la nueva matriz. El peso de los destinos en el `Amenity`
        agrupado es la suma de los pesos originales que fueron asignados al
        destino; la geometría de cada nuevo destino es el centroide de los
        destinos originales.

        Parameters
        ---
        amenity : Amenity
            Necesidad cuyos destinos se buscan agrupar. Esta necesidad será
            eliminada de las llaves de `TravelTimeMatrices.matrices` y será
            reemplazada por una nueva necesidad (retornada por el método).
        group_col : str
            Columna de `amenity.amenity_gdf` que será utilizada para agrupar
            los destinos.

        Returns
        ---
        El nuevo objeto `Amenity`, que es utilizado como llave en
        `TravelTimeMatrices.matrices` para acceder a la nueva matriz creada.
        """

        if amenity not in self.matrices.keys():
            raise ValueError(
                f"{amenity} no está en el objeto TravelTimeMatrices."
            )

        assigned_ttm = self.matrices[amenity].merge(
            amenity.amenity_gdf[["id", group_col]],
            left_on="to_id",
            right_on="id",
        )
        grouped_ttm = (
            assigned_ttm.groupby(["from_id", group_col])["travel_time"]
            .min()
            .reset_index()
            .rename(columns={group_col: "to_id"})
        ).assign(to_id=lambda df: amenity.name + "/" + df["to_id"])
        grouped_gdf = (
            gpd.GeoDataFrame(
                amenity.amenity_gdf.groupby([group_col])[
                    ["weight", "geometry"]
                ].agg(
                    {
                        "weight": "sum",
                        "geometry": lambda geoms: geoms.to_crs(
                            config.projected_crs
                        )
                        .union_all()
                        .centroid,
                    }
                ),
                crs=config.projected_crs,
            )
            .to_crs(4326)
            .reset_index()
            .rename(columns={group_col: "id"})
        )
        grouped_gdf["name"] = grouped_gdf["id"]
        new_amenity = Amenity(amenity.name, grouped_gdf)
        self._matrices[new_amenity] = grouped_ttm
        self._matrices.pop(amenity)

        return new_amenity

    @property
    def origins(self) -> Origins:
        """Orígenes desde los cuales se calcularon los tiempos de viaje."""
        return self._origins

    @property
    def matrices(self) -> dict[Amenity, pd.DataFrame]:
        """Matrices de tiempo de viaje para cada necesidad."""
        return self._matrices


def merge_ttms(
    ttms: list[TravelTimeMatrices],
) -> TravelTimeMatrices:
    """
    Une dos grupos de matrices de tiempos de viaje, calculadas sobre el mismo
    conjunto de orígenes pero para distintas necesidades.

    Parameters
    ---
    ttms : list[TravelTimeMatrices]
        Lista de grupos de matrices de tiempos de viaje que se desean unir.
        Deben tener el mismo conjunto de orígenes. Si hay una misma necesidad
        calculada en dos o más elementos, se utilizará la del último elemento.

    Returns
    ---
    Nuevo objeto TravelTimeMatrices con todas las matrices.
    """

    # check all origins are equal
    origins = ttms[0].origins
    if not all(ttm.origins == origins for ttm in ttms):
        raise ValueError(
            "Todos los objetos `TimeTravelMatrices` deben generarse a partir "
            "del mismo objeto `Origins`."
        )

    # combine matrices dicts and check for duplicate amenity names (giving
    # warning)
    matrices: dict[Amenity, pd.DataFrame] = reduce(
        operator.ior, iter(ttm.matrices for ttm in ttms), {}
    )
    _check_duplicate_amenities(matrices.keys())

    return TravelTimeMatrices(origins, matrices)
