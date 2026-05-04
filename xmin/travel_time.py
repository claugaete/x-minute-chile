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
    dropna: bool,
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
            ttm = r5py.TravelTimeMatrix(
                transport_network, origin_chunk, destinations, **kwargs
            )
            if dropna:
                ttm = ttm.dropna()
            result.append(ttm)
            del ttm
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
    origins : Origins
        Orígenes desde los cuales se calcularon los tiempos de viaje.
    amenities : list[Amenity]
        Las distintas necesidades para las cuales se calculan tiempos de viaje.
    matrices : dict[Amenity, DataFrame]
        Un diccionario cuyas llaves son los nombres de las distintas
        `Amenities`, y los valores son las matrices de tiempo de viaje desde
        cada origen (columna `from_id`) a cada destino que posee la `Amenity`
        (columna `to_id`). El tiempo de viaje `travel_time` es un valor entero
        representando el tiempo de viaje en minutos, y es `None` si el tiempo
        de viaje es mayor al tiempo máximo permitido.
    """

    def __init__(
        self,
        origins: Origins,
        amenities: list[Amenity],
        matrices: dict[str, pd.DataFrame],
    ):

        self._origins = origins
        self._amenities = {amenity.name: amenity for amenity in amenities}
        self._matrices = matrices

    @property
    def origins(self) -> Origins:
        """Orígenes desde los cuales se calcularon los tiempos de viaje."""
        return self._origins

    @property
    def amenities(self) -> dict[str, Amenity]:
        """Necesidades para las cuales se calcularon los tiempos de viaje."""
        return self._amenities

    @property
    def matrices(self) -> dict[str, pd.DataFrame]:
        """Matrices de tiempo de viaje para cada necesidad."""
        return self._matrices

    @classmethod
    def compute(
        cls,
        origins: Origins,
        amenities: list[Amenity],
        gtfs_paths: str | Path | list[str] | list[Path],
        osm_path: str | Path,
        snap_to_network: str | bool = False,
        snap_street_mode: r5py.TransportMode | str = r5py.TransportMode.CAR,
        chunk_size: int | None = 32,
        dropna: bool = True,
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
        chunk_size : int or None, default: 32
            Cantidad de orígenes que se pasarán a `r5py.TravelTimeMatrix` en
            cada *chunk*. Cada *chunk* es una ejecución nueva de
            `r5py.TravelTimeMatrix`, por lo que se pierde un poco de eficiencia
            en comparación a realizar una única llamada, pero con la ventaja de
            poder mostrar una barra de progreso. Si se desea obtener la mayor
            eficiencia (aunque sin barra de progreso), utilizar
            `chunk_size=None`.
        dropna : bool, default: True
            Si es verdadero, elimina filas con tiempos de viaje nulos (que son
            mayores al tiempo máximo permitido por el cálculo). Esto reduce en
            varios órdenes de magnitud el tamaño del DataFrame resultante
            (especialmente si se tiene un área urbana grande y tiempos de viaje
            cortos).
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
            dropna=dropna,
            **kwargs,
        )
        travel_time_matrix["_amenity_id"] = travel_time_matrix.merge(
            all_amenities[["id", "_amenity_id"]],
            left_on="to_id",
            right_on="id",
        )["_amenity_id"]

        # split matrix by amenities
        matrices = {
            amenity.name: travel_time_matrix[
                travel_time_matrix["_amenity_id"] == amenity.name
            ].drop(columns="_amenity_id")
            for amenity in amenities
        }

        return cls(origins, amenities, matrices)

    def group_amenity_destinations(self, amenity_name: str, group_col: str):
        """
        Agrupa destinos dentro de una necesidad según la columna `group_col`,
        considerando el menor tiempo de viaje dentro de los destinos
        individuales como el tiempo de viaje al destino agrupado (para cada
        origen). Modifica el objeto `TravelTimeMatrices`, quitando la matriz de
        `amenity_name` y reemplazándola por la nueva matriz agrupada.

        También guarda en `TravelTimeMatrices.amenities` el nuevo objeto
        `Amenity` agrupado. El peso de los destinos en el `Amenity` agrupado es
        la suma de los pesos originales que fueron asignados al destino; la
        geometría de cada nuevo destino es el centroide de los destinos
        originales.

        Parameters
        ---
        amenity_name : str
            Nombre de la necesidad cuyos destinos se buscan agrupar. Esta
            necesidad será eliminada de `TravelTimeMatrices.amenities` y será
            reemplazada por una nueva necesidad.
        group_col : str
            Columna de `amenity.amenity_gdf` que será utilizada para agrupar
            los destinos.
        """

        if amenity_name not in self.matrices.keys():
            raise ValueError(
                f"{amenity_name} no está en el objeto TravelTimeMatrices."
            )

        amenity = self._amenities[amenity_name]

        assigned_ttm = self.matrices[amenity_name].merge(
            amenity[["id", group_col]],
            left_on="to_id",
            right_on="id",
        )
        grouped_ttm = (
            assigned_ttm.groupby(["from_id", group_col])["travel_time"]
            .min()
            .reset_index()
            .rename(columns={group_col: "to_id"})
        ).assign(to_id=lambda df: amenity_name + "/" + df["to_id"])
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
        self._matrices[amenity_name] = grouped_ttm
        self._amenities[amenity_name] = Amenity(
            amenity_name, grouped_gdf, add_name_to_id=False
        )


def _check_origin_equality(ttms_list: list[TravelTimeMatrices]):
    """Raise error if origin H3 resolutions are different; warn if origin regions are
    different. Returns the first `Origins` object."""

    # check resolution is equal
    h3_resolutions = {ttm.origins.h3_resolution for ttm in ttms_list}
    if len(h3_resolutions) > 1:
        raise ValueError(
            "Todos los objetos `TimeTravelMatrices` deben tener la misma "
            "resolución en su grilla de orígenes. Se tienen las siguientes "
            f"resoluciones: {', '.join(h3_resolutions)}."
        )

    # warn if origins are not equal
    sample_origins = ttms_list[0].origins
    if not all(
        ttm.origins.regions.equals(sample_origins.regions) for ttm in ttms_list
    ):
        warnings.warn(
            "No todos los objetos `TimeTravelMatrices` fueron generados a "
            "partir de los mismos orígenes; esto podría causar resultados "
            "inesperados."
        )

    return sample_origins


def merge_amenities(
    ttms: list[TravelTimeMatrices],
) -> TravelTimeMatrices:
    """
    Une varios grupos de matrices de tiempos de viaje, calculadas sobre el
    mismo conjunto de orígenes pero para distintas necesidades.

    Parameters
    ---
    ttms : list[TravelTimeMatrices]
        Lista de grupos de matrices de tiempos de viaje que se desean unir.
        Deben tener el mismo conjunto de orígenes. Si hay dos necesidades que
        comparten el mismo nombre en distintos grupos de matrices, se utilizará
        la del último grupo.

    Returns
    ---
    Nuevo objeto TravelTimeMatrices con todas las matrices.
    """

    origins = _check_origin_equality(ttms)

    # combine matrices and amenities dicts
    matrices: dict[str, pd.DataFrame] = reduce(
        operator.ior, iter(ttm.matrices for ttm in ttms), {}
    )
    amenities: dict[str, Amenity] = reduce(
        operator.ior, iter(ttm.amenities for ttm in ttms), {}
    )

    return TravelTimeMatrices(origins, amenities, matrices)


def merge_populations(
    segmented_ttms: dict[str, TravelTimeMatrices],
) -> TravelTimeMatrices:
    """
    Une varios grupos de matrices de tiempos de viaje, calculadas sobre el
    mismo conjunto de orígenes y el mismo conjunto de necesidades, pero con una
    población distinta en cada celda de los orígenes (representando distintos
    grupos poblacionales).

    Parameters
    ---
    ttms : dict[str, TravelTimeMatrices]
        Grupos de matrices de tiempos de viaje que se desean unir.
        Deben tener el mismo conjunto de orígenes (solo diferenciándose en su
        población), y el mismo conjunto de necesidades a cubrir. Los destinos
        asociados a cada necesidad pueden ser distintos para distintos grupos
        poblacionales (por ejemplo, adultos mayores que tienen acceso a
        servicios de salud solo para ellos).

        Cada objeto `TravelTimeMatrices` está asociado al nombre del grupo
        poblacional al que corresponde; este nombre será agregado a la ID de
        cada celda de origen del grupo (para diferenciarla de los demás
        grupos).

    Returns
    ---
    Nuevo objeto TravelTimeMatrices con todas las matrices.
    """

    ttms = list(segmented_ttms.values())

    origins = _check_origin_equality(ttms)

    # warn if amenities are not equal
    amenity_names = ttms[0].amenities.keys()
    if not all(ttm.amenities.keys() == amenity_names for ttm in ttms):
        warnings.warn(
            "No todos los objetos `TimeTravelMatrices` consideran las mismas "
            "necesidades; esto podría causar resultados inesperados."
        )
        amenity_names = set.union(ttm.amenities.keys() for ttm in ttms)

    # iterate over TravelTimeMatrices objects, combining:
    # - origin grids (adding group prefix to each one)
    # - TTMs (adding group prefix to each origin)
    # - amenities (duplicates will be dropped later)
    new_grids = []
    new_ttms_matrices = {name: [] for name in amenity_names}
    new_amenity_gdfs = {name: [] for name in amenity_names}
    for prefix, ttm in segmented_ttms.items():
        new_grid = origins.h3_grid.assign(id=f"{prefix}/" + origins.h3_grid.id)
        new_grids.append(new_grid)
        for name, matrix in ttm.matrices.items():
            new_ttms_matrices[name].append(
                matrix.assign(from_id=f"{prefix}/" + matrix.from_id)
            )
        for name, amenity in ttm.amenities.items():
            new_amenity_gdfs[name].append(amenity.amenity_gdf)

    # concatenate new origin grid (with all groups), new matrices for each
    # amenity, and new amenities (in case different groups had different
    # destinations within an amenity)
    new_grid = gpd.GeoDataFrame(pd.concat(new_grids), crs=4326)
    new_origins = Origins(origins.regions, origins.h3_resolution, new_grid)
    new_ttms_matrices_concat = {
        name: pd.concat(matrices)
        for name, matrices in new_ttms_matrices.items()
    }
    new_amenities = [
        Amenity(
            name,
            gpd.GeoDataFrame(
                pd.concat(amenity_gdfs).drop_duplicates("id"), crs=4326
            ),
            add_name_to_id=False,
        )
        for name, amenity_gdfs in new_amenity_gdfs.items()
    ]

    return TravelTimeMatrices(
        new_origins, new_amenities, new_ttms_matrices_concat
    )
