from collections.abc import Callable
from pathlib import Path
import warnings

import geopandas as gpd
import quackosm as qosm
from shapely.geometry.base import BaseGeometry

from .config import config
from .geometry import to_centroids


class Amenity:
    """
    Clase que guarda puntos geográficos que satisfacen una necesidad
    específica.

    Parameters
    ---
    name : str
        Nombre de la necesidad.
    amenity_gdf : GeoDataFrame
        GeoDataFrame con los puntos que satisfacen la necesidad. Requiere al
        menos una columna `id` y una columna `geometry`; opcionalmente puede
        tener las siguientes columnas:

        - `weight` si se desea ponderar un punto por sobre otro
            (si no existe esta columna, se asumirá un peso 1 para todos los
            puntos).
        - `name` si se desea asignar un nombre que sea considerado por algunos
            métodos de visualización interactiva (que aparezca el nombre al
            hacer "hover" por encima del punto).

        Si alguna geometría de la columna `geometry` no es del tipo
        `Point`, se lanzará una advertencia y se convertirá a `Point` usando su
        centroide.
    bounds : BaseGeometry or None, default: None
        Si se especifica, se filtrará `amenity_gdf` para que solo contenga las
        filas que intersectan con el polígono dado.
    """

    def __repr__(self):
        return f"Amenity(name={self._name})"

    def __init__(
        self,
        name: str,
        amenity_gdf: gpd.GeoDataFrame,
        bounds: BaseGeometry | None = None,
    ):

        self._name = name
        self._amenity_gdf = amenity_gdf.to_crs(4326)

        if not (
            "id" in self._amenity_gdf.columns
            and "geometry" in self._amenity_gdf.columns
        ):
            raise ValueError(
                "`amenity_gdf` debe tener al menos las columnas `id` y "
                "`geometry`"
            )

        if not self._amenity_gdf["id"].is_unique:
            raise ValueError("IDs deben ser únicas en `amenity_gdf`")

        if bounds:
            self._amenity_gdf = self._amenity_gdf[
                self._amenity_gdf.intersects(bounds)
            ]

        if "weight" not in self._amenity_gdf.columns:
            self._amenity_gdf["weight"] = 1

        # add amenity name to each id (in case a destination from a different
        # amenity shares the same id)
        self._amenity_gdf["id"] = (
            self._name + "/" + self._amenity_gdf["id"].astype(str)
        )

        # make sure all geometries are point geometries
        if not (self._amenity_gdf.geom_type == "Point").all():
            warnings.warn(
                "GeoDataFrame contiene geometrías que no son puntos; estas "
                "serán convertidas a puntos mediante sus centroides."
            )
            # if not, convert to centroids
            self._amenity_gdf = to_centroids(self._amenity_gdf)
    
    @classmethod        
    def from_osm(
        cls,
        name: str,
        osm_path: str | Path,
        osm_filter: dict,
        keep_all_tags: bool | list[str] = True,
        bounds: BaseGeometry | None = None,
        use_area_as_weight: bool = False,
        area_to_weight_function: Callable[[float], float] = lambda x: x,
    ) -> "Amenity":
        """
        Crea una `Amenity` con puntos generados programáticamente a partir de
        un filtro de POIs de OSM.

        Parameters
        ---
        name : str
            Nombre de la necesidad
        osm_path : Path or str
            Ruta al archivo PBF de OSM del cual extraer los POIs.
        osm_filter : dict
            Diccionario con el filtro que se va a pasar a OSM para obtener las
            categorías correspondientes a la necesidad. Por ejemplo, para
            centros de salud, se podría utilizar el filtro `{"amenity":
            ["hospital", "clinic"]}`. Para más información sobre categorías
            existentes, visitar
            https://wiki.openstreetmap.org/wiki/Map_features.
        keep_all_tags : bool, default: True
            Si es verdadero, se guarda cada etiqueta en su propia columna del
            GeoDataFrame. Si es falso, se eliminan las etiquetas asociadas a
            cada POI (lo cual hace el GeoDataFrame más ligero). Si es una
            lista, solo se guardan las etiquetas incluidas en la lista.
        bounds : BaseGeometry or None, default: None
            Polígono dentro del cual se desean buscar los POIs en OSM. Si no se
            especifica, se buscarán puntos en toda la geometría del archivo
            PBF.
        use_area_as_weight : bool, default: False
            Booleano indicando si se utiliza el área de los POIs obtenidos en
            OSM como referencia para obtener el peso de cada uno.
        area_to_weight_function : (float) -> float, default: identity
            Función a utilizar para convertir el área de cada POI en su peso.
            Es importante considerar que algunos POIs podrían ser puntos (no
            polígonos), y por ende tener área 0. Podría ser necesario
            considerar este caso borde a la hora de convertir áreas a pesos (si
            se utiliza la función identidad, estos puntos tendrían peso 0 y
            podrían no afectar al resultado final).

        Returns
        ---
        Un objeto `Amenity` correspondiente a la necesidad.
        """

        amenity_gdf: gpd.GeoDataFrame = (
            qosm.convert_pbf_to_geodataframe(
                osm_path,
                tags_filter=osm_filter,
                working_directory=config.quackosm_working_directory,
                geometry_filter=bounds,
                keep_all_tags=False if not keep_all_tags else True,
                explode_tags=True,
            )
            .rename_axis("id")
            .reset_index()
        )
        if isinstance(keep_all_tags, list):
            amenity_gdf = amenity_gdf[["id"] + keep_all_tags + ["geometry"]]
        if use_area_as_weight:
            amenity_gdf["weight"] = amenity_gdf.to_crs(
                config.projected_crs
            ).area.apply(area_to_weight_function)

        return cls(name, to_centroids(amenity_gdf))

    @property
    def name(self) -> str:
        """Nombre de la necesidad."""
        return self._name

    @property
    def amenity_gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con los puntos que satisfacen la necesidad."""
        return self._amenity_gdf
