from collections.abc import Callable
from pathlib import Path
import warnings

import geopandas as gpd
import pyrosm

import xmin


class Amenity:
    """
    Clase que guarda puntos geográficos que satisfacen una necesidad
    específica.
    
    Parameters
    ---
    name : str
        Nombre de la necesidad.
    amenity_gdf : GeoDataFrame
        GeoDataFrame con los puntos que satisfacen la necesidad. Requiere
        al menos una columna `id` y una columna `geometry`; opcionalmente
        puede tener una columna `weight` si se desea ponderar un punto por
        sobre otro. Si alguna geometría de la columna `geometry` no es del
        tipo `Point`, se lanzará una advertencia y se convertirá a `Point`
        usando su centroide.
    """

    def __repr__(self):
        return f"Amenity(name={self._name})"

    def __init__(self, name: str, amenity_gdf: gpd.GeoDataFrame):
        
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

        # make sure all geometries are point geometries
        if not (self._amenity_gdf.geom_type == "Point").all():
            warnings.warn(
                "GeoDataFrame contiene geometrías que no son puntos; estas "
                "serán convertidas a puntos mediante sus centroides."
            )
            # if not, convert to centroids
            xmin._convert_geometries_to_centroids(self._amenity_gdf)
            
    @property
    def name(self) -> str:
        """Nombre de la necesidad."""
        return self._name

    @property
    def amenity_gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con los puntos que satisfacen la necesidad."""
        return self._amenity_gdf


def osm_amenity(
    name: str,
    osm_path: str | Path,
    osm_filter: dict,
    use_area_as_weight: bool = False,
    area_to_weight_function: Callable[[float], float] = lambda x: x,
):
    """
    Crea una `Amenity` con puntos generados programáticamente a partir de un
    filtro de POIs de OSM.

    Parameters
    ---
    name : str
        Nombre de la necesidad
    osm_path : Path or str
        Ruta al archivo de OSM del cual extraer los POIs.
    osm_filter : dict
        Diccionario con el filtro que se va a pasar a OSM para obtener las
        categorías correspondientes a la necesidad. Por ejemplo, para
        centros de salud, se podría utilizar el filtro `{"amenity":
        ["hospital", "clinic"]}`.
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
    """

    osm = pyrosm.OSM(str(osm_path))
    amenity_gdf: gpd.GeoDataFrame = osm.get_pois(osm_filter)
    if use_area_as_weight:
        amenity_gdf["weight"] = amenity_gdf.to_crs(
            xmin.projected_crs
        ).area.apply(area_to_weight_function)

    xmin._convert_geometries_to_centroids(amenity_gdf)

    return Amenity(name, amenity_gdf)
