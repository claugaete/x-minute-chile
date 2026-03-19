from collections.abc import Callable
import warnings

import geopandas as gpd
import pyrosm

import xmin


class Amenity:
    """
    Clase que guarda puntos geográficos que satisfacen una necesidad
    específica.
    """

    def __init__(self, name: str, amenity_gdf: gpd.GeoDataFrame | None):

        self.name = name
        self.amenity_gdf = amenity_gdf

    def _convert_geometries_to_centroids(self):
        """
        Convierte las geometrías de `self.amenity_gdf` que no son puntos,
        cambiándolas por sus centroides.
        """
        original_crs = self.amenity_gdf.crs
        self.amenity_gdf.geometry = self.amenity_gdf.geometry.to_crs(
            xmin.projected_crs
        ).centroid.to_crs(original_crs)
        
    def get_pois(self, osm: pyrosm.OSM):
        """
        Cargar POIs para la necesidad a partir de un objeto de OpenStreetMap.
        No-op por defecto (para Amenities que ya tengan sus elementos
        pre-cargados).
        """
        pass


class CustomAmenity(Amenity):
    """
    Clase para guardar una `Amenity` con un conjunto de puntos creado
    manualmente.
    """

    def __init__(self, name: str, amenity_gdf: gpd.GeoDataFrame):
        """
        Parameters
        ---
        amenity_gdf: GeoDataFrame
            GeoDataFrame con los puntos que satisfacen la necesidad. Requiere
            al menos una columna `id` y una columna `geometry`; opcionalmente
            puede tener una columna `weight` si se desea ponderar un punto por
            sobre otro. Si alguna geometría de la columna `geometry` no es del
            tipo `Point`, se lanzará una advertencia y se convertirá a `Point`
            usando su centroide.
        """

        super().__init__(name, amenity_gdf)

        if not (
            "id" in self.amenity_gdf.columns
            and "geometry" in self.amenity_gdf.columns
        ):
            raise ValueError(
                "`amenity_gdf` debe tener al menos las columnas `id` y "
                "`geometry`"
            )

        # make sure all geometries are point geometries
        if not (self.amenity_gdf.geom_type == "Point").all():
            warnings.warn(
                "GeoDataFrame contiene geometrías que no son puntos; estas "
                "serán convertidas a puntos mediante sus centroides."
            )
            # if not, convert to centroids
            self._convert_geometries_to_centroids()


class OsmAmenity(Amenity):
    """
    Clase para guardar una `Amenity` con puntos generados programáticamente a
    partir de un filtro de POIs de OSM. Cada vez que la clase es utilizada para
    calcular un índice en un polígono, se actualiza `amenity_gdf` para incluir
    los POIs filtrados dentro del polígono.
    """

    def __init__(
        self,
        name: str,
        osm_filter: dict,
        use_area_as_weight: bool = False,
        area_to_weight_function: Callable[[float], float] = lambda x: x,
    ):
        """
        Parameters
        ---
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

        super().__init__(name, amenity_gdf=None)
        self.osm_filter = osm_filter
        self.use_area_as_weight = use_area_as_weight
        self.area_to_weight_function = area_to_weight_function

    def get_pois(self, osm: pyrosm.OSM):

        self.amenity_gdf: gpd.GeoDataFrame = osm.get_pois(self.osm_filter)
        if self.use_area_as_weight:
            self.amenity_gdf["weight"] = self.amenity_gdf.to_crs(
                xmin.projected_crs
            ).area.apply(self.area_to_weight_function)
        self._convert_geometries_to_centroids()
