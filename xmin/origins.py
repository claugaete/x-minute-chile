import geopandas as gpd
from tobler.util import h3fy

from xmin.geometry import overlay_column


class Origins:
    """
    Clase que define las regiones para las cuales se calcularán índices de
    accesibilidad. Se recomienda su construcción utilizando el método
    `Origins.create_grid()`.

    Parameters
    ---
    regions : GeoDataFrame
        GeoDataFrame con las regiones para las cuales se calcularán índices de
        accesibilidad.
    h3_resolution : int
        Resolución de las celdas de `h3_grid`.
    h3_grid : GeoDataFrame
        Grilla de celdas H3, de resolucion `h3_resolution`. Debe tener, además
        de la geometría de cada celda, columnas `id` y `population`. Pueden
        existir múltiples filas con la misma geometría (por ejemplo, una fila
        para cada grupo etario dentro de una misma zona, con su propia `id` y
        `population`).
    """

    def __init__(
        self,
        regions: gpd.GeoDataFrame,
        h3_resolution: int,
        h3_grid: gpd.GeoDataFrame,
    ):

        self._regions = regions.to_crs(4326)
        self._h3_resolution = h3_resolution
        self._h3_grid = h3_grid

    @classmethod
    def create_grid(
        cls,
        regions: gpd.GeoDataFrame,
        h3_resolution: int,
        population_gdf: gpd.GeoDataFrame | None = None,
    ) -> "Origins":
        """
        Crea un objeto Origins, dividiendo una región en celdas de H3 y
        asignándole una población a cada una.

        Parameters
        ---
        regions : GeoDataFrame
            GeoDataFrame con las regiones para las cuales se calcularán índices
            de accesibilidad.
        h3_resolution : int
            Resolución de H3 que se utilizará para dividir las celdas. Más
            información en https://h3geo.org/docs/core-library/restable/
        population_gdf : GeoDataFrame or None, default: None
            GeoDataFrame con datos de población. Debe tener por lo menos las
            columnas `population` y `geometry` (que debe contener geometrías de
            tipo `Point`, `Polygon` o `MultiPolygon`). No es necesario que
            cubra el mismo área que `bounds`. Si no se recibe un GeoDataFrame,
            se asumirá una población idéntica (1) en cada origen.

        Returns
        ---
        Objeto Origins con la información solicitada.
        """

        regions = regions.to_crs(4326)
        h3_grid = h3fy(regions, resolution=h3_resolution)

        if population_gdf is None:
            h3_grid["population"] = 1
        else:
            if not (
                "population" in population_gdf.columns
                and "geometry" in population_gdf.columns
            ):
                raise ValueError(
                    "`population_gdf` debe tener al menos las columnas "
                    "`population` y `geometry`"
                )
            h3_grid["population"] = overlay_column(
                h3_grid, population_gdf, "population"
            )
        h3_grid = h3_grid.rename_axis("id").reset_index()

        return cls(regions, h3_resolution, h3_grid)

    @property
    def regions(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con los límites de las regiones."""
        return self._regions

    @property
    def h3_grid(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con las celdas hexagonales de las regiones."""
        return self._h3_grid

    @property
    def h3_resolution(self) -> gpd.GeoDataFrame:
        """Nivel de resolución de la grilla de celdas hexagonales."""
        return self._h3_resolution
