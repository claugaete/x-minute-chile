import geopandas as gpd
from tobler.util import h3fy

import xmin


class Origins:
    """
    Clase que define las regiones para las cuales se calcularán índices de
    accesibilidad.

    Parameters
    ---
    regions : GeoDataFrame
        GeoDataFrame con las regiones para las cuales se calcularán índices de
        accesibilidad.
    h3_resolution : int
        Resolución de H3 que se utilizará para dividir las celdas. Más
        información en https://h3geo.org/docs/core-library/restable/
    population_gdf : GeoDataFrame or None, default: None
        GeoDataFrame con datos de población. Debe tener por lo menos las
        columnas `population` y `geometry`. No es necesario que cubra el mismo
        área que `bounds`. Si no se recibe un GeoDataFrame, no se cargarán
        datos de población en los orígenes y los índices que utilicen esa
        información asumirán la misma población en cada celda.
    """

    def __init__(
        self,
        regions: gpd.GeoDataFrame,
        h3_resolution: int,
        population_gdf: gpd.GeoDataFrame | None = None,
    ):

        self._regions = regions.to_crs(4326)
        self._h3_resolution = h3_resolution

        self._h3_grid = h3fy(regions, resolution=h3_resolution)
        if population_gdf is not None:
            if not (
                "population" in population_gdf.columns
                and "geometry" in population_gdf.columns
            ):
                raise ValueError(
                    "`population_gdf` debe tener al menos las columnas "
                    "`population` y `geometry`"
                )
            self._overlay_population_h3(population_gdf)
        self._h3_grid = self._h3_grid.rename_axis("id").reset_index()

    def _overlay_population_h3(self, population_gdf: gpd.GeoDataFrame):
        """
        Intersecta la población de una ciudad (normalmente dividida en sus
        manzanas) con su grilla H3, dividiendo aquellas zonas que caen en
        múltiples celdas de H3, y repartiendo la población propocional al área
        de la zona que cae en cada celda. Modifica la grilla de H3, agregándole
        la columna `population`.
        """

        # add column with area of each zone
        population_gdf = population_gdf.to_crs(4326).assign(
            area=population_gdf.to_crs(xmin.projected_crs).geometry.area
        )

        # overlay zones with H3 cells, splitting zones that fall between
        # multiple cells into multiple "fragments"
        population_gdf_split = population_gdf.overlay(
            self._h3_grid.reset_index(),
            keep_geom_type=True
        )

        # assign population to each zone fragment, proportional to the area of
        # the original zone that the split zone has
        population_gdf_split["population_split"] = (
            population_gdf_split["population"]
            * population_gdf_split.to_crs(xmin.projected_crs).geometry.area
            / population_gdf_split["area"]
        )

        # group by H3 cell to get the total population in each cell
        h3_population = population_gdf_split.groupby("hex_id")[
            "population_split"
        ].sum()

        self._h3_grid["population"] = h3_population.fillna(0)

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
