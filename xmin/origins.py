import geopandas as gpd
from shapely import Polygon
from tobler.util import h3fy

import xmin


class Origins:
    """
    Clase que define las regiones para las cuales se calcularán índices de
    accesibilidad.
    """

    def __init__(
        self,
        bounds: Polygon,
        h3_level: int = 8,
        population_gdf: gpd.GeoDataFrame | None = None,
    ):
        """
        Parameters
        ---
        bounds : Polygon or MultiPolygon
            Polígono (o colección de polígonos) que define los límites de las
            regiones para las cuales se calcularán índices de accesibilidad
        h3_level : int, default: 8
            Nivel de H3 que se utilizará para dividir las celdas. Más
            información en https://h3geo.org/docs/core-library/restable/
        population_gdf : GeoDataFrame or None, default: None
            GeoDataFrame con datos de población. Debe tener por lo menos las
            columnas `population` y `geometry`. No es necesario que cubra el
            mismo área que `bounds`. Si no se recibe un GeoDataFrame, no se
            cargarán datos de población en los orígenes y no se permitirá el
            uso de índices que necesiten esa información.
        """

        self.bounds = bounds
        self.h3_level = h3_level

        self.h3_grid = h3fy(bounds, resolution=self.h3_level)
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

    def _overlay_population_h3(
        self, population_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Intersecta la población de una ciudad (normalmente dividida en sus
        manzanas) con su grilla H3, dividiendo aquellas zonas que caen en
        múltiples celdas de H3, y repartiendo la población propocional al área
        de la zona que cae en cada celda. Retorna la grilla de H3 con
        poblaciones asignadas.
        """

        # add column with area of each zone
        population_gdf = population_gdf.assign(
            area=population_gdf.to_crs(xmin.projected_crs).geometry.area
        )

        # overlay zones with H3 cells, splitting zones that fall between
        # multiple cells into multiple "fragments"
        population_gdf_split = population_gdf.overlay(
            self.h3_grid.reset_index()
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

        self.h3_grid["population"] = h3_population.fillna(0)
