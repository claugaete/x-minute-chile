from typing import TypeVar
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from .amenities import Amenity
from .geometry import to_centroids
from .indices import IndexFunction
from .origins import Origins
from .travel_time import TravelTimeMatrices
from .visualization import AccessibilityVisualizer

V = TypeVar("V")


def _convert_amenities_to_names(d: dict[Amenity | str, V]) -> dict[str, V]:
    """Toma un diccionario y convierte las llaves que son `Amenity` a `str`."""
    return {(k.name if isinstance(k, Amenity) else k): v for k, v in d.items()}


class AccessibilityRatings:
    """
    Guarda ratings de accesibilidad, calculando índices para distintas
    necesidades según los tiempos de viaje desde cada origen a los distintos
    destinos, y ponderándolos para obtener un índice de accesibilidad general.

    Parameters
    ---
    origins : Origins
        Orígenes desde los cuales se calcularon los índices.
    amenities : list[Amenity]
        Las distintas necesidades usadas como destinos en el cálculo de los
        índices.
    weights : dict[Amenity, float]
        Pesos relativos de las necesidades que fueron usados para la
        ponderación de la accesibilidad global.
    gdf : GeoDataFrame
        GeoDataFrame con los ratings de accesibilidad calculados para cada
        necesidad (el nombre de la columna corresponde a Amenity.name); y con
        una columna `total` con la accesibilidad global (ponderada).
    """

    def __init__(
        self,
        origins: Origins,
        amenities: list[Amenity],
        weights: dict[Amenity | str, float],
        gdf: gpd.GeoDataFrame,
    ):
        self._origins = origins
        self._amenities = {amenity.name: amenity for amenity in amenities}
        self._weights = _convert_amenities_to_names(weights)
        self._gdf = gdf
        self._visualize = AccessibilityVisualizer(
            self._gdf, self._origins, self._amenities, self._weights
        )

    @property
    def origins(self) -> Origins:
        """Orígenes desde los cuales se calculó la accesibilidad."""
        return self._origins

    @property
    def amenities(self) -> dict[str, Amenity]:
        """Necesidades para las cuales se calculó la accesibilidad."""
        return self._amenities

    @property
    def weights(self) -> dict[Amenity, float]:
        """Pesos relativos de cada necesidad para el cálculo de índices."""
        return self._weights

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con los ratings de accesibilidad calculados."""
        return self._gdf

    @property
    def visualize(self) -> AccessibilityVisualizer:
        """Módulo con distintas opciones de visualización para los ratings."""
        return self._visualize

    @classmethod
    def compute(
        cls,
        travel_time_matrices: TravelTimeMatrices,
        index_function: IndexFunction | dict[Amenity | str, IndexFunction],
        weights: dict[Amenity | str, float] | None = None,
    ) -> "AccessibilityRatings":
        """
        Calcula los índices de accesibilidad y retorna una instancia de
        AccessibilityRatings.

        Parameters
        ---
        time_travel_matrices : TravelTimeMatrices
            Objeto que guarda las matrices de viaje desde cada origen hacia los
            distintos destinos.
        index_function : IndexFunction or dict[Amenity | str, IndexFunction]
            Funcion(es) para calcular los índices de accesibilidad de las
            distintas necesidades. Si se recibe una IndexFunction, se aplicará
            la misma para todas las Amenities. Si se recibe un diccionario, se
            asocia cada necesidad con su función a utilizar para el cálculo de
            su índice. Las llaves del diccionario pueden ser los objetos
            `Amenity` o sus nombres.
        weights : dict[Amenity | str, float] or None, default: None
            Pesos relativos de las distintas necesidades. Las llaves pueden ser
            los objetos `Amenity` o sus nombres. El índice final se calcula
            ponderando los índices de las distintas necesidades utilizando
            estos pesos. Si no se entregan pesos, todos los índices tendrán el
            mismo peso.
        """

        origins = travel_time_matrices.origins
        amenities = travel_time_matrices.amenities
        matrices = travel_time_matrices.matrices

        population = origins.h3_grid.set_index("id")["population"]
        ratings_gdf = origins.h3_grid.set_index("id")[["geometry"]]
        amenities_with_index: list[str] = []

        # replace Amenity keys with strings if index_function is a dict
        # if not, assign same function to all amenities
        if isinstance(index_function, dict):
            index_function = _convert_amenities_to_names(index_function)
        else:
            index_function = {
                name: index_function for name in amenities.keys()
            }

        # replace Amenity keys with strings if weights is a dict
        # if not, assign same weight to all amenities that have an index
        # function
        if isinstance(weights, dict):
            weights = _convert_amenities_to_names(weights)
        else:
            weights = {k: 1 for k in index_function.keys()}

        # calcular el índice particular de cada necesidad
        print("Calculando índices para cada necesidad...")
        for name, amenity in tqdm(amenities.items()):
            ttm = matrices[name]
            amenity_gdf = amenity.amenity_gdf.set_index("id")
            current_index = index_function.get(name)
            if current_index is None:
                warnings.warn(
                    f'Amenity "{name}" no tiene una función de índice '
                    "asociada en `index_function`. No se considerará la "
                    "necesidad en el cálculo del índice."
                )
            else:
                ratings_gdf[name] = current_index.calculate_index(
                    ttm, population, amenity_gdf["weight"]
                )
                amenities_with_index.append(name)
        ratings_gdf = ratings_gdf.fillna(0)

        # arreglamos discrepancias entre índices calculados y pesos recibidos
        amenities_with_no_weight = set(amenities_with_index).difference(
            weights.keys()
        )
        if amenities_with_no_weight:
            warnings.warn(
                "Las siguientes Amenities no tienen un peso asociado: "
                + ", ".join(name for name in amenities_with_no_weight)
                + ". Su peso será considerado como 0. Para evitar este "
                "warning, asigna un peso a las necesidades (puede ser 0 si no "
                "deseas incluirlas en el cálculo final)."
            )
        for amenity in amenities_with_no_weight:
            weights[amenity] = 0

        weights_with_no_index = set(weights.keys()).difference(
            amenities_with_index
        )
        if weights_with_no_index:
            warnings.warn(
                "Las siguientes Amenities tienen un peso asociado, pero no se "
                "les calculó un índice: "
                + ", ".join(name for name in weights_with_no_index)
                + ". Estas Amenities serán ignoradas. Para evitar este "
                "warning, asigna un índice a cada necesidad."
            )
        for name in weights_with_no_index:
            weights.pop(name, None)

        # ponderamos índices para obtener el final
        weight_sum = sum(weights.values())
        ratings_gdf["total"] = 0

        for name, weight in weights.items():
            ratings_gdf["total"] += ratings_gdf[name] * weight / weight_sum

        return cls(origins, amenities.values(), weights, ratings_gdf)

    def crop(
        self,
        new_bounds: gpd.GeoDataFrame | BaseGeometry,
    ) -> "AccessibilityRatings":
        """
        Realiza un cropping al GeoDataFrame con los valores de accesibilidad,
        considerando solo los orígenes incluídos dentro de la nueva frontera.

        Parameters
        ---
        new_bounds : GeoDataFrame or BaseGeometry
            Nuevos límites a considerar. Puede ser un GeoDataFrame o un
            polígono. Si los nuevos límites no están completamente contenidos
            en los límites antiguos, solo se considerará la intersección entre
            ambos.

        Returns
        ---
        Un nuevo objeto `AccessibilityRatings` solo incluyendo los orígenes
        contenidos dentro de los nuevos límites.
        """

        new_bounds_too_large_text = (
            "Área nueva no está completamente contenida en el área original; "
            "solo se considerará la porción del área nueva que está contenida "
            "en el área original."
        )

        old_regions = self.origins.regions
        old_regions_union = old_regions.union_all()

        if isinstance(new_bounds, gpd.GeoDataFrame):
            if not new_bounds.union_all().within(old_regions_union):
                warnings.warn(new_bounds_too_large_text)
                new_regions = new_bounds.to_crs(4326).clip(old_regions_union)
            else:
                new_regions = new_bounds
        else:
            if not new_bounds.within(old_regions_union):
                warnings.warn(new_bounds_too_large_text)
            new_regions = old_regions.clip(new_bounds)

        new_origins = Origins(
            new_regions,
            h3_resolution=self.origins.h3_resolution,
            h3_grid=self.origins.h3_grid[
                to_centroids(self.origins.h3_grid).within(
                    new_regions.union_all()
                )
            ],
        )

        new_grid_indexed = new_origins.h3_grid.set_index("id")

        # a las celdas nuevas les asignamos los ratings antiguos
        new_ratings = self.gdf.reindex(new_grid_indexed.index)

        return AccessibilityRatings(
            new_origins, self.amenities.values(), self.weights, new_ratings
        )

    def aggregate(
        self, new_resolution: int, weighted: bool = False
    ) -> "AccessibilityRatings":
        """
        Agrega los valores de accesibilidad calculados para una resolución
        menor de H3; esto reduce la dependencia a los orígenes escogidos para
        calcular la accesibilidad en cada celda.

        Nota: Las celdas de un nivel no están completamente contenidas en una
        única celda de resolución inferior; para efectos de esta función, la
        accesibilidad promedio de una celda considera todas las subceldas cuyo
        **centroide** cae en la celda. De forma análoga, la población asignada
        a cada celda es la suma de las poblaciones de las subceldas cuyos
        centroides caen en la celda. Esta población podría diferir levemente de
        la población obtenida al crear un nuevo objeto Origins con la
        resolución nueva y usando el `population_gdf` original. Para más
        información respecto a esta discrepancia, ver
        https://observablehq.com/@nrabinowitz/h3-hierarchical-non-containment.

        Parameters
        ---
        new_resolution : int
            La nueva resolución de H3; no puede ser mayor a la resolución
            actual.
        weighted : bool, default: False
            Si ponderar la accesibilidad de cada celda según su población.

        Returns
        ---
        Un nuevo objeto AccessibilityRatings con la resolución reducida.
        """

        if new_resolution > self.origins.h3_resolution:
            raise ValueError(
                f"Nueva resolución ({new_resolution}) no puede ser mayor a la "
                f"resolución actual ({self.origins.h3_resolution})."
            )

        old_origins = self.origins.h3_grid

        # get new origins (no population data yet)
        new_origins_no_population = Origins.create_grid(
            self.origins.regions, new_resolution
        )

        # map each old cell to the new cell where its centroid falls
        left_gdf = to_centroids(self.gdf)
        joined = (
            left_gdf.sjoin(
                new_origins_no_population.h3_grid[["id", "geometry"]],
                how="right",
                predicate="within",
                lsuffix="old",
                rsuffix=None,
            )
            .drop_duplicates("id_old")
            .set_index("id_old")
        )
        cell_map = joined["id"]

        # assign new population using cell map
        new_population = (
            old_origins.assign(id_new=old_origins["id"].map(cell_map))
            .groupby("id_new")["population"]
            .sum()
        )
        grid_with_population = (
            new_origins_no_population.h3_grid.set_index("id")
            .assign(population=new_population)
            .reset_index()
        )
        new_origins = Origins(
            new_origins_no_population.regions,
            h3_resolution=new_origins_no_population.h3_resolution,
            h3_grid=grid_with_population,
        )

        # we average the accessibility values
        cols_to_agg = [col for col in self.gdf.columns if col != "geometry"]
        if not weighted:
            averaged_accs = joined.groupby(["id", "geometry"])[
                cols_to_agg
            ].mean()
        else:

            old_population = old_origins.set_index("id")["population"]

            def weighted_average(df: pd.DataFrame):
                weights = old_population.loc[df.index]
                if weights.sum() == 0:
                    return df[cols_to_agg].mean()
                return (
                    df[cols_to_agg].multiply(weights, axis=0).sum()
                    / weights.sum()
                )

            averaged_accs = joined.groupby(["id", "geometry"])[
                cols_to_agg
            ].apply(weighted_average)

        averaged_accs = averaged_accs.reset_index().set_index("id")

        new_gdf = gpd.GeoDataFrame(
            averaged_accs, geometry="geometry", crs=self.gdf.crs
        )

        return AccessibilityRatings(
            new_origins, self.amenities.values(), self.weights, new_gdf
        )

    @staticmethod
    def _split_gdf(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
        gdf = gdf.assign(group=gdf["id"].str[:-16], id=gdf["id"].str[-15:])
        return {
            group: group_gdf.drop(columns="group")
            for group, group_gdf in gdf.groupby("group")
        }

    def split(self) -> dict[str, "AccessibilityRatings"]:
        """
        Separa orígenes según prefijo (grupo poblacional al que pertenece cada
        celda).

        Returns
        ---
        Un diccionario donde las llaves son los nombres de los grupos
        poblacionales, y los valores son los objetos AccessibilityRatings
        asociados a cada grupo.
        """

        split_grid = self._split_gdf(self.origins.h3_grid)
        split_ratings = self._split_gdf(self.gdf.reset_index())

        return {
            group: AccessibilityRatings(
                origins=Origins(
                    self.origins.regions,
                    self.origins.h3_resolution,
                    split_grid[group],
                ),
                amenities=self.amenities.values(),
                weights=self.weights,
                gdf=split_ratings[group].set_index("id"),
            )
            for group in split_ratings.keys()
        }
