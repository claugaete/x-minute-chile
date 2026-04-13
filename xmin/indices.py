from abc import ABC, abstractmethod
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from xmin.amenities import Amenity
from xmin.geometry import to_centroids
from xmin.origins import Origins


class IndexFunction(ABC):
    """
    Clase abstracta para una función de índice, que dados tiempos de viaje
    desde orígenes hasta necesidades, calcula un número entre 0 y 1 indicando
    la "accesibilidad" de cada origen a la necesidad en cuestión.

    Cada función de índice debe utilizar obligatoriamente el argumento
    `travel_times` con la matriz de viaje desde cada origen a cada destino que
    satisface la necesidad. Opcionalmente, la función puede utilizar las
    poblaciones de los orígenes (`population`) y los pesos relativos de los
    destinos (`amenity_weights`).
    """

    @abstractmethod
    def calculate_index(
        self,
        travel_times: pd.DataFrame,
        population: pd.Series,
        amenity_weights: pd.Series,
    ) -> pd.Series:
        """
        Calcula el índice desde cada origen, a partir de la información
        entregada.
        """
        pass


class AccessibilityRatings:
    """
    Guarda ratings de accesibilidad, calculando índices para distintas
    necesidades según los tiempos de viaje desde cada origen a los distintos
    destinos, y ponderándolos para obtener un índice de accesibilidad general.

    Parameters
    ---
    origins : Origins
        Orígenes desde los cuales se calcularon los índices.
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
        weights: dict[Amenity, float],
        gdf: gpd.GeoDataFrame,
    ):
        self._origins = origins
        self._weights = weights
        self._gdf = gdf

    @property
    def origins(self) -> Origins:
        """Orígenes desde los cuales se calculó la accesibilidad."""
        return self._origins

    @property
    def weights(self) -> dict[Amenity, float]:
        """Pesos relativos de cada necesidad para el cálculo de índices."""
        return self._weights

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """GeoDataFrame con los ratings de accesibilidad calculados."""
        return self._gdf

    @classmethod
    def compute(
        cls,
        origins: Origins,
        time_travel_matrices: dict[Amenity, pd.DataFrame],
        index_function: IndexFunction | dict[Amenity, IndexFunction],
        weights: dict[Amenity, float] | None = None,
    ) -> "AccessibilityRatings":
        """
        Calcula los índices de accesibilidad y retorna una instancia de
        AccessibilityRatings.

        Parameters
        ---
        origins : Origins
            Orígenes para los cuales se calculará el índice de accesibilidad.
        time_travel_matrices : dict[Amenity, DataFrame]
            Matrices de viaje desde cada origen hacia los distintos destinos,
            separado según la necesidad que cubre cada destino.
        index_function : IndexFunction or dict[Amenity, IndexFunction]
            Funcion(es) para calcular los índices de accesibilidad de las
            distintas necesidades. Si se recibe una IndexFunction, se aplicará
            la misma para todas las Amenities. Si se recibe un diccionario, se
            asocia cada Amenity con su función a utilizar para el cálculo de su
            índice.
        weights : dict[Amenity, float] or None, default: None
            Pesos relativos de las distintas necesidades. El índice final se
            calcula ponderando los índices de las distintas necesidades
            utilizando estos pesos. Si no se entregan pesos, todos los índices
            tendrán el mismo peso.
        """

        population = origins.h3_grid.set_index("id")["population"]

        amenity_index_values: dict[Amenity, pd.Series] = {}
        ratings_gdf = origins.h3_grid.set_index("id")[["geometry"]]

        # calcular el índice particular de cada necesidad
        print("Calculando índices para cada necesidad...")
        for amenity, ttm in tqdm(time_travel_matrices.items()):
            amenity_gdf = amenity.amenity_gdf.set_index("id")
            current_index = (
                index_function.get(amenity)
                if isinstance(index_function, dict)
                else index_function
            )
            if current_index is None:
                warnings.warn(
                    f'Amenity "{amenity.name}" no tiene una función de índice '
                    "asociada en `index_function`. No se considerará la "
                    "necesidad en el cálculo del índice final."
                )
            else:
                amenity_index_values[amenity] = current_index.calculate_index(
                    ttm, population, amenity_gdf["weight"]
                )
                ratings_gdf[amenity.name] = amenity_index_values[amenity]

        # asignamos pesos equilibrados si no se indica lo contrario
        if weights is None:
            weights = {k: 1 for k in amenity_index_values.keys()}

        # arreglamos discrepancias entre índices calculados y pesos recibidos
        amenities_with_no_weight = set(amenity_index_values.keys()).difference(
            weights.keys()
        )
        if amenities_with_no_weight:
            warnings.warn(
                "Las siguientes Amenities no tienen un peso asociado: "
                + ", ".join(
                    amenity.name for amenity in amenities_with_no_weight
                )
                + ". Su peso será considerado como 0. Para evitar este "
                "warning, asigna un peso a las necesidades (puede ser 0 si no "
                "deseas incluirlas en el cálculo final)."
            )
        for amenity in amenities_with_no_weight:
            weights[amenity] = 0

        weights_with_no_index = set(weights.keys()).difference(
            amenity_index_values.keys()
        )
        if weights_with_no_index:
            warnings.warn(
                "Las siguientes Amenities tienen un peso asociado, pero no se "
                "les calculó un índice: "
                + ", ".join(amenity.name for amenity in weights_with_no_index)
                + ". Estas Amenities serán ignoradas. Para evitar este "
                "warning, asigna un índice a cada necesidad."
            )
        for amenity in weights_with_no_index:
            weights.pop(amenity, None)

        # ponderamos índices para obtener el final
        weight_sum = sum(weights.values())
        ratings_gdf["total"] = 0

        for amenity, weight in weights.items():
            ratings_gdf["total"] += (
                amenity_index_values[amenity] * weight / weight_sum
            )

        return cls(origins, weights, ratings_gdf)

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

        return AccessibilityRatings(new_origins, self.weights, new_ratings)

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

        Returns
        ---
        Un nuevo objeto AccessibilityRatings con la resolución reducida.
        """

        if new_resolution > self.origins.h3_resolution:
            raise ValueError(
                f"Nueva resolución ({new_resolution}) no puede ser mayor a la "
                f"resolución actual ({self.origins.h3_resolution})."
            )

        new_origins = Origins.create_grid(
            self.origins.regions,
            new_resolution,
            population_gdf=to_centroids(self.origins.h3_grid),
        )

        # each old cell gets assigned to the new cell where its centroid falls
        left_gdf = to_centroids(self.gdf)
        if weighted:
            left_gdf = left_gdf.join(
                self.origins.h3_grid.set_index("id")["population"]
            )
        joined = left_gdf.sjoin(
            new_origins.h3_grid.drop(columns="population"),
            how="right",
            predicate="within",
            lsuffix="left",
            rsuffix=None,
        )

        # we average the accessibility values
        cols_to_agg = [col for col in self.gdf.columns if col != "geometry"]
        if not weighted:
            averaged_accs = joined.groupby(["id", "geometry"])[
                cols_to_agg
            ].mean()
        else:

            def weighted_average(df: pd.DataFrame):
                weights = df["population"]
                if weights.sum() == 0:
                    return df[cols_to_agg].mean()
                return (
                    df[cols_to_agg].multiply(weights, axis=0).sum()
                    / weights.sum()
                )

            averaged_accs = joined.groupby(["id", "geometry"])[
                cols_to_agg + ["population"]
            ].apply(weighted_average)

        averaged_accs = averaged_accs.reset_index().set_index("id")

        new_gdf = gpd.GeoDataFrame(
            averaged_accs, geometry="geometry", crs=self.gdf.crs
        )

        return AccessibilityRatings(new_origins, self.weights, new_gdf)


class BinaryIndex(IndexFunction):
    """
    Índice binario de accesibilidad. Para cada origen, si existe un destino
    accesible dentro de un tiempo de viaje máximo, retorna 1; si no, retorna 0.

    Parameters
    ---
    threshold : float
        Tiempo máximo de viaje permitido.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def calculate_index(
        self,
        travel_times: pd.DataFrame,
        population: pd.Series,
        amenity_weights: pd.Series,
    ) -> pd.Series:
        return (
            travel_times.set_index("to_id")
            .groupby("from_id")["travel_time"]
            .min()
            .apply(lambda x: 1 if x is not None and x <= self.threshold else 0)
        )


class TwoStepFca(IndexFunction):
    """
    Índice del tipo 2-step floating catchment area (2SFCA) [1]_. Calcula una
    razón entre la cantidad de necesidades disponibles en un tiempo de viaje
    menor al máximo (oferta), y la cantidad de personas que acceden a esas
    necesidades (demanda).

    Parameters
    ---
    threshold : float
        Tiempo máximo de viaje permitido.
    desired_ratio : float | None
        Razón de personas a necesidades para que el índice entregue un valor
        del 100%.

        Por ejemplo, si `desired_ratio=3000`, se necesita una necesidad cada
        3000 personas para obtener una accesibilidad del 100%. Si hay más
        personas por necesidad, se obtiene un valor menor, mientras que si hay
        menos, se sigue obteniendo un 100%.

        Si `desired_ratio=None`, se utilizará como máximo el mejor resultado
        obtenido entre los orígenes (que obtendrá el 100%), mientras que el
        resto de orígenes obtendrá valores proporcionalmente menores según sus
        resultados.

    References
    ---
    .. [1] Luo, Wei y Fahui Wang: Measures of spatial accessibility to health
        care in a GIS environment: synthesis and a case study in the Chicago
        region. Environment and planning B: planning and design, 30(6):865–884,
        2003. https://doi.org/10.1068/b29120
    """

    def __init__(self, threshold: float, desired_ratio: float | None):
        self.threshold = threshold
        self.desired_ratio = desired_ratio

    def calculate_index(
        self,
        travel_times: pd.DataFrame,
        population: pd.Series,
        amenity_weights: pd.Series,
    ) -> pd.Series:

        def calculate_need_to_population_ratio(travel_times: pd.Series):
            cells_in_catchment = travel_times[
                (travel_times <= self.threshold)
            ].index
            population_in_catchment = population.loc[cells_in_catchment].sum()
            return (
                1 / population_in_catchment
                if population_in_catchment > 0
                else 0
            )

        ratios_2sfca = (
            travel_times.set_index("from_id")
            .groupby("to_id")
            .agg(calculate_need_to_population_ratio)
            .squeeze()
            .rename("ratio")
        )

        def calculate_2sfca(travel_times: pd.Series):
            dests_in_catchment = travel_times[
                (travel_times <= self.threshold)
            ].index
            ratios_in_catchment = (
                amenity_weights.loc[dests_in_catchment]
                * ratios_2sfca.loc[dests_in_catchment]
            )
            return ratios_in_catchment.sum()

        unclipped_2sfca = (
            travel_times.set_index("to_id")
            .groupby("from_id")
            .agg(calculate_2sfca)
            .squeeze()
            .rename("accessibility")
        )

        self.desired_ratio = (
            1 / unclipped_2sfca.max()
            if self.desired_ratio is None
            else self.desired_ratio
        )

        return (
            unclipped_2sfca.clip(upper=1 / self.desired_ratio)
            * self.desired_ratio
        )
