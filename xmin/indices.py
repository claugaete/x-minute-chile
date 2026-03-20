from abc import ABC, abstractmethod

import geopandas as gpd
import pandas as pd

from xmin.amenities import Amenity
from xmin.origins import Origins


class IndexFunction(ABC):
    """
    Clase abstracta para una función de índice, que dados tiempos de viaje
    desde orígenes hasta necesidades, calcula un número entre 0 y 1 indicando
    la "accesibilidad" de cada origen a la necesidad en cuestión.
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


def calculate_weighted_index(
    origins: Origins,
    time_travel_matrices: dict[Amenity, pd.DataFrame],
    index: IndexFunction | dict[Amenity, IndexFunction],
    weights: dict[Amenity, IndexFunction] | None = None,
) -> gpd.GeoDataFrame:
    """
    TODO
    """

    population = origins.h3_grid.set_index("id")["population"]

    amenity_indices: dict[Amenity, pd.Series] = {}

    for amenity, ttm in time_travel_matrices.items():
        amenity_gdf = amenity.amenity_gdf.set_index("id")
        if "weight" not in amenity_gdf.columns:
            amenity_gdf["weight"] = 1
        amenity_indices[amenity] = index[amenity].calculate_index(
            ttm, population, amenity_gdf["weight"]
        )

    weight_sum = sum(weights.values())

    weighted_index = pd.Series(0, index=origins.h3_grid["id"])

    for amenity, weight in weights.items():
        weighted_index += amenity_indices[amenity] * weight / weight_sum

    return origins.h3_grid.set_index("id").assign(accessibility=weighted_index)


class BinaryIndex(IndexFunction):
    """
    TODO
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
    TODO
    """

    def __init__(self, threshold: float, desired_ratio: float):
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
            return 1 / population_in_catchment

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

        return (
            travel_times.set_index("to_id")
            .groupby("from_id")
            .agg(calculate_2sfca)
            .squeeze()
            .rename("accessibility")
            .clip(upper=1 / self.desired_ratio)
            * self.desired_ratio
        )
