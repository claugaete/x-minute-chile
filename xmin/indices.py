from abc import ABC, abstractmethod
import warnings

import geopandas as gpd
import pandas as pd

from xmin.amenities import Amenity
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


def calculate_weighted_index(
    origins: Origins,
    time_travel_matrices: dict[Amenity, pd.DataFrame],
    index_function: IndexFunction | dict[Amenity, IndexFunction],
    weights: dict[Amenity, float] | None = None,
) -> gpd.GeoDataFrame:
    """
    Calcula un índice de accesibilidad general, ponderando índices calculados
    para distintas necesidades según los tiempos de viaje desde cada origen a
    los distintos destinos.

    Parameters
    ---
    origins : Origins
        Orígenes para los cuales se calculará el índice de accesibilidad.
    time_travel_matrices : dict[Amenity, DataFrame]
        Matrices de viaje desde cada origen hacia los distintos destinos,
        separado según la necesidad que cubre cada destino.
    index_function : IndexFunction or dict[Amenity, IndexFunction]
        Funcion(es) para calcular los índices de accesibilidad de las distintas
        necesidades. Si se recibe una IndexFunction, se aplicará la misma para
        todas las Amenities. Si se recibe un diccionario, se asocia cada
        Amenity con su función a utilizar para el cálculo de su índice.
    weights : dict[Amenity, float] or None, default: None
        Pesos relativos de las distintas necesidades. El índice final se
        calcula ponderando los índices de las distintas necesidades utilizando
        estos pesos. Si no se entregan pesos, todos los índices tendrán el
        mismo peso.
    """

    if "population" not in origins.h3_grid.columns:
        population = pd.Series(1, index=origins.h3_grid["id"])
    else:
        population = origins.h3_grid.set_index("id")["population"]

    amenity_index_values: dict[Amenity, pd.Series] = {}
    index_values_gdf = origins.h3_grid.set_index("id")

    # calcular el índice particular de cada necesidad
    for amenity, ttm in time_travel_matrices.items():
        amenity_gdf = amenity.amenity_gdf.set_index("id")
        if "weight" not in amenity_gdf.columns:
            amenity_gdf["weight"] = 1
        current_index = (
            index_function.get(amenity)
            if isinstance(index_function, dict)
            else index_function
        )
        if current_index is None:
            warnings.warn(
                f'Amenity "{amenity.name}" no tiene una función de índice '
                "asociada en `index_function`. No se considerará la necesidad "
                "en el cálculo del índice final."
            )
        else:
            amenity_index_values[amenity] = current_index.calculate_index(
                ttm, population, amenity_gdf["weight"]
            )
            index_values_gdf[amenity.name] = amenity_index_values[amenity]

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
            + ", ".join(amenity.name for amenity in amenities_with_no_weight)
            + ". Su peso será considerado como 0. Para evitar este warning, "
            "asigna un peso a las necesidades (puede ser 0 si no deseas "
            "incluirlas en el cálculo final)."
        )

    weights_with_no_index = set(weights.keys()).difference(
        amenity_index_values.keys()
    )
    if weights_with_no_index:
        warnings.warn(
            "Las siguientes Amenities tienen un peso asociado, pero no se les "
            "calculó un índice: "
            + ", ".join(amenity.name for amenity in weights_with_no_index)
            + ". Su peso será considerado como 0. Para evitar este warning, "
            "asigna un índice a cada necesidad."
        )
    for amenity in weights_with_no_index:
        weights.pop(amenity, None)

    # ponderamos índices para obtener el final
    # TODO poner cada índice individual en su propia columna
    weight_sum = sum(weights.values())
    index_values_gdf["total"] = 0

    for amenity, weight in weights.items():
        index_values_gdf["total"] += (
            amenity_index_values[amenity] * weight / weight_sum
        )

    return index_values_gdf


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
    Índice del tipo 2-step floating catchment area (2SFCA). Calcula una razón
    entre la cantidad de necesidades disponibles en un tiempo de viaje menor al
    máximo (oferta), y la cantidad de personas que acceden a esas necesidades
    (demanda).

    Parameters
    ---
    threshold : float
        Tiempo máximo de viaje permitido.
    desired_ratio : float
        Razón de personas a necesidades para que el índice entregue un valor
        del 100%. Si `desired_ratio=3000`, se necesita una necesidad cada 3000
        personas para obtener una accesibilidad del 100%. Si hay más personas
        por necesidad, se obtiene un valor menor, mientras que si hay menos, se
        sigue obteniendo un 100%.
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

        return (
            travel_times.set_index("to_id")
            .groupby("from_id")
            .agg(calculate_2sfca)
            .squeeze()
            .rename("accessibility")
            .clip(upper=1 / self.desired_ratio)
            * self.desired_ratio
        )
