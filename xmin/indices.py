from abc import ABC, abstractmethod

import pandas as pd


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


class BinaryIndex(IndexFunction):
    """
    Índice binario de accesibilidad. Para cada origen, si existe un destino
    accesible dentro de un tiempo de viaje máximo, retorna 1; si no, retorna 0.

    Parameters
    ---
    threshold : float
        Tiempo máximo de viaje permitido, en minutos.
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
        Tiempo máximo de viaje permitido, en minutos.
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

        def calculate_inverse_population(travel_times: pd.Series):
            """Calcula el inverso de la población en el catchment area de cada
            destino."""
            cells_in_catchment = travel_times[
                (travel_times <= self.threshold)
            ].index
            population_in_catchment = population.loc[cells_in_catchment].sum()
            return (
                1 / population_in_catchment
                if population_in_catchment > 0
                else 0
            )

        inverse_populations = (
            travel_times.set_index("from_id")
            .groupby("to_id")
            .agg(calculate_inverse_population)
            .squeeze()
            .rename("inverse_population")
        )

        def calculate_2sfca(travel_times: pd.Series):
            """Calcula 2SFCA, dividiendo los pesos de los destinos en el
            catchment por la población que alcanza esos destinos."""
            dests_in_catchment = travel_times[
                (travel_times <= self.threshold)
            ].index
            ratios_in_catchment = (
                amenity_weights.loc[dests_in_catchment]
                * inverse_populations.loc[dests_in_catchment]
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


class EnhancedTwoStepFca(IndexFunction):
    """
    Índice del tipo enhanced 2-step floating catchment area (2SFCA) [1]_.
    Similar a `TwoStepFca`, pero en vez de tener un único tiempo máximo
    (generando una división dicotómica entre quienes tienen acceso a un destino
    y quienes no), tiene distintos "niveles", cada uno con un tiempo máximo y
    un peso asociado. Estos pesos van disminuyendo a medida que los tiempos
    máximos aumentan, para reflejar el menor acceso.

    Parameters
    ---
    catchment_areas : list[tuple[float, float, float]]
        Catchment areas a utilizar en el cálculo, cada una definida por una
        tupla `(min_time, max_time, weight)`. Los destinos cuyo tiempo de viaje
        estén en el intervalo semiabierto `[min_time, max_time)` se ponderarán
        según el peso `weight` asociado.
    desired_ratio : float | None
        Razón de personas a necesidades (ponderadas según distancia) para que
        el índice entregue un valor del 100%.

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
    .. [1] Luo, Wei y Yi Qi: An enhanced two-step floating catchment area
        (E2SFCA) method for measuring spatial accessibility to primary care
        physicians. Health & place, 15(4):1100–1107, 2009.
        https://doi.org/10.1016/j.healthplace.2009.06.002
    """

    def __init__(
        self,
        catchment_areas: list[tuple[float, float, float]],
        desired_ratio: float | None,
    ):
        self.catchment_areas = catchment_areas
        self.desired_ratio = desired_ratio

    def calculate_index(
        self,
        travel_times: pd.DataFrame,
        population: pd.Series,
        amenity_weights: pd.Series,
    ) -> pd.Series:

        def calculate_weighted_inverse_population(travel_times: pd.Series):
            """Calcula el inverso de la población en el catchment area de cada
            destino."""
            denominator = 0
            for best_time, worst_time, weight in self.catchment_areas:
                cells_in_catchment = travel_times[
                    (travel_times >= best_time) & (travel_times < worst_time)
                ].index
                population_in_catchment = population.loc[
                    cells_in_catchment
                ].sum()
                denominator += population_in_catchment * weight
            return 1 / denominator if denominator > 0 else 0

        inverse_populations = (
            travel_times.set_index("from_id")
            .groupby("to_id")
            .agg(calculate_weighted_inverse_population)
            .squeeze()
            .rename("ratio")
        )

        def calculate_e2sfca(travel_times: pd.Series):
            """Calcula E2SFCA, dividiendo los pesos de los destinos en cada
            catchment por la población que alcanza esos destinos."""
            accessibility = 0
            for best_time, worst_time, weight in self.catchment_areas:
                dests_in_catchment = travel_times[
                    (travel_times >= best_time) & (travel_times < worst_time)
                ].index
                ratios_in_catchment = (
                    amenity_weights.loc[dests_in_catchment]
                    * inverse_populations.loc[dests_in_catchment]
                )
                accessibility += ratios_in_catchment * weight
            return accessibility

        unclipped_2sfca = (
            travel_times.set_index("to_id")
            .groupby("from_id")
            .agg(calculate_e2sfca)
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
