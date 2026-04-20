import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from xmin.amenities import Amenity
from xmin.geometry import to_centroids
from xmin.indices import IndexFunction
from xmin.origins import Origins
from xmin.visualization import AccessibilityVisualizer


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
        self._visualize = AccessibilityVisualizer(
            self._gdf, self._origins, [amenity for amenity in weights.keys()]
        )

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

    @property
    def visualize(self) -> AccessibilityVisualizer:
        """Módulo con distintas opciones de visualización para los ratings."""
        return self._visualize

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
