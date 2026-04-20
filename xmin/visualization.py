from dataclasses import dataclass, field
from pathlib import Path

from folium.folium import Map
import geopandas as gpd
from matplotlib.axes import Axes
import pandas as pd
import quackosm as qosm
from shapely.geometry.base import BaseGeometry

import xmin
from xmin.origins import Origins


@dataclass
class OverlayConfig:
    """
    Configuración de capas adicionales a mostrar en la visualización.

    Attributes
    ---
    show_borders : bool, default: True
        Si se mostrarán las fronteras de las regiones del análisis.
    show_roads : bool, default: False
        Si se mostrarán caminos para contextualizar la región de análisis.
    borders_kwds : dict, default: {}
        Diccionario de argumentos que serán pasados al momento de graficar las
        fronteras.
    roads_kwds : dict, default: {}
        Diccionario de argumentos que serán pasados al momento de graficar los
        caminos.
    roads_gdf : GeoDataFrame or None, default: None
        GeoDataFrame con los caminos a graficar; si es nulo y `show_roads` es
        verdadero, se espera recibir `roads_pbf_path`.
    roads_pbf_path : Path | str | None, default: None
        Ruta al archivo PBF del cual extraer los caminos a graficar. Se extraen
        los caminos que contienen alguna las siguientes etiquetas: `{"highway":
        ["motorway", "trunk", "primary", "secondary", "tertiary"]}`. Si es nulo
        y `show_roads` es verdadero, se espera recibir `roads_gdf`.
    """

    show_borders: bool = True
    show_roads: bool = False
    borders_kwds: dict = field(default_factory=dict)
    roads_kwds: dict = field(default_factory=dict)
    roads_gdf: gpd.GeoDataFrame | None = None
    roads_pbf_path: Path | str | None = None


class AccessibilityVisualizer:
    """
    Clase que contiene distintos métodos para visualizar los ratings entregados
    por un objeto AccessibilityRatings a un conjunto de orígenes.

    Parameters
    ---
    gdf : GeoDataFrame
        GeoDataFrame con los ratings del área a analizar.
    origins : Origins
        Orígenes representando el área a analizar.
    """

    def __init__(self, gdf: gpd.GeoDataFrame, origins: Origins):
        self.gdf = gdf
        self.origins = origins

    @staticmethod
    def _get_roads(roads_pbf_path: Path | str, bounds: BaseGeometry):
        """Obtener caminos principales a graficar mediante QuackOSM."""
        roads_gdf = qosm.convert_pbf_to_geodataframe(
            pbf_path=roads_pbf_path,
            tags_filter={
                "highway": [
                    "motorway",
                    "trunk",
                    "primary",
                    "secondary",
                    "tertiary",
                ]
            },
            working_directory=xmin.quackosm_working_directory,
            geometry_filter=bounds,
            keep_all_tags=False,
        )
        return roads_gdf.clip(bounds)

    def _show(
        self,
        values: pd.Series,
        interactive: bool,
        overlay_cfg: OverlayConfig,
        **kwargs,
    ) -> Axes | Map:
        """
        Muestra el gráfico/mapa deseado.

        Parameters
        ---
        values : Series
            Serie de valores a mostrar. El índice de la serie deben ser las IDs
            de las grillas de H3 incluidas en el análisis.
        interactive : bool
            Si la visualización será interactiva (mapa) o estática (gráfico).
        overlay_cfg : OverlayConfig
            Configuración de capas adicionales. Ver `OverlayConfig`.
        kwargs
            Argumentos que serán pasados al momento de graficar la grilla de
            origenes con los valores de `values`.
        """

        if overlay_cfg.show_roads and not overlay_cfg.roads_gdf:
            if overlay_cfg.roads_pbf_path:
                roads_gdf = self._get_roads(
                    overlay_cfg.roads_pbf_path,
                    self.origins.regions.union_all(),
                )
            else:
                raise ValueError(
                    "`show_roads` es verdadero, pero no se entregó un "
                    "GeoDataFrame con las calles a mostrar ni una ruta a un "
                    "archivo PBF del cual extraer las calles."
                )

        col_name = values.name
        gdf_to_plot = self.gdf.copy()
        gdf_to_plot[col_name] = values

        if interactive:
            default_explore_kwds = {"tiles": "CartoDB Voyager"}
            default_overlay_kwds = {
                "fill": False,
                "tooltip": False,
                "color": "black",
            }

            m = gdf_to_plot.explore(
                col_name, **(default_explore_kwds | kwargs)
            )
            if overlay_cfg.show_borders:
                self.origins.regions.explore(
                    m=m, **(default_overlay_kwds | overlay_cfg.borders_kwds)
                )
            if overlay_cfg.show_roads:
                roads_gdf.explore(
                    m=m, **(default_overlay_kwds | overlay_cfg.roads_kwds)
                )
            return m
        else:
            # make sure legend and plot have same alpha by default (they can be
            # overwritten if needed)
            default_legend_kwds = {
                "alpha": (
                    xmin.alpha_when_roads_shown
                    if overlay_cfg.show_roads
                    else 1
                )
            } | kwargs.pop("legend_kwds", {})
            default_plot_kwds = {
                "alpha": (
                    xmin.alpha_when_roads_shown
                    if overlay_cfg.show_roads
                    else 1
                ),
                "legend_kwds": default_legend_kwds,
            }

            default_borders_kwds = {"edgecolor": "black", "facecolor": "none"}
            default_roads_kwds = default_borders_kwds | {
                "linewidth": 0.5,
                "zorder": -1,
            }

            ax = gdf_to_plot.plot(col_name, **(default_plot_kwds | kwargs))
            if overlay_cfg.show_borders:
                self.origins.regions.plot(
                    ax=ax, **(default_borders_kwds | overlay_cfg.borders_kwds)
                )
            if overlay_cfg.show_roads:
                roads_gdf.plot(
                    ax=ax, **(default_roads_kwds | overlay_cfg.roads_kwds)
                )
            return ax

    def choropleth(
        self,
        column: str | pd.Series,
        interactive: bool = False,
        overlay_cfg: OverlayConfig = OverlayConfig(),
        **kwargs,
    ) -> Axes | Map:
        """
        Realiza una visualización de coropletas con el índice deseado.

        Parameters
        ---
        column : str or Series
            Serie de valores a mostrar. Puede ser el nombre de uno de los
            índices ya calculados, o un cálculo propio. En el segundo caso, el
            índice de la serie deben ser las IDs de las grillas de H3 incluidas
            en el análisis.
        interactive : bool, default: False
            Si la visualización será interactiva (mapa) o estática (gráfico).
        overlay_cfg : OverlayConfig, default: OverlayConfig()
            Configuración de capas adicionales. Ver `OverlayConfig`.
        kwargs
            Argumentos que serán pasados a `GeoDataFrame.plot()` (si
            `interactive=False`) o `GeoDataFrame.explore()` (si
            `interactive=True`) al momento de graficar la grilla de orígenes
            con los valores de `column`.
        """

        if isinstance(column, str):
            column = self.gdf[column]

        return self._show(
            column,
            interactive,
            overlay_cfg,
            **kwargs,
        )

    def nexi_discomfort(
        self,
        column: str,
        interactive: bool = False,
        overlay_cfg: OverlayConfig = OverlayConfig(),
        **kwargs,
    ):
        """
        Visualización de NEXI-Discomfort, según la definición de Olivari et. al
        [1]_. Dado un origen con un rating de accesibilidad `R` y una población
        `P`, se define el índice NEXI-Discomfort como `(1-R)*P`. Esto entrega
        un mayor valor a zonas muy pobladas que tienen mala accesibilidad.

        Parameters
        ---
        column : str
            Columna que se desea utilizar como rating base `R` de
            NEXI-Discomfort.
        interactive : bool, default: False
            Si la visualización será interactiva (mapa) o estática (gráfico).
        overlay_cfg : OverlayConfig, default: OverlayConfig()
            Configuración de capas adicionales. Ver `OverlayConfig`.
        kwargs
            Argumentos que serán pasados a `GeoDataFrame.plot()` (si
            `interactive=False`) o `GeoDataFrame.explore()` (si
            `interactive=True`) al momento de graficar la grilla de orígenes
            con los valores de `column`.

        References
        ---
        .. [1] Olivari, Beatrice, Piergiorgio Cipriano, Maurizio Napolitano y
            Luca Giovannini: Are Italian cities already 15-minute? Presenting
            the Next Proximity Index: A novel and scalable way to measure it,
            based on open data. Journal of Urban Mobility, 4:100057, 2023, ISSN
            2667-0917. https://doi.org/10.1016/j.urbmob.2023.100057
        """

        discomfort = (1 - self.gdf[column]) * self.origins.h3_grid.set_index(
            "id"
        )["population"]
        discomfort = discomfort.rename("discomfort")

        return self.choropleth(discomfort, interactive, overlay_cfg, **kwargs)
