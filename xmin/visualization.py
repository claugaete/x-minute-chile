from dataclasses import dataclass, field
from pathlib import Path
import warnings

import folium
from folium.folium import Map
import geopandas as gpd
import mapclassify
from mapclassify.classifiers import MapClassifier
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
import numpy as np
import pandas as pd
import quackosm as qosm
from shapely.geometry.base import BaseGeometry

import xmin
from xmin.amenities import Amenity
from xmin.origins import Origins


def add_bivariate_legend(
    m: Map,
    color_matrix: np.ndarray,
    n: int,
    var1_label: str,
    var2_label: str,
    legend_kwds: dict,
) -> Map:
    """
    Agrega una leyenda a un mapa bivariado de coropletas, inyectando el código
    correspondiente en HTML directamente al objeto.

    Parameters
    ---
    m : Map
        Mapa al cual agregar la leyenda.
    color_matrix : ndarray
        Matriz de `n*n` con los colores de la leyenda.
    n : int
        Número de filas y columnas que tiene la matriz de colores.
    var1_label : str
        Nombre de la primera variable (eje y).
    var2_label : str
        Nombre de la segunda variable (eje x).
    legend_kwds : dict
        Keywords que serán utilizadas para el estilo de la leyenda:

        - `cell_width` define el ancho de cada celda de color.
        - `cell_height` define el alto de cada celda de color.
        - El resto de parámetros definen el estilo de la caja que contiene a la
          leyenda (padding, posición, color, etc.).

    Returns
    ---
    Mapa con la leyenda agregada.
    """

    cell_width = legend_kwds.pop("cell_width", "20px")
    cell_height = legend_kwds.pop("cell_height", "20px")

    cells = ""
    for i in range(n - 1, -1, -1):
        cells += (
            "<tr>"
            + "".join(
                f'<td style="width:{cell_width};height:{cell_height};'
                f'background:{color_matrix[i][j]};"></td>'
                for j in range(n)
            )
            + "</tr>"
        )

    div_style = "".join(f"{k}:{v};" for k, v in legend_kwds.items())

    html = f"""
    <div style="{div_style}">
        <div style="
            display:flex;
            align-items:stretch;
            gap:4px;
        ">
            <div style="
                display:flex;
                align-items:center;
                justify-content:center;
                writing-mode:vertical-rl;
                transform:rotate(180deg);
            ">
                {var1_label} →
            </div>
            <div style="
                display:flex;
                flex-direction:column;
            ">
                <table style="
                    border-collapse:collapse;
                ">
                    {cells}
                </table>
                <div style="
                    text-align:center;
                    margin-top:3px;
                ">
                    {var2_label} →
                </div>
            </div>
        </div>
    </div>
    """

    m.get_root().html.add_child(folium.Element(html))
    return m


@dataclass
class OverlayConfig:
    """
    Configuración de capas adicionales a mostrar en la visualización.

    Attributes
    ---
    show_borders : bool, default: True
        Si se mostrarán las fronteras de las regiones del análisis.
    show_amenities : bool | list[Amenity], default: False
        Si se mostrarán los destinos asociados a las distintas necesidades
        consideradas. Si se recibe una lista de objetos Amenity, se mostrarán
        los destinos asociados a esos objetos.
    show_roads : bool, default: False
        Si se mostrarán caminos para contextualizar la región de análisis.
    borders_kwds : dict, default: {}
        Diccionario de argumentos que serán pasados al momento de graficar las
        fronteras.
    amenities_kwds : dict, default: {}
        Diccionario de argumentos que serán pasados al momento de graficar las
        necesidades.
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
    show_amenities: bool | list[Amenity] = False
    show_roads: bool = False
    borders_kwds: dict = field(default_factory=dict)
    amenities_kwds: dict = field(default_factory=dict)
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
    weights : dict[Amenity, float]
        Pesos relativos de las distintas necesidades para el cálculo del índice
        "total" de `gdf`.
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        origins: Origins,
        weights: dict[Amenity, float],
    ):
        self._gdf = gdf
        self._origins = origins
        self._weights = weights

    @staticmethod
    def _get_roads(
        roads_pbf_path: Path | str, bounds: BaseGeometry
    ) -> gpd.GeoDataFrame:
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

    @staticmethod
    def _get_destinations(amenities: list[Amenity]) -> gpd.GeoDataFrame:
        """Obtener destinos a partir de un conjunto de necesidades."""
        return gpd.GeoDataFrame(
            pd.concat(
                [
                    amenity.amenity_gdf[
                        list(
                            {
                                "id",
                                "name",
                                "weight",
                                "geometry",
                            }.intersection(amenity.amenity_gdf.columns)
                        )
                    ].assign(category=amenity.name)
                    for amenity in amenities
                ]
            ),
            crs=4326,
        )

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
  
        Returns
        ---
        Objeto graficado (mapa de `folium` o ejes de `matplotlib`, según el
        valor de `interactive`).
        """

        if isinstance(overlay_cfg.show_amenities, list):
            if overlay_cfg.amenities_kwds == []:
                raise ValueError("`show_amenities` es una lista vacía.")
            else:
                amenities_to_plot = self._get_destinations(
                    overlay_cfg.show_amenities
                )
        elif overlay_cfg.show_amenities:
            amenities_to_plot = self._get_destinations(self._weights.keys())

        if overlay_cfg.show_roads and not overlay_cfg.roads_gdf:
            if overlay_cfg.roads_pbf_path:
                roads_gdf = self._get_roads(
                    overlay_cfg.roads_pbf_path,
                    self._origins.regions.union_all(),
                )
            else:
                raise ValueError(
                    "`show_roads` es verdadero, pero no se entregó un "
                    "GeoDataFrame con las calles a mostrar ni una ruta a un "
                    "archivo PBF del cual extraer las calles."
                )

        col_name = values.name
        gdf_to_plot = self._gdf.copy()
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
                self._origins.regions.explore(
                    m=m, **(default_overlay_kwds | overlay_cfg.borders_kwds)
                )
            if overlay_cfg.show_amenities:
                amenities_to_plot.explore(
                    "category", m=m, **overlay_cfg.amenities_kwds
                )
            if overlay_cfg.show_roads:
                roads_gdf.explore(
                    m=m, **(default_overlay_kwds | overlay_cfg.roads_kwds)
                )
            return m
        else:
            # make sure legend and plot have same alpha by default (they can be
            # overwritten if needed)
            default_plot_kwds = {
                "alpha": (
                    xmin.alpha_when_roads_shown
                    if overlay_cfg.show_roads
                    else 1
                ),
            }

            default_borders_kwds = {"edgecolor": "black", "facecolor": "none"}
            default_roads_kwds = default_borders_kwds | {
                "linewidth": 0.5,
                "zorder": -1,
            }
            default_amenities_kwds = {
                "markersize": 2,
                "legend": True,
            }

            ax = gdf_to_plot.plot(col_name, **(default_plot_kwds | kwargs))
            if overlay_cfg.show_borders:
                self._origins.regions.plot(
                    ax=ax, **(default_borders_kwds | overlay_cfg.borders_kwds)
                )
            if overlay_cfg.show_amenities:
                amenities_to_plot.plot(
                    "category",
                    ax=ax,
                    **(default_amenities_kwds | overlay_cfg.amenities_kwds),
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
  
        Returns
        ---
        Objeto graficado (mapa de `folium` o ejes de `matplotlib`, según el
        valor de `interactive`).
        """

        if isinstance(column, str):
            column = self._gdf[column]

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
  
        Returns
        ---
        Objeto graficado (mapa de `folium` o ejes de `matplotlib`, según el
        valor de `interactive`).

        References
        ---
        .. [1] Olivari, Beatrice, Piergiorgio Cipriano, Maurizio Napolitano y
            Luca Giovannini: Are Italian cities already 15-minute? Presenting
            the Next Proximity Index: A novel and scalable way to measure it,
            based on open data. Journal of Urban Mobility, 4:100057, 2023, ISSN
            2667-0917. https://doi.org/10.1016/j.urbmob.2023.100057
        """

        discomfort = (1 - self._gdf[column]) * self._origins.h3_grid.set_index(
            "id"
        )["population"]
        discomfort = discomfort.rename("discomfort")

        return self.choropleth(discomfort, interactive, overlay_cfg, **kwargs)

    def bivariate_choropleth(
        self,
        column_1: str | pd.Series,
        column_2: str | pd.Series,
        n: int = 3,
        classifier: type[MapClassifier] = mapclassify.EqualInterval,
        color_matrix: np.ndarray | None = None,
        legend: bool = True,
        legend_kwds: dict | None = None,
        interactive: bool = False,
        overlay_cfg: OverlayConfig = OverlayConfig(),
        **kwargs,
    ):
        """
        Realiza una visualización de coropletas bivariada con los índices
        deseados.

        Parameters
        ---
        column_1 : str or Series
            Primera serie de valores a mostrar. Puede ser el nombre de uno de
            los índices ya calculados, o un cálculo propio. En el segundo caso,
            el índice de la serie deben ser las IDs de las grillas de H3
            incluidas en el análisis.
        column_2 : str or Series
            Segunda serie de valores a mostrar; aplican las mismas
            restricciones que para `column_1`.
        n : int, default: 3
            Número de clases en las que se va a dividir cada variable. Se
            recomienda mantener `n=3`; valores más altos generan un mapa de
            colores demasiado grande.
        classifier : type[MapClassifier]
            Clase correspondiente a un `MapClassifier` de la librería
            `mapclassify`, que permite dividir los datos de cada serie en
            "bins" según algún criterio. El inicializador de la clase debe
            recibir un conjunto de datos y un número de clases en las que
            dividir los datos.
        color_matrix : ndarray or None, default: None
            Matriz de colores para asignar a las distintas clases. Debe ser una
            matriz de `n*n`, donde `color_matrix[i, j]` guarda el color
            asociado para elementos que caen en el bin `i` según `column_1`, y
            en el bin `j` según `column_2`.

            Si `n=3` y `color_matrix` es None, se utilizará una matriz de
            colores por defecto, extraída del blog de Joshua Stevens sobre
            mapas de coropletas bivariados
            (https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/).
            Para `n != 3`, se debe entregar `color_matrix` obligatoriamente.
        legend : bool, default: True
            Si se muestra la leyenda asociada a los distintos colores.
        legend_kwds : dict | None, default: None
            Lista de parámetros para la leyenda.

            Si `interactive=False`, el diccionario puede contener los
            siguientes parámetros:
                - `"bounds"` (default: [1.05, 0.70, 0.25, 0.25]): límites de la
                  leyenda. Debe ser una 4-tupla `[x0, y0, width, height]`,
                  indicando la esquina inferior izquierda de la leyenda, su
                  ancho y su alto. Todos estos valores son relativos al tamaño
                  del gráfico (entre 0 y 1).
                - `"label_fontsize"` (default: 10): tamaño de la fuente de los
                  labels de la leyenda.

            Si `interactive=True`, el diccionario puede contener los siguientes
            parámetros:
                - `"cell_width"` (default: "20px"): ancho de cada celda de
                  color.
                - `"cell_height"` (default: "20px"): alto de cada celda de
                  color.
                - El resto de parámetros definen el estilo de la caja que
                  contiene a la
                leyenda (padding, posición, color, etc.). Los parámetros por
                defecto que pueden ser sobreescritos son:
                ```
                {
                    "position": "fixed",
                    "top": "10px",
                    "right": "10px",
                    "z-index": 1000,
                    "background": "white",
                    "padding": "10px",
                    "border-radius": "4px",
                    "box-shadow": "0 0 6px rgba(0,0,0,0.3)",
                    "font-size": "11px",
                    "font-family": "Arial",
                }
                ```

        interactive : bool, default: False
            Si la visualización será interactiva (mapa) o estática (gráfico).
        overlay_cfg : OverlayConfig, default: OverlayConfig()
            Configuración de capas adicionales. Ver `OverlayConfig`.
        kwargs
            Argumentos que serán pasados a `GeoDataFrame.plot()` (si
            `interactive=False`) o `GeoDataFrame.explore()` (si
            `interactive=True`) al momento de graficar la grilla de orígenes
            con los valores de `column`.
  
        Returns
        ---
        Objeto graficado (mapa de `folium` o ejes de `matplotlib`, según el
        valor de `interactive`).
        """

        if color_matrix is None:
            if n == 3:
                color_matrix = np.array(
                    [
                        ["#e8e8e8", "#ace4e4", "#5ac8c8"],
                        ["#dfb0d6", "#a5add3", "#5698b9"],
                        ["#be64ac", "#8c62aa", "#3b4994"],
                    ]
                )
            else:
                raise ValueError(
                    "Si `n != 3`, se debe entregar `color_matrix` de `n * n`"
                )
        color_list = color_matrix.flatten()
        cmap = ListedColormap(color_list)

        if kwargs.pop("cmap", None) is not None:
            warnings.warn(
                "Ignorando parámetro `cmap`; para cambiar el mapa de colores, "
                "usar `color_matrix`."
            )
        if kwargs.pop("categories", None) is not None:
            warnings.warn(
                "Ignorando parámetro `categories`; las categorías son "
                "determinadas según `classifier`."
            )

        gdf_class = self._gdf.copy()
        if isinstance(column_1, str):
            column_1 = gdf_class[column_1]
        if isinstance(column_2, str):
            column_2 = gdf_class[column_2]

        class_1 = f"{column_1.name}_class"
        class_2 = f"{column_2.name}_class"

        # assign classes for each column and combine the classes
        split_1 = classifier(column_1, n)
        gdf_class[class_1] = split_1.yb
        split_2 = classifier(column_2, n)
        gdf_class[class_2] = split_2.yb
        gdf_class["bivariate_class"] = (
            gdf_class[class_1] * n + gdf_class[class_2]
        )
        gdf_class["color"] = gdf_class["bivariate_class"].map(
            lambda x: color_list[int(x)]
        )

        output = self._show(
            gdf_class["bivariate_class"],
            interactive,
            overlay_cfg,
            cmap=cmap,
            categories=list(range(n * n)),
            legend=False,
            **kwargs,
        )

        if legend:
            if interactive:
                default_legend_kwds = {
                    "cell_width": "20px",
                    "cell_height": "20px",
                    "position": "fixed",
                    "top": "10px",
                    "right": "10px",
                    "z-index": 1000,
                    "background": "white",
                    "padding": "10px",
                    "border-radius": "4px",
                    "box-shadow": "0 0 6px rgba(0,0,0,0.3)",
                    "font-size": "11px",
                    "font-family": "Arial",
                }
                if legend_kwds is not None:
                    legend_kwds = default_legend_kwds | legend_kwds
                else:
                    legend_kwds = default_legend_kwds
                return add_bivariate_legend(
                    output,
                    color_matrix,
                    n,
                    column_1.name,
                    column_2.name,
                    legend_kwds,
                )
            else:
                default_legend_kwds = {
                    "bounds": [1.05, 0.70, 0.25, 0.25],
                    "label_fontsize": 10,
                }
                if legend_kwds is not None:
                    legend_kwds = default_legend_kwds | legend_kwds
                else:
                    legend_kwds = default_legend_kwds

                ax: Axes = output
                legend_ax = ax.inset_axes(
                    legend_kwds["bounds"],
                )
                for spine in legend_ax.spines.values():
                    spine.set_visible(False)
                for i in range(n):
                    for j in range(n):
                        legend_ax.fill_between(
                            [j, j + 1],
                            [i, i],
                            [i + 1, i + 1],
                            color=color_matrix[i][j],
                        )
                legend_ax.set_xlabel(
                    f"{column_2.name} →",
                    fontsize=legend_kwds["label_fontsize"],
                )
                legend_ax.set_ylabel(
                    f"{column_1.name} →",
                    fontsize=legend_kwds["label_fontsize"],
                )
                legend_ax.set_xticks([])
                legend_ax.set_yticks([])

                return ax

        return output

    def most_urgent_amenity(
        self,
        reorder: bool = True,
        cmap: str | Colormap = "Accent",
        na_color: str | None = None,
        interactive: bool = False,
        overlay_cfg: OverlayConfig = OverlayConfig(),
        **kwargs,
    ):
        """
        Visualiza la necesidad "más urgente" para cada origen; esto es, la
        necesidad que aumentaría más el rating "total" (ponderado) si el rating
        de la necesidad aumentase a 100%. Matemáticamente, para cada origen
        `i`, la necesidad más urgente es la necesidad `A` que maximiza
        `weight[A] * (1 - Rating[i, A])`, donde `Rating[i, A]` es el rating de
        accesibilidad de la necesidad `A` en el origen `i`. Si un origen tiene
        todas sus necesidades al 100%, el resultado es "N/A".

        Parameters
        ---
        reorder : bool, default: True
            Si reordenar las necesidades alfabéticamente (para la leyenda y el
            orden en que se asocian los colores de `cmap`). La categoría "N/A"
            siempre quedará al final.
        cmap : str or Colormap, default: Accent
            Colormap a utilizar. Si es un string, se extraerá el colormap
            correspondiente de `matplotlib.colormaps`. Se utilizarán los
            **primeros** N colores, donde N es la cantidad de necesidades a
            considerar. Si `na_color` es None, entonces se utilizarán los
            primeros N+1 colores, con la categoría "N/A" usando siempre el
            último.
        na_color : str or None, default: None
            Color especial para celdas con valor "N/A". Si es None, se utiliza
            el color N+1 de `cmap`.
        interactive : bool, default: False
            Si la visualización será interactiva (mapa) o estática (gráfico).
        overlay_cfg : OverlayConfig, default: OverlayConfig()
            Configuración de capas adicionales. Ver `OverlayConfig`.
        kwargs
            Argumentos que serán pasados a `GeoDataFrame.plot()` (si
            `interactive=False`) o `GeoDataFrame.explore()` (si
            `interactive=True`) al momento de graficar la grilla de orígenes
            con los valores de `column`.
  
        Returns
        ---
        Objeto graficado (mapa de `folium` o ejes de `matplotlib`, según el
        valor de `interactive`).
        """

        weights_by_name = {
            amenity.name: weight for amenity, weight in self._weights.items()
        }

        urgency_df = self._gdf.drop(columns=["total", "geometry"]).apply(
            lambda rating: weights_by_name[rating.name] * (1 - rating)
        )
        urgency_df["N/A"] = urgency_df.sum(axis=1) == 0
        categories = urgency_df.columns
        if reorder:
            categories = list(
                categories.drop("N/A").sort_values(key=lambda x: x.str.lower())
            ) + ["N/A"]
        categorical_dtype = pd.CategoricalDtype(categories=categories)
        most_urgent = urgency_df.idxmax(axis="columns").astype(
            categorical_dtype
        )
        most_urgent.name = "most_urgent"

        if isinstance(cmap, str):
            cmap = mpl.colormaps.get_cmap(cmap)
        colors = [cmap(i) for i in range(len(categories) - 1)]
        if na_color is not None:
            colors.append(na_color)
        else:
            colors.append(cmap(len(categories) - 1))
        cmap = ListedColormap(colors)

        return self._show(
            most_urgent,
            interactive,
            overlay_cfg,
            cmap=cmap,
            categorical=True,
            **kwargs,
        )
