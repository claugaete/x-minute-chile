# X-Minute Chile

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Introduction (English)

X-Minute Chile is a project focused on calculating urban accessibility indices
for different Chilean cities, in the context of the X-minute city concept (a
generalization of the [15-minute
city](https://en.wikipedia.org/wiki/15-minute_city)). The project consists of:
a library called `xmin`, which provides tools for calculating accessibility
indices flexibly and reproducibly for urban areas around the world; and scripts
and notebooks that utilize `xmin` to analyze accessibility in different Chilean
cities and exemplify use cases for the library.

The directory [`notebooks/en`](notebooks/en) contains Jupyter notebooks written
in English, applying `xmin` in multiple contexts (including an analysis of
cities outside Chile). The rest of this repository (including documentation for
`xmin`) is currently in Spanish.

## Introducción

X-Minute Chile es un proyecto enfocado en el cálculo de índices de
accesibilidad urbana para distintas ciudades chilenas, en el contexto de la
ciudad de X minutos (una generalización del concepto de la [ciudad de 15
minutos](https://en.wikipedia.org/wiki/15-minute_city)). El proyecto se puede
dividir en dos partes:

1. La librería `xmin`, que entrega herramientas para calcular índices de
   accesibilidad de manera flexible y replicable en cualquier área urbana del
   mundo.
1. Scripts y notebooks que utilizan `xmin` para realizar análisis de
   accesibilidad de distinta complejidad en ciudades de Chile, ejemplificando
   casos de uso de la librería.

X-Minute Chile fue desarrollado por Claudio Gaete como Memoria para optar al
título de Ingeniero Civil en Computación en la Universidad de Chile.

![Comparación de acceso a farmacias en distintas ciudades de
Chile](reports/figures/multiple_city_comparison.png)

## Índice de contenidos

- [Organización del proyecto](#organización-del-proyecto)
- [Instalación y requisitos](#instalación-y-requisitos)
    * [Instalación de librería](#instalación-de-librería)
    * [Ejecución de análisis](#ejecución-de-análisis)
    * [Dependencias](#dependencias)

## Organización del proyecto

```
├── LICENSE            <- Licencia MIT
├── Makefile           <- Makefile para facilitar la ejecución de comandos
|                         comunes.
├── README.md          <- ¡El archivo que estás leyendo! Resumen del proyecto,
|                         sus requisitos y estructura.
├── data
│   ├── interim        <- Datos intermedios que han sido extraídos y/o
|   |                     modificados.
│   ├── processed      <- Datos finales, utilizados en los análisis.
│   └── raw            <- Datos originales, tal y como fueron descargados.
│
├── notebooks          <- Notebooks de Jupyter. Incluye exploración y ejemplos
|                         de casos de uso de la librería, aplicados al contexto
|                         chileno.
│
├── pyproject.toml     <- Archivo de configuración del proyecto.
│
├── reports            <- Reportes finales generados (en formato PDF).
│   └── figures        <- Figuras generadas en los notebooks y utilizadas en
|                         los reportes.
│
├── scripts            <- Scripts para la descarga y procesamiento de datos de
|                         Chile.
│
├── uv.lock            <- Lockfile creado por uv.
│
└── xmin               <- Código fuente de la librería xmin.
    │
    ├── __init__.py             <- Permite que xmin sea un módulo de Python.
    │
    ├── amenities.py            <- Generación de necesidades y destinos.
    │
    ├── config.py               <- Guarda variables de configuración.
    │
    ├── dataset                 <- Submódulo auxiliar con funciones para
    |                              descargar y procesar datos.
    │
    ├── geometry.py             <- Funciones auxiliares para tratar con
    |                              geometrías.
    │
    ├── indices.py              <- Definición de funciones/índices de
    |                              accesibilidad.
    │
    ├── origins.py              <- Generación de orígenes.
    │
    ├── ratings.py              <- Cálculo de puntuaciones de accesibilidad.
    │
    ├── travel_time.py          <- Cálculo de matrices de tiempo de viaje.
    │
    └── visualization.py        <- Generación de visualizaciones.
```

## Instalación y requisitos

X-Minute Chile fue desarrollado con [`uv`](https://docs.astral.sh/uv/), por lo
que se requiere instalarlo para ejecutar el proyecto. Las instrucciones de
instalación se pueden encontrar
[aquí](https://docs.astral.sh/uv/getting-started/installation/).

### Instalación de librería

Para instalar la librería, primero se debe clonar el proyecto y entrar en el
directorio correspondiente al repositorio.

```
git clone https://github.com/claugaete/x-minute-chile.git
cd x-minute-chile
```

Luego, basta con ejecutar `uv sync` para instalar todas las dependencias
necesarias para utilizar la librería.

### Ejecución de análisis

Si se desea ejecutar los notebooks con análisis aplicados a Chile, primero se
deben descargar y procesar los datos necesarios. Para esto, se ejecuta **uno**
de los siguientes comandos (ambos hacen lo mismo):

```
make dataset_all
uv run scripts/make_dataset.py all
```

Si se desea actualizar los datasets que cambian con cierta frecuencia, basta
con ejecutar **uno** de los siguientes comandos:

```
make dataset_update
uv run scripts/make_dataset.py update
```

Habiendo descargado y procesado los datos, se pueden ejecutar todos los
notebooks excepto el de [accesibilidad de segundo
orden](notebooks/use-cases/accessibilidad_segundo_orden.ipynb), pues este
requiere un previo *geocoding* de direcciones de empleos formales, que no se
realiza por defecto al procesar los datos. El mismo notebook contiene la
información necesaria para procesar los datos y poder ejecutar el análisis.

### Dependencias

`uv sync` instala múltiples dependencias. Aquí se listan, clasificadas según
los módulos del proyecto en que cada una es utilizada, e indicando los motivos
por los cuales se instaló cada una:

- Librerías esenciales (utilizadas en múltiples módulos):
    - `pip`.
    - `geopandas`.
    - `pandas` y `shapely` (utilizados por `geopandas`).
    - `numpy`.
    - `tqdm`.
- Generación de orígenes:
    - `h3` y `tobler`: generación de celdas H3 y *area-weighted interpolation*
      para la asignación de poblaciones.
- Generación de destinos:
    - `quackosm` y `duckdb`: obtención de puntos de interés a partir de
      archivos PBF de OpenStreetMap.
- Cálculo de tiempos de viaje:
    - `r5py`: cálculo de matrices de tiempo de viaje (TTMs).
- Visualizaciones:
    - `matplotlib`: generación de visualizaciones estáticas.
    - `folium`: generación de visualizaciones interactivas.
    - `matplotlib-map-utils`: inclusión de escala gráfica en visualizaciones
      estáticas.
    - `contextily`: inclusión de mapa base en visualizaciones estáticas.
    - `mapclassify`: clasificación de variables en *bins* discretos
      (visualización de coropletas bivariadas).
    - `esda`: cálculo de indicadores locales (visualización de LISA).
- Carga y procesamiento de datos:
    - `python-dotenv`: carga de variables de entorno.
    - `partridge`: procesamiento de *feeds* GTFS.
    - `beautifulsoup4`: *web scraping* para descarga de datos.
    - `openpyxl`: manejo de datos en formato Excel.
    - `scikit-learn`: métodos de *clustering* para el agrupamiento espacial de
      datos.
    - `nominatim-api`: *geocoding* de fuentes de empleo formal (no se instala
      por defecto).
- Notebooks: `ipykernel` y `ipywidgets`.
- Desarrollo:
    - `ruff`: linting y formateo.

--------

