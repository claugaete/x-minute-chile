# X-Minute Chile

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Cálculo de índices de accesibilidad urbana en el contexto chileno

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
└── xmin   <- Código fuente de la librería xmin.
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

--------

