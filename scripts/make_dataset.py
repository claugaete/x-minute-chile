# script para descargar datos de Chile (actualizados)
from abc import ABC, abstractmethod
import argparse
import json
from pathlib import Path
import re
from typing import Callable
import zipfile

from bs4 import BeautifulSoup
import geopandas as gpd
import numpy as np
import pandas as pd
import quackosm as qosm
import rarfile
import requests
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

import xmin
from xmin.dataset.download import download_file, makedir
from xmin.dataset.gtfs import clean_gtfs_frequencies, clean_gtfs_shapes
from xmin.dataset.parks import clean_parks

DATA_PATH = Path(__file__).parent.resolve() / ".." / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
INTERIM_DATA_PATH = DATA_PATH / "interim"
PROCESSED_DATA_PATH = DATA_PATH / "processed"


def unzip(zip_path: Path, out_dir: Path):
    """Extrae un ZIP desde `zip_path` a la carpeta `out_dir`."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def unrar(rar_path: Path, out_dir: Path):
    """Extrae un RAR desde `rar_path` a la carpeta `out_dir`, permitiendo
    manejar errores si no se logra extrar automáticamente."""
    try:
        with rarfile.RarFile(rar_path, "r") as rar_ref:
            rar_ref.extractall(out_dir)
    except rarfile.RarCannotExec:
        failed_unrar_choice = ""
        while failed_unrar_choice not in ("1", "2"):
            failed_unrar_choice = input(
                "rarfile requiere que `unrar` o `unar` se encuentre en el "
                "PATH.\n"
                "(1) Ya agregué la herramienta al PATH y quiero intentar la "
                "extracción automática nuevamente.\n"
                f"(2) Ya extraí manualmente el RAR a {out_dir} y quiero "
                "continuar.\n"
                "Opción escogida (1 o 2): "
            )
        if failed_unrar_choice == "1":
            unrar(rar_path, out_dir)
        else:
            return


def _clean_ide_dataset_numbers(
    extracted_shp_path: Path,
    keep_as_float: list[str] = ["LATITUD", "LONGITUD"],
) -> gpd.GeoDataFrame:
    """
    Lee un GeoDataFrame y cambia todas las columnas que son float a int, excepto latitud y longitud (que deberían mantenerse como float). Esto ocurre para los datasets descargados desde el Geoportal de IDE Chile (https://geoportal.cl/catalog).

    Parameters
    ---
    extracted_shp_path : Path
        Ruta al archivo Shapefile a leer.

    Returns
    ---
    GeoDataFrame con los números corregidos.
    """

    gdf = gpd.read_file(extracted_shp_path)
    int_cols = gdf.select_dtypes(include=["float"]).columns.drop(keep_as_float)
    gdf[int_cols] = gdf[int_cols].astype(int)

    return gdf


class MakeDataset(ABC):
    """
    Interfaz para clases que tienen dos funcionalidades: - Descargar archivos y
    guardarlos en `data/raw`. - Limpiar los archivos descargados y guardarlos
    en `data/processed`.

    Se debe definir el atributo `name` con el nombre que se desea utilizar para
    el dataset en el script principal (que permite seleccionar el dataset a
    descargar).
    """

    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define a 'name' attribute")

    @abstractmethod
    def download(self):
        """Descarga archivos y los guarda en `self.raw_path`."""
        pass

    @abstractmethod
    def clean(self):
        """Limpia los archivos descargados desde `self.raw_path` y los guarda
        en `self.processed_path`."""
        pass

    def download_and_clean(self):
        """Ejecuta `download` seguido de `clean`."""
        self.download()
        self.clean()


class MakeOsm(MakeDataset):
    """Descarga PBF de Chile actualizado, desde Geofabrik."""

    name = "osm"

    def download(self):
        download_file(
            "https://download.geofabrik.de/south-america/chile-latest.osm.pbf",
            RAW_DATA_PATH / "osm" / "Chile.osm.pbf",
        )

    def clean(self):
        pass


class MakeCenso(MakeDataset):
    """Descarga la cartografía del Censo 2024 a nivel país, convirtiendo todas
    las capas al CRS EPSG:4326 (usado en el resto del proyecto)."""

    name = "censo"
    zip_path = RAW_DATA_PATH / "censo" / "Cartografia.zip"

    def download(self):
        download_file(
            "https://storage.googleapis.com/bktdescargascenso2024/"
            "Cartografia/GPKG/Cartografia_censo2024_Pais.zip",
            self.zip_path,
        )

    def clean(self):
        print("Extrayendo ZIP...")
        interim_path = INTERIM_DATA_PATH / "censo"
        unzip(self.zip_path, interim_path)

        input_gpkg = interim_path / "Cartografia_censo2024_Pais.gpkg"
        output_gpkg = PROCESSED_DATA_PATH / "censo" / "Cartografia.gpkg"

        print("Convirtiendo capas a EPSG:4326...")
        layers = gpd.list_layers(input_gpkg)["name"].tolist()
        for i, layer in tqdm(enumerate(layers), total=len(layers)):
            gdf = gpd.read_file(input_gpkg, layer=layer)
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            mode = "w" if i == 0 else "a"
            gdf.to_file(output_gpkg, layer=layer, driver="GPKG", mode=mode)


class MakeGtfsSantiago(MakeDataset):
    """
    Descarga y limpia el GTFS de Santiago, desde la página del DTPM:
    https://www.dtpm.cl/index.php/noticias/gtfs-vigente/. Descarga el primer
    archivo disponible, sin considerar eventos especiales.
    """

    name = "gtfs-santiago"
    zip_path = "gtfs/Santiago.zip"

    def download(self):
        url = "https://www.dtpm.cl/index.php/noticias/gtfs-vigente/"

        # finding the correct download link
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        link = soup.find(
            "a", href=lambda h: h and "/descargas/gtfs" in h.lower()
        )

        download_file(
            "https://www.dtpm.cl/" + link["href"],
            RAW_DATA_PATH / self.zip_path,
        )

    def clean(self):
        makedir(PROCESSED_DATA_PATH / self.zip_path, is_file=True)
        print("Limpiando GTFS...")
        clean_gtfs_frequencies(
            RAW_DATA_PATH / self.zip_path, PROCESSED_DATA_PATH / self.zip_path
        )


class MakeGtfsRegional(MakeDataset):
    """
    Descarga y limpia los GTFS de regiones, obtenidos desde la página de la
    DTPR:
    https://dtpr.gob.cl/tramites-operadores/sistemas-regulados-de-transporte-publico-mayor/.

    Debido a la poca estructura de la página de la DTPR, se deben agregar
    manualmente los enlaces a descargar. Esto implica que no se agregarán
    nuevas ciudades automáticamente aunque estas aparezcan en la página de la
    DTPR, y es muy posible que el método `download()` se rompa si la DTPR
    decide cambiar los enlaces de descarga.
    """

    name = "gtfs-regional"

    # regiones con GTFS disponible, y su enlace de descarga
    regions = {
        "Antofagasta": "2025/12/antofagasta.zip",
        "Buin_Paine": "2025/12/buin-Paine.zip",
        "Calama": "2025/12/calama.zip",
        "Castro": "2025/12/castro.zip",
        "Chillan": "2022/08/chillan.zip",
        "Concepcion": "2025/12/granconcepcion.zip",
        "Iquique": "2025/12/iquique.zip",
        "Linares": "2025/12/linares.zip",
        "Osorno": "2025/12/osorno.zip",
        "Puerto_Montt": "2025/12/Puerto_Montt.zip",
        "Punta_Arenas": "2025/12/PuntaArenas.zip",
        "Quellon": "2025/12/quellon.zip",
        "Quintero_Puchuncavi": "2025/12/Quintero-Puchuncavi.zip",
        "Temuco": "2025/12/Temuco.zip",
        "Tocopilla": "2025/12/tocopilla.zip",
        "Tome": "2025/12/tome.zip",
        "Valdivia": "2025/12/valdivia.zip",
        "Valparaiso": "2025/12/GranValparaiso.zip",
        "Villarrica": "2025/12/villarrica.zip",
    }

    def download(self):
        base_url = "https://dtpr.gob.cl/wp-content/uploads/"
        for name, url in tqdm(self.regions.items()):
            download_file(
                base_url + url,
                RAW_DATA_PATH / "gtfs" / f"{name}.zip",
                leave=False,
            )

    def clean(self):
        makedir(PROCESSED_DATA_PATH / "gtfs")
        print("Limpiando archivos GTFS...")
        pbar = tqdm(self.regions.keys())
        for name in pbar:
            pbar.set_description(f"{name}.zip")
            clean_gtfs_shapes(
                RAW_DATA_PATH / "gtfs" / f"{name}.zip",
                PROCESSED_DATA_PATH / "gtfs" / f"{name}.zip",
            )


class MakeFarmacias(MakeDataset):
    """
    Descarga y limpia datos de farmacias en Chile, actualizados diariamente por
    el Ministerio de Salud.
    """

    name = "farmacias"

    json_path = RAW_DATA_PATH / "amenities" / "farmacias" / "farmacias.json"

    def _fix_coord_format(self, coord_str: str) -> str:
        """
        Corrige posibles errores de formato de una string que representa una
        coordenada (p. ej. -70.543). Los errores que corrige son los
        siguientes:

        - Un guión en vez de un punto para el separador decimal.
        - Uno o más caracteres extra al final del número (por ejemplo, comas o
        símbolos de grado (°)).

        Si el string tiene otro tipo de error (p. ej. no es un número en
        absoluto), se retorna el string original.

        Parameters
        ---
        coord_str : str
            String a corregir.

        Returns
        ---
        String corregido si se encontró un error corregible; string original si
        no.
        """

        # optional leading dash, followed by digits, followed by period/dash,
        # followed by digits, followed by extra stuff
        pattern = r"^(-?\d+)[.-](\d+)[^\d]*$"
        match = re.match(pattern, coord_str)

        if match:
            return f"{match.group(1)}.{match.group(2)}"

        # if no match is captured, return string as-is
        return coord_str

    def _fix_coord_series_format(self, coord_series: pd.Series) -> pd.Series:
        """
        Aplica `self._fix_cord_format` a una serie de pandas e intenta
        convertir los strings a números. Los strings que no se pueden convertir
        serán retornados como nulos.

        Parameters
        ---
        coord_series : pd.Series
            Serie con strings que representan coordenadas.

        Returns
        ---
        Serie con números que representan las coordenadas corregidas (o nulo si
        no se logró convertir a un número).
        """
        return pd.to_numeric(
            coord_series.apply(self._fix_coord_format), errors="coerce"
        )

    def download(self):
        download_file(
            "https://midas.minsal.cl/farmacia_v2/WS/getLocales.php",
            self.json_path,
        )

    def clean(self):
        print("Creando archivo GeoPackage...")
        with open(self.json_path, "r") as file:
            data = json.load(file)
        farmacias_df = pd.DataFrame(data)
        farmacias_gdf = gpd.GeoDataFrame(
            farmacias_df,
            geometry=gpd.points_from_xy(
                self._fix_coord_series_format(farmacias_df["local_lng"]),
                self._fix_coord_series_format(farmacias_df["local_lat"]),
                crs=4326,
            ),
        )

        # quitamos columnas innecesarias y renombramos otras
        farmacias_gdf = farmacias_gdf.drop(
            columns=["fecha", "funcionamiento_dia"]
        ).rename(
            columns={
                "local_id": "id",
                "local_nombre": "name",
                "comuna_nombre": "comuna",
                "localidad_nombre": "localidad",
                "local_direccion": "direccion",
            }
        )

        gpkg_path = PROCESSED_DATA_PATH / "amenities" / "farmacias.gpkg"
        makedir(gpkg_path, is_file=True)
        farmacias_gdf.to_file(gpkg_path)


class MakeSalud(MakeDataset):
    """
    Descarga y limpia datos de establecimientos de salud en Chile, de diciembre
    de 2025. Estos datos son entregados por el Ministerio de Salud en el
    siguiente enlace:
    https://geoportal.cl/geoportal/catalog/36779/Establecimientos%20de%20salud%20de%20Chile%20Diciembre%202025.
    """

    name = "salud"
    zip_path = (
        RAW_DATA_PATH / "amenities" / "salud" / "establecimientos_salud.zip"
    )

    def download(self):
        download_file(
            "https://geoportal.cl/geoportal/catalog/download/5b2f29d1-94d4-398f-a207-7f9f3056e5d1",
            self.zip_path,
        )

    def clean(self):
        print("Extrayendo ZIP...")
        interim_path = INTERIM_DATA_PATH / "amenities" / "salud"
        dest_path = PROCESSED_DATA_PATH / "amenities"
        unzip(self.zip_path, interim_path)

        print("Creando archivo GeoPackage...")
        salud_gdf = _clean_ide_dataset_numbers(
            interim_path
            / "l_910_v1_establecimientos_de_salud_diciembre_2025.shp"
        )
        salud_gdf = salud_gdf.rename(columns={"ID_ORIG": "id"})
        salud_gdf["F_INICIO"] = pd.to_datetime(salud_gdf["F_INICIO"])

        makedir(dest_path)
        salud_gdf.to_file(dest_path / "salud.gpkg")


class MakeEducacion(MakeDataset):
    """
    Descarga y limpia datasets de establecimientos educacionales en Chile,
    descargando datos proporcionados por el Ministerio de Educación a través
    del Geoportal IDE. Se cuenta con los siguientes datasets:

    - Establecimientos de educación parvularia, año 2021:
      https://geoportal.cl/geoportal/catalog/35553/Establecimientos%20Educaci%C3%B3n%20Parvularia.
    - Establecimientos de educación escolar, año 2021:
      https://geoportal.cl/geoportal/catalog/35408/Establecimientos%20Educaci%C3%B3n%20Escolar.
    - Establecimientos de educación superior, año 2020:
      https://geoportal.cl/geoportal/catalog/35554/Establecimientos%20de%20Educaci%C3%B3n%20Superior.

    Parameters
    ---
    cluster_min_eps : float, default: 500
        Distancia utilizada como `min_eps` en el algoritmo de clustering DBSCAN
        para agrupar edificios cercanos de una misma universidad bajo el mismo
        "campus".
    """

    name = "educacion"

    def __init__(self, cluster_min_eps: float = 500):
        super().__init__()
        self.cluster_min_eps = cluster_min_eps

    def zip_path(self, name):
        """Ruta del ZIP para cada dataset de educación."""
        return (
            RAW_DATA_PATH / "amenities" / "educacion" / "{}.zip".format(name)
        )

    def _clean_parvularia(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf["id"] = gdf.index
        return gdf.rename(columns={"NOM_ESTAB": "name"}).drop(columns="AGNO")

    def _clean_escolar(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gdf.rename(columns={"RBD": "id", "NOM_RBD": "name"}).drop(
            columns="AGNO"
        )

    def _clean_superior(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Agrupa edificios cercanos que pertenecen a la misma universidad,
        reduciendo sus pesos para no considerarlos como distintas universidades
        sino que como una sola universidad con varias "entradas".
        """

        gdf = gdf.assign(
            name=gdf["NOMBRE_INS"] + " - " + gdf["NOMBRE_INM"],
            id=gdf.index,
        ).drop(columns="AÑO")
        gdf_proj = gdf.to_crs(xmin.projected_crs)

        # guardamos índices y sus clusters, ordenados por universidad
        all_indices = []
        all_labels = []

        for _, group in gdf_proj.groupby("COD_INST"):
            if len(group) == 1:
                labels = [0]
            else:
                # obtenemos coordenadas de edificios
                coords = np.array(
                    [[geom.x, geom.y] for geom in group.geometry]
                )

                # DBSCAN clustering
                labels = DBSCAN(
                    eps=self.cluster_min_eps, min_samples=1, metric="euclidean"
                ).fit_predict(coords)

            all_indices.extend(group.index)
            all_labels.extend(labels)

        # asignamos cluster a cada edificio en el gdf
        label_series = pd.Series(all_labels, index=all_indices)
        gdf["_cluster"] = (
            gdf["COD_INST"].astype(str) + "__" + label_series.astype(str)
        )
        cluster_sizes = gdf["_cluster"].map(gdf["_cluster"].value_counts())
        gdf["weight"] = 1 / cluster_sizes

        return gdf.drop(columns="_cluster")

    def download(self):
        urls = {
            "parvularia": "https://www.geoportal.cl/geoportal/catalog/download/b52e4229-365e-3163-b211-679cc6b2fd99",
            "escolar": "https://www.geoportal.cl/geoportal/catalog/download/d6bf9431-3282-3738-bd69-baf0d1ad63ec",
            "superior": "https://www.geoportal.cl/geoportal/catalog/download/0dde8427-113a-356a-bace-ed4d51ddcb05",
        }
        for name, url in tqdm(urls.items()):
            download_file(url, self.zip_path(name), leave=False)

    def clean(self):
        filenames = {
            "parvularia": "layer_establecimientos_educacion_parvularia_20220309024143.shp",
            "escolar": "layer_establecimientos_educacion_escolar_20220309024120.shp",
            "superior": "layer_establecimientos_de_educacion_superior_20220309024111.shp",
        }
        clean_functions: dict[
            str, Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]
        ] = {
            "parvularia": self._clean_parvularia,
            "escolar": self._clean_escolar,
            "superior": self._clean_superior,
        }
        interim_path = INTERIM_DATA_PATH / "amenities" / "educacion"
        gpkg_path = PROCESSED_DATA_PATH / "amenities" / "educacion.gpkg"

        print("Extrayendo ZIPs...")
        for name in filenames.keys():
            unzip(self.zip_path(name), interim_path)

        print("Creando archivo GeoPackage...")
        makedir(gpkg_path, is_file=True)
        for name, clean in clean_functions.items():
            gdf = _clean_ide_dataset_numbers(interim_path / filenames[name])
            cleaned_gdf = clean(gdf)
            cleaned_gdf.to_file(gpkg_path, layer=name, driver="GPKG")


class MakeAreasVerdes(MakeDataset):
    """
    Descarga y limpia datos de áreas verdes urbanas (plazas y parques), según
    la cartografía de Indicadores de Calidad de Plazas y Parques Urbanos, del
    Instituto Nacional de Estadísticas (2019): https://arcg.is/1LTLCf

    La limpieza involucra convertir los polígonos que representan a las
    distintas plazas y parques en puntos. Si `n` puntos pertenecen a la misma
    área verde (igual nombre y comuna), entonces cada punto tendrá un peso
    `1/n` en su columna `weight`. Si un área verde no tiene nombre, se
    considera por sí sola (no se combina con otras áreas verdes aledañas, pues
    no hay manera de saber si pertenecen al mismo complejo o no). Para evitar
    que un área con muchas áreas verdes pequeñas sin nombre tenga una
    accesibilidad excesivamente alta, se recomienda ponderar cada punto por el
    tamaño del área verde correspondiente, y/o eliminar las áreas verdes sin
    nombre.

    Parameters
    ---
    min_dist : float, default: 200
        Distancia mínima (en metros) que deben tener dos puntos representativos
        de una misma área verde.
    """

    name = "areas-verdes"
    rar_path = RAW_DATA_PATH / "amenities" / "verdes" / "verdes.rar"

    def __init__(self, min_dist: float = 200):
        super().__init__()
        self.min_dist = min_dist

    def download(self):
        download_file(
            "https://geoarchivos.ine.cl/Files/Calidad_PlPq/SHP.rar",
            self.rar_path,
        )

    def clean(self):
        print("Extrayendo RAR...")
        inter_dir = INTERIM_DATA_PATH / "amenities" / "verdes"
        unrar(self.rar_path, inter_dir)

        print("Leyendo GeoDataFrames...")
        verdes_g1g2 = gpd.read_file(
            inter_dir / "CALIDAD_pzpq_2019_G1G2.shp"
        ).to_crs(4326)
        verdes_g3g4 = gpd.read_file(inter_dir / "PZPQ_2018_G3G4.shp").to_crs(
            4326
        )
        verdes_gdf = gpd.GeoDataFrame(
            pd.concat([verdes_g1g2, verdes_g3g4], ignore_index=True), crs=4326
        )
        verdes_gdf["fenced"] = verdes_gdf["Cierres"].isin(
            ["Con_cierre_perim_gratuito", "Con_cierre_perim_pagado"]
        )
        verdes_gdf["name"] = (
            verdes_gdf["TIPO_EP"]
            + " "
            + verdes_gdf["NOMBRE_EP"].fillna(
                verdes_gdf.index.to_series().astype(str)
            )
            + " "
            + verdes_gdf["COMUNA"]
        )

        # use quackosm to extract roads intersecting parks
        print("Obteniendo rutas al interior de áreas verdes...")
        chile_pbf_path = RAW_DATA_PATH / "osm" / "Chile.osm.pbf"
        verdes_union = verdes_gdf.union_all()
        roads_gdf = qosm.convert_pbf_to_geodataframe(
            pbf_path=chile_pbf_path,
            tags_filter={
                "highway": [
                    "footway",
                    "path",
                    "pedestrian",
                    "steps",
                    "living_street",
                    "residential",
                    "service",
                ]
            },
            working_directory=xmin.quackosm_working_directory,
            geometry_filter=verdes_union,
            keep_all_tags=False,
        )

        # assign representative points
        print("Asignando puntos representativos a áreas verdes...")
        points_gdf = clean_parks(
            verdes_gdf,
            roads_gdf,
            is_fenced_column="fenced",
            index_column="name",
            min_dist=self.min_dist,
        )
        points_gdf = points_gdf.assign(id=points_gdf.index)

        # split parks and plazas, and save
        print("Creando archivo GeoPackage...")
        gpkg_path = PROCESSED_DATA_PATH / "amenities" / "areas_verdes.gpkg"
        makedir(gpkg_path, is_file=True)
        parks_gdf = points_gdf[points_gdf["TIPO_EP"] == "PARQUE"]
        plazas_gdf = points_gdf[points_gdf["TIPO_EP"] == "PLAZA"]
        parks_gdf.to_file(gpkg_path, layer="parques", driver="GPKG")
        plazas_gdf.to_file(gpkg_path, layer="plazas", driver="GPKG")


if __name__ == "__main__":

    make_osm = MakeOsm()
    make_censo = MakeCenso()
    make_gtfs_santiago = MakeGtfsSantiago()
    make_gtfs_regional = MakeGtfsRegional()
    make_salud = MakeSalud()
    make_farmacias = MakeFarmacias()
    make_educacion = MakeEducacion()
    make_areas_verdes = MakeAreasVerdes()

    all_datasets: list[MakeDataset] = [
        make_osm,
        make_censo,
        make_gtfs_santiago,
        make_gtfs_regional,
        make_salud,
        make_farmacias,
        make_educacion,
        make_areas_verdes,
    ]

    # datasets que reciben actualizaciones frecuentes (para evitar descargar
    # los que no se actualizan frecuentemente)
    updated_datasets: list[MakeDataset] = [
        make_osm,
        make_gtfs_santiago,
        make_farmacias,
    ]

    options = "\n".join(
        [
            "- all: todos los datasets",
            "- update: todos los datasets que reciben actualizaciones "
            "frecuentes",
        ]
        + [f"- {dataset.name}" for dataset in all_datasets]
    )

    parser = argparse.ArgumentParser(
        description="Descarga y/o limpia los datasets solicitados."
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Nombre del dataset a procesar. Opciones:\n" + options,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Solo descargar el dataset solicitado.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Solo limpiar el dataset solicitado.",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        datasets_to_process = all_datasets
    elif args.dataset == "update":
        datasets_to_process = updated_datasets
    else:
        datasets_to_process = [
            dataset for dataset in all_datasets if dataset.name == args.dataset
        ]
        if not datasets_to_process:
            raise ValueError(
                "El dataset solicitado no existe, por favor solicitar un "
                "dataset entre las opciones:\n" + options
            )

    for dataset in datasets_to_process:
        print(f"\n--- {dataset.name.upper()} ---")
        if not args.download and not args.clean:
            dataset.download_and_clean()
        if args.download:
            dataset.download()
        if args.clean:
            dataset.clean()
