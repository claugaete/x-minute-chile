# script para descargar datos de Chile (actualizados)
from abc import ABC, abstractmethod
import os
from pathlib import Path
import zipfile

from xmin.dataset.download import download_file

DATA_PATH = Path(__file__).parent.resolve() / ".." / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"


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
    """Descarga la cartografía del Censo 2024 a nivel país."""

    name = "censo"

    def download(self):
        out_path = RAW_DATA_PATH / "censo"
        zip_path = out_path / "Cartografia.zip"

        download_file(
            "https://storage.googleapis.com/bktdescargascenso2024/"
            "Cartografia/GPKG/Cartografia_censo2024_Pais.zip",
            zip_path,
        )

        print("Extrayendo ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(out_path)
        os.remove(zip_path)
        os.rename(
            out_path / "Cartografia_censo2024_Pais.gpkg",
            out_path / "Cartografia.gpkg",
        )

    def clean(self):
        pass


if __name__ == "__main__":
    make_osm = MakeOsm()
    make_censo = MakeCenso()

    all_datasets: list[MakeDataset] = [make_osm, make_censo]

    # datasets que reciben actualizaciones frecuentes (para evitar descargar
    # los que no se actualizan frecuentemente)
    updated_datasets: list[MakeDataset] = [make_osm]

    for dataset in all_datasets:
        print(f"\n--- {dataset.name.upper()} ---")
        dataset.download_and_clean()
