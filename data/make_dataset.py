# script para descargar datos de Chile (actualizados)
import os
from pathlib import Path
import zipfile

from xmin.dataset.download import download_file

DEFAULT_DATA_PATH = Path(__file__).parent.resolve() / "raw"


def download_osm(path: Path = DEFAULT_DATA_PATH):
    """Descarga PBF de Chile actualizado, desde Geofabrik."""
    download_file(
        "https://download.geofabrik.de/south-america/chile-latest.osm.pbf",
        path / "osm" / "Chile.osm.pbf",
    )


def download_censo(path: Path = DEFAULT_DATA_PATH):
    """Descarga la cartografía del Censo 2024 a nivel país."""

    folder_path = path / "censo"
    zip_path = folder_path / "Cartografia.zip"

    download_file(
        "https://storage.googleapis.com/bktdescargascenso2024/"
        "Cartografia/GPKG/Cartografia_censo2024_Pais.zip",
        zip_path,
    )

    print("Extrayendo ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(zip_path)
    os.rename(
        folder_path / "Cartografia_censo2024_Pais.gpkg",
        folder_path / "Cartografia.gpkg",
    )


if __name__ == "__main__":
    print("--- OSM ---")
    download_osm()
    
    print("\n--- CENSO ---")
    download_censo()
