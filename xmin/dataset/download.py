# helper functions for downloading files
from pathlib import Path

from dateutil.parser import parse as parsedate
import requests
from tqdm import tqdm


def download_file(url: str, download_path: Path | str, chunk_size: int = 8192):
    """
    Descarga un archivo, mostrando una barra de progreso

    Parameters
    ---
    url : str
        URL desde la cual se desea descargar el archivo.
    download_path : str
        Ruta en la cual se desea guardar el archivo.
    chunk_size : int, default: 8192
        Número de bytes que se leen a memoria en cada paso (para ir aumentando
        el progreso).
    """
    
    Path(download_path).parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    file_size = response.headers.get("Content-Length")
    last_modified = response.headers.get("Last-Modified")
    print(
        f"Descargando: {Path(download_path).name}, última modificación: "
        + parsedate(last_modified).strftime("%Y-%m-%d")
        if last_modified
        else "N/A"
    )
    if file_size is None:
        progress_bar = tqdm(unit="B", unit_scale=True)
    else:
        progress_bar = tqdm(unit="B", unit_scale=True, total=int(file_size))

    with open(download_path, mode="wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()
