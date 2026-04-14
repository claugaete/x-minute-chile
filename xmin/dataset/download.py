# helper functions for downloading files
from pathlib import Path
import warnings

from dateutil.parser import parse as parsedate
import requests
from tqdm import tqdm


def makedir(path: Path, is_file: bool = False) -> None:
    """
    Revisa si un directorio existe, creándolo si no es el caso. Si
    `parent=True`, se asume que `path` es la ruta de un archivo, y se revisa si
    existe el directorio que lo contiene (su padre).
    """

    path_res = path.resolve()
    if is_file:
        path_res = path_res.parent

    if not path_res.exists():
        warnings.warn(f"La ruta {path_res} no existe, por lo que será creada.")
        path_res.mkdir(parents=True)


def download_file(
    url: str, download_path: Path | str, chunk_size: int = 8192, **kwargs
):
    """
    Descarga un archivo, mostrando una barra de progreso y asegurándose que el
    directorio exista antes de guardar el archivo.

    Parameters
    ---
    url : str
        URL desde la cual se desea descargar el archivo.
    download_path : str
        Ruta en la cual se desea guardar el archivo.
    chunk_size : int, default: 8192
        Número de bytes que se leen a memoria en cada paso (para ir aumentando
        el progreso).
    **kwargs
        Argumentos que serán pasados a la barra de progreso, creada con
        `tqdm.tqdm`.
    """

    makedir(Path(download_path), is_file=True)

    response = requests.get(url, stream=True)
    file_size = response.headers.get("Content-Length")
    last_modified = response.headers.get("Last-Modified")
    desc = f"Descargando {Path(download_path).name}, últ. mod.: " + (
        parsedate(last_modified).strftime("%Y-%m-%d")
        if last_modified
        else "N/A"
    )
    if file_size is None:
        progress_bar = tqdm(unit="B", unit_scale=True, desc=desc, **kwargs)
    else:
        progress_bar = tqdm(
            unit="B",
            unit_scale=True,
            total=int(file_size),
            desc=desc,
            **kwargs,
        )

    with open(download_path, mode="wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()
