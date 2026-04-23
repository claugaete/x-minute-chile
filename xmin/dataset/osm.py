# helper functions to extract subregions from PBF files
from pathlib import Path
import subprocess
import tempfile

from shapely.geometry import MultiPolygon, Polygon

from ..config import config
from .download import makedir


def shapely_to_osmosis_polygon(
    poly: Polygon | MultiPolygon, name: str = "polygon"
) -> str:
    """
    Función auxiliar para convertir un polígono/multipolígono de Shapely en un
    polígono en formato "Osmosis polygon filter file format" para filtrar
    archivos .osm.pbf.

    Parameters
    ---
    poly : Polygon or MultiPolygon
        Polígono a convertir.
    name : str, default: "polygon"
        Nombre del polígono convertido.

    Returns
    ---
    Texto correspondiente al polígono en formato "Osmosis polygon filter file
    format".
    """
    lines = [name]
    if isinstance(poly, Polygon):
        poly = MultiPolygon([poly])
    for i, subpoly in enumerate(poly.geoms):

        # exterior coords
        lines.append(str(i))
        for x, y in subpoly.exterior.coords:
            lines.append(f"    {x:.7f} {y:.7f}")
        lines.append("END")

        # interior coords (holes)
        for j, interior in enumerate(subpoly.interiors):
            lines.append(f"{i}-{j}")
            for x, y in interior.coords:
                lines.append(f"    {x:.7f} {y:.7f}")
            lines.append("END")
    lines.append("END")
    return "\n".join(lines)


def extract_osm_subset(
    inpath: str | Path,
    outpath: str | Path,
    bounds: Polygon | MultiPolygon,
    clip: bool = True,
):
    """
    Extrae una porción de un archivo PBF utilizando osmconvert. Requiere que
    osmconvert esté instalado en el sistema; si osmconvert no está en el PATH,
    se debe modificar la variable `xmin.osmconvert_path` a la ruta del
    ejecutable.

    Parameters
    ---
    inpath : str or Path
        Ruta al archivo PBF de entrada.
    outpath : str or Path
        Ruta al archivo PBF de salida.
    bounds : Polygon or MultiPolygon
        Polígono con la porción a extraer.
    clip : bool, default: True
        Decide si cortar o no las áreas/líneas que estén parcialmente
        contenidas en `bounds`.
    """

    # write bounds to temporary .poly file and use osmconvert
    with tempfile.NamedTemporaryFile(mode="w", suffix=".poly") as tmp:
        tmp.write(shapely_to_osmosis_polygon(bounds))
        tmp.flush()
        makedir(Path(outpath), is_file=True)
        args = [
            config.osmconvert_path,
            inpath,
            f"-B={tmp.name}",
            f"-o={outpath}",
        ]
        if not clip:
            args.append("--complete-ways")
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
