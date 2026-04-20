# helper geometry functions
import warnings

import geopandas as gpd
import numpy as np
from shapely import MultiPoint, Point, Polygon

import xmin


def to_centroids(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Convierte las geometrías de un GeoDataFrame que no sean puntos,
    cambiándolas por sus centroides.

    Parameters
    ---
    gdf : GeoDataFrame
        GeoDataFrame para el cual cambiar su geometría.

    Returns
    ---
    Un nuevo GeoDataFrame con las geometrías modificadas.
    """
    original_crs = gdf.crs
    return gdf.assign(
        geometry=gdf.geometry.to_crs(xmin.projected_crs).centroid.to_crs(
            original_crs
        )
    )


def convert_polygon_to_representative_points(
    polygon: Polygon,
    entry_points: list[Point] | MultiPoint = [],
    add_extra_points: bool = True,
    min_dist: float | None = None,
    max_dist: float | None = None,
) -> list[Point]:
    """
    Convierte un polígono en una lista de puntos representativos alrededor del
    borde del polígono. Considera las siguientes restricciones:

    - Dos puntos consecutivos deben estar a una distancia igual o mayor a
      `min_dist`.
    - Dos puntos consecutivos deben estar a una distancia menor o igual a
      `max_dist` (a no ser que `add_extra_points` sea falso, con lo cual no se
      agregan puntos intermedios para cumplir esta restricción).
    - Se deben mantener los puntos de `entry_points`, a no ser que dos de ellos
      estén a una distancia menor que `min_dist`; en ese caso, se elimina uno
      de los puntos.

    Todas las distancias anteriores se refieren a la distancia recorriendo el
    borde del polígono (que puede ser mayor a la distancia euclidiana).

    Parameters
    ---
    polygon : Polygon
        Polígono que se desea convertir a sus puntos representativos.
    entry_points : list[Point] or MultiPoint, default: []
        Lista de puntos de entrada que se desean mantener.
    add_extra_points: bool, default: True
        Si se desea agregar puntos intermedios cuando los puntos de
        `entry_points` estén muy alejados entre sí.
    min_dist : float or None, default: None
        Distancia mínima que debe haber entre dos puntos representativos
        consecutivos. Debe cumplirse que `min_dist <= max_dist/2`. Si es None,
        se asignará `min_dist = max_dist/2`.
    max_dist : float or None, default: None
        Distancia máxima que debe haber entre dos puntos representativos
        consecutivos. Debe cumplirse que `max_dist >= 2*min_dist`. Si es None,
        se asignará `max_dist = 2*min_dist`.
    """

    if max_dist is None:
        if min_dist is None:
            raise ValueError("Se debe definir `max_dist` o bien `min_dist`.")
        else:
            max_dist = 2 * min_dist
    else:
        if min_dist is None:
            min_dist = max_dist / 2
        elif min_dist > max_dist / 2:
            raise ValueError(
                "`max_dist` debe ser al menos el doble de `min_dist`."
            )

    if not (add_extra_points or entry_points):
        warnings.warn(
            "No hay puntos de entrada y `add_extra_points` es falso; se "
            "retornará una lista sin puntos."
        )
        return []

    ring = polygon.exterior

    # REMOVE ENTRIES THAT ARE TOO CLOSE
    # convert each entry point to its distance along the ring
    snapped_entries = []
    for pt in entry_points:
        dist = ring.project(pt)
        snapped = ring.interpolate(dist)
        snapped_entries.append((dist, snapped))
    snapped_entries.sort(key=lambda x: x[0])

    # remove entry points that are closer than min_dist to their predecessor
    kept_entries: list[tuple[float, Point]] = []
    for dist, pt in snapped_entries:
        if kept_entries and (dist - kept_entries[-1][0]) < min_dist:
            continue
        kept_entries.append((dist, pt))
    # compare last point to first point
    if len(kept_entries) >= 2:
        first_last_dist = ring.length - (
            kept_entries[-1][0] - kept_entries[0][0]
        )
        if first_last_dist < min_dist:
            kept_entries.pop()

    if not add_extra_points:
        return [pt for _, pt in kept_entries]

    # ADD INTERMEDIATE POINTS
    # add random entry point if none are given
    if len(kept_entries) == 0:
        kept_entries.append((0, ring.coords[0]))

    # create anchors (entries + first entry repeated, so that every space
    # between entries is considered)
    anchor_dists = [dist for dist, _ in kept_entries] + [
        ring.length + kept_entries[0][0]
    ]

    # add points between anchors if they're too far apart
    final_points = []
    for i in range(len(anchor_dists) - 1):
        if anchor_dists[i + 1] - anchor_dists[i] > max_dist:
            # NOTE: the distance between the last point in `np.arange(start,
            # stop, step)` and `stop` is always <= step. in this case, because
            # `stop=anchor_dists[i+1] - min_dist`, the distance between the
            # last point and `anchor_dists[i+1]` is <= `2*min_dist` <=
            # `max_dist`, and it's also at least `min_dist` apart (because the
            # last point can't be after `stop` and `stop` is `min_dist` apart
            # from `anchor_dists[i+1]`). thus, all restrictions between points
            # apply.
            inter_dists = np.arange(
                anchor_dists[i], anchor_dists[i + 1] - min_dist, min_dist
            )
            final_points += [ring.interpolate(dist) for dist in inter_dists]
        else:
            final_points.append(ring.interpolate(anchor_dists[i]))

    return final_points
