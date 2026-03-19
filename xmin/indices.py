from collections.abc import Callable
import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyrosm
import r5py

import xmin
from xmin.amenities import Amenity
from xmin.origins import Origins


def try_snap_to_network(
    transport_network: r5py.TransportNetwork,
    point_gdf: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame | None = None,
):
    """
    TODO
    """

    snapped_points: gpd.GeoSeries = transport_network.snap_to_network(
        point_gdf.geometry, street_mode=r5py.TransportMode.CAR
    )

    # if point is empty, or falls outside its designated area, undo snapping
    # and return original point
    return point_gdf.assign(
        geometry=np.where(
            snapped_points.is_empty
            | (
                area_gdf is not None
                and ~area_gdf.contains(snapped_points.geometry)
            ),
            point_gdf.geometry,
            snapped_points.geometry,
        )
    )


def calculate_amenity_index(
    origins: Origins,
    amenity: Amenity,
    index_function: Callable[..., pd.Series],
    gtfs_path: Path,
    osm_path: Path,
    snap_to_network: bool = False,
    **index_function_kwargs,
):
    # load r5py and pois
    transport_network = r5py.TransportNetwork(osm_path, [gtfs_path])
    osm_object = pyrosm.OSM(str(osm_path))

    amenity.get_pois(osm_object)

    origin_points_gdf = origins.h3_grid.assign(
        geometry=origins.h3_grid.geometry.to_crs(
            xmin.projected_crs
        ).centroid.to_crs(4326)
    )

    amenity_gdf = amenity.amenity_gdf
    if "weight" not in amenity_gdf.columns:
        amenity_gdf["weight"] = 1

    if snap_to_network:
        origin_points_gdf = try_snap_to_network(
            transport_network, origin_points_gdf, origins.h3_grid
        )
        amenity_gdf = try_snap_to_network(transport_network, amenity_gdf)

    travel_time_matrix = r5py.TravelTimeMatrix(
        transport_network,
        origins=origin_points_gdf,
        destinations=amenity_gdf,
        max_time=datetime.timedelta(minutes=60),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        departure=datetime.datetime(2025, 10, 20, 10, 0, 0),
        departure_time_window=datetime.timedelta(minutes=30),
    )

    accessibility = index_function(
        travel_time_matrix,
        origin_points_gdf.set_index("id")["population"],
        amenity_gdf.set_index("id")["weight"],
        **index_function_kwargs,
    )

    return origins.h3_grid.set_index("id").assign(accessibility=accessibility)


def binary(
    travel_times: pd.DataFrame,
    population: pd.Series,
    amenity_weights: pd.Series,
    threshold: float,
) -> pd.Series:
    return (
        travel_times.set_index("to_id")
        .groupby("from_id")["travel_time"]
        .min()
        .apply(lambda x: 1 if x is not None and x <= 15 else 0)
    )


def tsfca(
    travel_times: pd.DataFrame,
    population: pd.Series,
    amenity_weights: pd.Series,
    threshold: float,
    desired_ratio: float,
) -> pd.Series:

    def calculate_need_to_population_ratio(travel_times: pd.Series):
        cells_in_catchment = travel_times[(travel_times <= threshold)].index
        population_in_catchment = population.loc[cells_in_catchment].sum()
        return 1 / population_in_catchment

    ratios_2sfca = (
        travel_times.set_index("from_id")
        .groupby("to_id")
        .agg(calculate_need_to_population_ratio)
        .squeeze()
        .rename("ratio")
    )

    def calculate_2sfca(travel_times: pd.Series):
        dests_in_catchment = travel_times[(travel_times <= threshold)].index
        ratios_in_catchment = (
            amenity_weights.loc[dests_in_catchment]
            * ratios_2sfca.loc[dests_in_catchment]
        )
        return ratios_in_catchment.sum()

    return (
        travel_times.set_index("to_id")
        .groupby("from_id")
        .agg(calculate_2sfca)
        .squeeze()
        .rename("accessibility")
        .clip(upper=1 / desired_ratio)
        * desired_ratio
    )
