# helper functions to clean GTFS files
from copy import copy
from pathlib import Path

import pandas as pd
import partridge as ptg


def seconds_to_gtfs_time(total_seconds: float) -> str:
    """
    Convierte un tiempo en segundos a un formato utilizable por GTFS, del
    estilo HH:MM:SS.

    Código de Danny Whalen, extraído de:
    https://gist.github.com/invisiblefunnel/6c9f3a9b537d3f0ad192c24777b6ae57

    Parameters
    ---
    total_seconds : float
        Número a convertir.

    Returns
    ---
    String con el tiempo formateado.
    """

    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    time = list(
        map(
            lambda x: str(x).rjust(2, "0"),
            [int(hours), int(minutes), int(seconds)],
        )
    )
    return f"{time[0]}:{time[1]}:{time[2]}"


def clean_gtfs(inpath: str | Path, outpath: str | Path):
    """
    Limpia un archivo GTFS, eliminando `frequencies.txt` y modificando
    `stop_times.txt` y `trips.txt`, dejando un GTFS con igual comportamiento.
    Esto hace el archivo más pesado, pero mucho más eficiente a la hora de
    computar matrices de tiempo de viaje mediante r5py. También se elimina
    `shapes.txt` al no utilizarse.

    Código de Danny Whalen, extraído de:
    https://gist.github.com/invisiblefunnel/6c9f3a9b537d3f0ad192c24777b6ae57

    Parameters
    ---
    inpath : str or Path
        Ruta de archivo GTFS a modificar.
    outpath : str or Path
        Ruta de archivo GTFS modificado.
    """

    feed = ptg.load_feed(inpath)

    trips_by_id = {}
    for _, trip in feed.trips.iterrows():
        trips_by_id[trip.trip_id] = dict(trip)

    # get stop sequence for each trip
    print("(1/4) Obteniendo ruta de cada viaje...")
    trip_patterns = {}
    for trip_id, stop_times in feed.stop_times.sort_values(
        "stop_sequence"
    ).groupby("trip_id"):
        stops_df = tuple(stop_times.stop_id)
        mintime = stop_times.arrival_time.min()
        times = tuple(t - mintime for t in stop_times.arrival_time)
        trip_patterns[trip_id] = (stops_df, times)

    # "duplicate" trips according to their frequency
    print("(2/4) Duplicando viajes según frecuencia...")
    freq_trips = []
    for _, freq in feed.frequencies.iterrows():
        window_start = int(freq.start_time)
        window_end = int(freq.end_time)
        for start in range(window_start, window_end, freq.headway_secs):
            freq_trips.append(
                {
                    "trip_id": freq.trip_id,
                    "start": start,
                }
            )

    # assign new trips (each with their unique id) and corresponding stops
    print("(3/4) Agregando viajes nuevos a GTFS...")
    new_trips = []
    new_stop_times = []
    for i, ftrip in enumerate(freq_trips, start=1):
        new_trips.append(copy(trips_by_id[ftrip["trip_id"]]))
        new_trips[-1]["trip_id"] = i  # override trip_id

        stops, times = trip_patterns[ftrip["trip_id"]]
        for j in range(len(stops)):
            t = seconds_to_gtfs_time(times[j] + ftrip["start"])
            new_stop_times.append(
                {
                    "trip_id": i,
                    "stop_id": stops[j],
                    "arrival_time": t,
                    "departure_time": t,
                    "stop_sequence": j + 1,
                }
            )

    print("(4/4) Escribiendo archivos...")
    trips_df = pd.DataFrame(new_trips)
    stop_times_df = pd.DataFrame(new_stop_times)

    new_feed = ptg.load_raw_feed(inpath)
    new_feed.set("trips.txt", trips_df)
    new_feed.set("stop_times.txt", stop_times_df)

    # remove unneeded files
    new_feed.set("frequencies.txt", ptg.utilities.empty_df())
    new_feed.set("shapes.txt", ptg.utilities.empty_df())

    ptg.writers.write_feed_dangerously(new_feed, str(outpath))
