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


def clean_gtfs_frequencies(
    inpath: str | Path, outpath: str | Path, chunk_size: int = 50_000
):
    """
    Limpia un archivo GTFS, eliminando `frequencies.txt` y modificando
    `stop_times.txt` y `trips.txt`, dejando un GTFS con igual comportamiento.
    Esto hace el archivo más pesado, pero mucho más eficiente a la hora de
    computar matrices de tiempo de viaje mediante r5py.

    También se elimina `shapes.txt` al no utilizarse, y se formatea `stops.txt`
    para que todos los paraderos tengan nombre.

    Código de Danny Whalen, extraído de:
    https://gist.github.com/invisiblefunnel/6c9f3a9b537d3f0ad192c24777b6ae57

    Parameters
    ---
    inpath : str or Path
        Ruta de archivo GTFS a modificar.
    outpath : str or Path
        Ruta de archivo GTFS modificado.
    chunk_size : int, default: 50_000
        Número de viajes a agrupar por chunk para el cálculo de los nuevos
        archivos `stop_times.txt` y `trips.txt`. Cada chunk se convierte en un
        DataFrame y luego estos se concatenan para generar los archivos
        finales; si la función falla por falta de memoria, se recomienda
        disminuir `chunk_size`.
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
    total = len(freq_trips)
    trips_chunks: list[pd.DataFrame] = []
    stop_times_chunks: list[pd.DataFrame] = []

    for chunk_start in range(0, total, chunk_size):
        chunk = freq_trips[chunk_start : min(chunk_start + chunk_size, total)]

        trips_rows = []
        stop_times_rows = []

        for i, ftrip in enumerate(chunk):
            new_trip = copy(trips_by_id[ftrip["trip_id"]])
            new_trip["trip_id"] = chunk_start + i
            trips_rows.append(new_trip)

            stops, times = trip_patterns[ftrip["trip_id"]]
            base = ftrip["start"]
            for j, (stop, t) in enumerate(zip(stops, times), start=1):
                gt = seconds_to_gtfs_time(t + base)
                stop_times_rows.append(
                    {
                        "trip_id": chunk_start + i,
                        "stop_id": stop,
                        "arrival_time": gt,
                        "departure_time": gt,
                        "stop_sequence": j,
                    }
                )

        trips_chunks.append(pd.DataFrame(trips_rows))
        stop_times_chunks.append(pd.DataFrame(stop_times_rows))

    # get trips and stop times that aren't in frequencies.txt
    raw_feed = ptg.load_raw_feed(inpath)
    freq_trip_ids = set(raw_feed.frequencies["trip_id"])
    all_trip_ids = set(raw_feed.trips["trip_id"])
    diff_trip_ids = all_trip_ids.difference(freq_trip_ids)
    missing_trips_df = raw_feed.trips[raw_feed.trips["trip_id"].isin(diff_trip_ids)]
    missing_stop_times_df = raw_feed.stop_times[raw_feed.stop_times["trip_id"].isin(diff_trip_ids)]

    # assign names to stops that lack them
    stops_df = raw_feed.stops
    stops_df["stop_name"] = stops_df["stop_name"].fillna(stops_df["stop_id"])
    
    trips_df = pd.concat(trips_chunks + [missing_trips_df], ignore_index=True)
    stop_times_df = pd.concat(stop_times_chunks + [missing_stop_times_df], ignore_index=True)
    del trips_chunks, stop_times_chunks, freq_trips

    print("(4/4) Escribiendo archivos...")

    raw_feed.set("trips.txt", trips_df)
    raw_feed.set("stop_times.txt", stop_times_df)
    raw_feed.set("stops.txt", stops_df)

    # remove unneeded files
    raw_feed.set("frequencies.txt", ptg.utilities.empty_df())
    raw_feed.set("shapes.txt", ptg.utilities.empty_df())

    ptg.writers.write_feed_dangerously(raw_feed, str(outpath))


def clean_gtfs_basic(inpath: str | Path, outpath: str | Path):
    """
    Limpia un archivo GTFS, eliminando `shapes.txt` al no utilizarse, y
    formateando `stops.txt` para que todos los paraderos tengan nombre.

    Parameters
    ---
    inpath : str or Path
        Ruta de archivo GTFS a modificar.
    outpath : str or Path
        Ruta de archivo GTFS modificado.
    """

    new_feed = ptg.load_raw_feed(inpath)

    # agregar nombre a las stops que no lo tienen (zonas intermedias)
    stops_df = new_feed.stops
    stops_df["stop_name"] = stops_df["stop_name"].fillna(stops_df["stop_id"])

    new_feed.set("stops.txt", stops_df)
    new_feed.set("shapes.txt", ptg.utilities.empty_df())
    ptg.writers.write_feed_dangerously(new_feed, str(outpath))
