"""Retrieve weather observations for flight departure and arrival airports.

This script uses the MET Norway Frost API to find nearby weather stations for
each flight airport and retrieve temperature, wind speed, and wind direction
observations around departure and arrival times. Results are written to a raw
weather observation table in SQLite for later feature creation.
"""

import math
import sqlite3
import airportsdata
import pandas as pd
import requests

FROST_CLIENT_ID = "your_frost_client_id_here"  # Set this to your Frost API client ID before running

BASE_URL = "https://frost.met.no"

DB_PATH = "opensky.sqlite"
IN_TABLE = "flight_phase_features_v2"
OUT_RAW_WEATHER_TABLE = "flight_weather_observations_v2"

REQUESTED_ELEMENTS = [
    "air_temperature",
    "wind_speed",
    "wind_from_direction",
]

MAX_CANDIDATES = 5


if not FROST_CLIENT_ID:
    raise ValueError("Set FROST_CLIENT_ID first.")

airports = airportsdata.load("ICAO")

# In-memory caches to reduce repeated Frost API requests
candidate_station_cache = {}
available_elements_cache = {}
observation_cache = {}
best_station_cache = {}

EMPTY_OBS_DF = pd.DataFrame(
    columns=["referenceTime", "elementId", "value", "unit", "qualityCode"]
)


def get_airport_coords(icao):
    """Return latitude and longitude for an ICAO airport code."""
    airport = airports.get(icao)
    if airport is None:
        return None, None
    return airport["lat"], airport["lon"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two coordinates in kilometers."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def frost_get(endpoint, params=None):
    """Send a GET request to the Frost API and return the JSON response."""
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if not response.ok:
        raise requests.HTTPError(
            f"Frost error {response.status_code} for {response.url}",
            response=response,
        )

    return response.json()


def find_candidate_stations(lat, lon, max_candidates=MAX_CANDIDATES):
    """Find nearby Frost weather stations for a coordinate pair."""
    params = {
        "types": "SensorSystem",
        "geometry": f"nearest(POINT({lon} {lat}))",
        "nearestmaxcount": max_candidates,
    }
    data = frost_get("/sources/v0.jsonld", params=params)

    candidates = []
    for item in data.get("data", []):
        coords = item.get("geometry", {}).get("coordinates")
        if not coords or len(coords) < 2:
            continue

        station_lon, station_lat = coords[0], coords[1]
        candidates.append(
            {
                "id": item["id"],
                "name": item.get("name"),
                "lat": station_lat,
                "lon": station_lon,
                "distance_km": haversine_km(lat, lon, station_lat, station_lon),
            }
        )

    candidates.sort(key=lambda x: x["distance_km"])
    return candidates


def get_cached_candidate_stations(icao, lat, lon, max_candidates=MAX_CANDIDATES):
    """Return cached nearby weather stations for an airport."""
    cache_key = (icao, max_candidates)
    if cache_key not in candidate_station_cache:
        candidate_station_cache[cache_key] = find_candidate_stations(
            lat=lat,
            lon=lon,
            max_candidates=max_candidates,
        )
    return candidate_station_cache[cache_key]


def get_available_elements(source_id, start, end):
    """Return weather elements available from a station in a time interval."""
    params = {
        "sources": source_id,
        "referencetime": f"{start.isoformat()}/{end.isoformat()}",
        "levels": "default",
        "timeoffsets": "default",
    }

    url = f"{BASE_URL}/observations/availableTimeSeries/v0.jsonld"
    response = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if response.status_code == 404:
        return set()

    if not response.ok:
        raise requests.HTTPError(
            f"Frost error {response.status_code} for {response.url}",
            response=response,
        )

    data = response.json()
    return {
        item.get("elementId")
        for item in data.get("data", [])
        if item.get("elementId")
    }


def get_cached_available_elements(source_id, start, end):
    """Return cached available weather elements for a station and month."""
    start = pd.Timestamp(start)
    cache_key = (source_id, start.strftime("%Y-%m"))

    if cache_key not in available_elements_cache:
        available_elements_cache[cache_key] = get_available_elements(source_id, start, end)

    return available_elements_cache[cache_key]


def find_best_station_for_elements(
    icao,
    lat,
    lon,
    start,
    end,
    requested_elements,
    max_candidates=MAX_CANDIDATES,
):
    """Choose the nearest station that provides the requested weather elements."""
    month_key = pd.Timestamp(start).strftime("%Y-%m")
    cache_key = (icao, month_key, tuple(requested_elements), max_candidates)

    if cache_key in best_station_cache:
        return best_station_cache[cache_key]

    candidates = get_cached_candidate_stations(
        icao=icao,
        lat=lat,
        lon=lon,
        max_candidates=max_candidates,
    )

    if not candidates:
        best_station_cache[cache_key] = (None, [])
        return None, []

    scored = []
    for station in candidates:
        available = get_cached_available_elements(station["id"], start, end)
        matched = [element for element in requested_elements if element in available]
        scored.append(
            {
                **station,
                "matched_elements": matched,
                "match_count": len(matched),
            }
        )

    full = [station for station in scored if station["match_count"] == len(requested_elements)]
    if full:
        full.sort(key=lambda x: x["distance_km"])
        best = full[0]
        result = (best, best["matched_elements"])
        best_station_cache[cache_key] = result
        return result

    scored.sort(key=lambda x: (-x["match_count"], x["distance_km"]))
    best = scored[0]

    if best["match_count"] == 0:
        best_station_cache[cache_key] = (None, [])
        return None, []

    result = (best, best["matched_elements"])
    best_station_cache[cache_key] = result
    return result


def fetch_observations(source_id, start, end, elements):
    """Fetch weather observations for selected elements from a Frost station."""
    if not elements:
        return EMPTY_OBS_DF.copy()

    params = {
        "sources": source_id,
        "referencetime": f"{start.isoformat()}/{end.isoformat()}",
        "elements": ",".join(elements),
        "levels": "default",
        "timeoffsets": "default",
    }

    url = f"{BASE_URL}/observations/v0.jsonld"
    response = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if response.status_code in (404, 412):
        return EMPTY_OBS_DF.copy()

    if not response.ok:
        raise requests.HTTPError(
            f"Frost error {response.status_code} for {response.url}",
            response=response,
        )

    data = response.json()

    rows = []
    for item in data.get("data", []):
        ref_time = item.get("referenceTime")
        for obs in item.get("observations", []):
            rows.append(
                {
                    "referenceTime": ref_time,
                    "elementId": obs.get("elementId"),
                    "value": obs.get("value"),
                    "unit": obs.get("unit"),
                    "qualityCode": obs.get("qualityCode"),
                }
            )

    if not rows:
        return EMPTY_OBS_DF.copy()

    df = pd.DataFrame(rows)
    df["referenceTime"] = pd.to_datetime(df["referenceTime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def fetch_observations_cached(source_id, start, end, elements):
    """Fetch observations using an hourly cache to reduce repeated API calls."""
    if not elements:
        return EMPTY_OBS_DF.copy()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    # Cache observations on hourly windows to maximize API reuse between flights
    rounded_start = start.floor("1h")
    rounded_end = end.ceil("1h")

    cache_key = (
        source_id,
        rounded_start.isoformat(),
        rounded_end.isoformat(),
        tuple(sorted(elements)),
    )

    if cache_key not in observation_cache:
        observation_cache[cache_key] = fetch_observations(
            source_id=source_id,
            start=rounded_start,
            end=rounded_end,
            elements=elements,
        )

    df = observation_cache[cache_key]
    if df.empty:
        return df.copy()

    mask = (df["referenceTime"] >= start) & (df["referenceTime"] <= end)
    return df.loc[mask].copy()


def empty_weather_row(row, prefix, error):
    """Create an empty weather result row with an error message."""
    return {
        "flight_id": row["flight_id"],
        "airport_phase": prefix,
        "airport_icao": row[f"{prefix}_airport"],
        "station_id": None,
        "station_name": None,
        "station_distance_km": None,
        "window_start": None,
        "window_end": None,
        "matched_elements": None,
        "referenceTime": None,
        "elementId": None,
        "value": None,
        "unit": None,
        "qualityCode": None,
        "weather_error": error,
    }


def observation_rows_for_airport(row, prefix, airport_icao, lat, lon, event_time):
    """Retrieve weather observation rows for one airport event."""
    event_time = pd.to_datetime(event_time, unit="s", utc=True, errors="coerce")
    if pd.isna(event_time):
        return [empty_weather_row(row, prefix, "Missing event time")]

    if prefix == "dep":
        window_start = event_time - pd.Timedelta(minutes=60)
        window_end = event_time + pd.Timedelta(minutes=60)
    else:
        window_start = event_time - pd.Timedelta(minutes=60)
        window_end = event_time

    station, elements = find_best_station_for_elements(
        airport_icao,
        lat,
        lon,
        window_start,
        window_end,
        REQUESTED_ELEMENTS,
    )

    base = {
        "flight_id": row["flight_id"],
        "airport_phase": prefix,
        "airport_icao": airport_icao,
        "station_id": station["id"] if station else None,
        "station_name": station["name"] if station else None,
        "station_distance_km": station["distance_km"] if station else None,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "matched_elements": ",".join(elements) if elements else None,
    }

    if not station or not elements:
        return [
            {
                **base,
                "referenceTime": None,
                "elementId": None,
                "value": None,
                "unit": None,
                "qualityCode": None,
                "weather_error": "No station with requested elements",
            }
        ]

    obs_df = fetch_observations_cached(station["id"], window_start, window_end, elements)
    if obs_df.empty:
        return [
            {
                **base,
                "referenceTime": None,
                "elementId": None,
                "value": None,
                "unit": None,
                "qualityCode": None,
                "weather_error": "No observations in time window",
            }
        ]

    rows = []
    for obs in obs_df.itertuples(index=False):
        rows.append(
            {
                **base,
                "referenceTime": obs.referenceTime.isoformat()
                if pd.notna(obs.referenceTime)
                else None,
                "elementId": obs.elementId,
                "value": obs.value,
                "unit": obs.unit,
                "qualityCode": obs.qualityCode,
                "weather_error": None,
            }
        )

    return rows


def retrieve_weather_for_flight(row):
    """Retrieve departure and arrival weather observations for one flight."""
    dep_lat, dep_lon = get_airport_coords(row["dep_airport"])
    arr_lat, arr_lon = get_airport_coords(row["arr_airport"])

    if dep_lat is None or dep_lon is None or arr_lat is None or arr_lon is None:
        return [
            empty_weather_row(row, "dep", "Missing airport coordinates"),
            empty_weather_row(row, "arr", "Missing airport coordinates"),
        ]

    rows = []
    rows.extend(
        observation_rows_for_airport(
            row=row,
            prefix="dep",
            airport_icao=row["dep_airport"],
            lat=dep_lat,
            lon=dep_lon,
            event_time=row["dep_time"],
        )
    )
    rows.extend(
        observation_rows_for_airport(
            row=row,
            prefix="arr",
            airport_icao=row["arr_airport"],
            lat=arr_lat,
            lon=arr_lon,
            event_time=row["arr_time"],
        )
    )
    return rows


def chunk_list(values, batch_size):
    """Yield batches from a list."""
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def process_batch(batch_ids):
    """Retrieve weather observations for a batch of flight IDs."""
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join(["?"] * len(batch_ids))
    query = f"""
    SELECT
        flight_id,
        estdepartureairport AS dep_airport,
        estarrivalairport AS arr_airport,
        dep_time,
        arr_time
    FROM "{IN_TABLE}"
    WHERE flight_id IN ({placeholders})
    """
    flights = pd.read_sql_query(query, conn, params=batch_ids)
    conn.close()

    rows = []
    for _, row in flights.iterrows():
        try:
            rows.extend(retrieve_weather_for_flight(row))
        except Exception as exc:
            rows.append(empty_weather_row(row, "dep", str(exc)))
            rows.append(empty_weather_row(row, "arr", str(exc)))

    return pd.DataFrame(rows)


def get_flight_ids_to_process(rebuild_raw_table=False):
    """Return flight IDs that still need weather observations."""
    conn = sqlite3.connect(DB_PATH)

    if rebuild_raw_table:
        conn.execute(f'DROP TABLE IF EXISTS "{OUT_RAW_WEATHER_TABLE}"')
        conn.commit()
        print(f"Dropped old table: {OUT_RAW_WEATHER_TABLE}")

    raw_table_exists = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        conn,
        params=(OUT_RAW_WEATHER_TABLE,),
    )

    if raw_table_exists.empty:
        query = f"""
        SELECT flight_id
        FROM "{IN_TABLE}"
        WHERE estdepartureairport IS NOT NULL
          AND estarrivalairport IS NOT NULL
          AND dep_time IS NOT NULL
          AND arr_time IS NOT NULL
        """
    else:
        query = f"""
        SELECT f.flight_id
        FROM "{IN_TABLE}" f
        LEFT JOIN (
            SELECT DISTINCT flight_id
            FROM "{OUT_RAW_WEATHER_TABLE}"
        ) w
          ON f.flight_id = w.flight_id
        WHERE f.estdepartureairport IS NOT NULL
          AND f.estarrivalairport IS NOT NULL
          AND f.dep_time IS NOT NULL
          AND f.arr_time IS NOT NULL
          AND w.flight_id IS NULL
        """

    flight_ids = pd.read_sql_query(query, conn)["flight_id"].tolist()
    conn.close()
    return flight_ids, raw_table_exists.empty or rebuild_raw_table


def main(batch_size=20, rebuild_raw_table=False):
    """Retrieve weather observations in batches and write them to SQLite."""
    flight_ids, first_write = get_flight_ids_to_process(rebuild_raw_table)
    print("Flights to process:", len(flight_ids))

    if not flight_ids:
        print("No new flights to process.")
        return

    total_batches = math.ceil(len(flight_ids) / batch_size)

    for batch_num, batch_ids in enumerate(chunk_list(flight_ids, batch_size), start=1):
        print(f"\nProcessing batch {batch_num}/{total_batches} with {len(batch_ids)} flights...")

        weather_df = process_batch(batch_ids)

        conn = sqlite3.connect(DB_PATH)
        weather_df.to_sql(
            OUT_RAW_WEATHER_TABLE,
            conn,
            if_exists="replace" if first_write else "append",
            index=False,
        )
        conn.close()

        first_write = False
        print(f"Batch {batch_num} done. Rows written: {len(weather_df)}")

        if "weather_error" in weather_df.columns:
            print("Rows with weather_error:", weather_df["weather_error"].notna().sum())

    conn_check = sqlite3.connect(DB_PATH)
    print("\nRow count in raw weather table:")
    print(pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{OUT_RAW_WEATHER_TABLE}"', conn_check))
    print("\nDistinct flights in raw weather table:")
    print(
        pd.read_sql_query(
            f'SELECT COUNT(DISTINCT flight_id) AS n FROM "{OUT_RAW_WEATHER_TABLE}"',
            conn_check,
        )
    )
    conn_check.close()


if __name__ == "__main__":
    main(batch_size=20, rebuild_raw_table=False)
