import math
import sqlite3
import requests
import pandas as pd
import airportsdata
import numpy as np

FROST_CLIENT_ID = "a17ba72e-e1a1-4f5a-85d5-50403d7814fe"

BASE_URL = "https://frost.met.no"

DB_PATH = "../opensky.sqlite"
FLIGHT_TABLE = "flight_phase_features_v2"
OUT_WEATHER_TABLE = "flight_weather_features_v2"
MERGED_TABLE = "flight_phase_features_weather_v2"

REQUESTED_ELEMENTS = [
    "air_temperature",
    "wind_speed",
    "wind_from_direction",
]

MAX_CANDIDATES = 5

if not FROST_CLIENT_ID:
    raise ValueError("Sett FROST_CLIENT_ID først.")

airports = airportsdata.load("ICAO")

# -----------------------------
# Global caches
# -----------------------------
candidate_station_cache = {}
available_elements_cache = {}
observation_cache = {}
best_station_cache = {}


# -----------------------------
# Basic helpers
# -----------------------------
def get_airport_coords(icao):
    a = airports.get(icao)
    if a is None:
        return None, None
    return a["lat"], a["lon"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def frost_get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    r = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if not r.ok:
        raise requests.HTTPError(f"Frost error {r.status_code} for {r.url}", response=r)

    return r.json()


# -----------------------------
# Station lookup
# -----------------------------
def find_candidate_stations(lat, lon, max_candidates=MAX_CANDIDATES):
    params = {
        "types": "SensorSystem",
        "geometry": f"nearest(POINT({lon} {lat}))",
        "nearestmaxcount": max_candidates,
    }

    data = frost_get("/sources/v0.jsonld", params=params)

    candidates = []
    for item in data.get("data", []):
        geom = item.get("geometry", {})
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        station_lon, station_lat = coords[0], coords[1]
        dist = haversine_km(lat, lon, station_lat, station_lon)

        candidates.append({
            "id": item["id"],
            "name": item.get("name"),
            "lat": station_lat,
            "lon": station_lon,
            "distance_km": dist
        })

    candidates.sort(key=lambda x: x["distance_km"])
    return candidates


def get_cached_candidate_stations(icao, lat, lon, max_candidates=MAX_CANDIDATES):
    cache_key = (icao, max_candidates)
    if cache_key not in candidate_station_cache:
        candidate_station_cache[cache_key] = find_candidate_stations(
            lat=lat,
            lon=lon,
            max_candidates=max_candidates
        )
    return candidate_station_cache[cache_key]


# -----------------------------
# Available elements
# -----------------------------
def get_available_elements(source_id, start, end):
    params = {
        "sources": source_id,
        "referencetime": f"{start.isoformat()}/{end.isoformat()}",
        "levels": "default",
        "timeoffsets": "default",
    }

    url = f"{BASE_URL}/observations/availableTimeSeries/v0.jsonld"
    r = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if r.status_code == 404:
        return set()

    if not r.ok:
        raise requests.HTTPError(f"Frost error {r.status_code} for {r.url}", response=r)

    data = r.json()

    elements = set()
    for item in data.get("data", []):
        element_id = item.get("elementId")
        if element_id:
            elements.add(element_id)

    return elements


def get_cached_available_elements(source_id, start, end):
    start = pd.Timestamp(start)
    cache_key = (source_id, start.strftime("%Y-%m"))

    if cache_key not in available_elements_cache:
        available_elements_cache[cache_key] = get_available_elements(source_id, start, end)

    return available_elements_cache[cache_key]


# -----------------------------
# Best station choice
# -----------------------------
def find_best_station_for_elements(icao, lat, lon, start, end, requested_elements, max_candidates=MAX_CANDIDATES):
    month_key = pd.Timestamp(start).strftime("%Y-%m")
    cache_key = (icao, month_key, tuple(requested_elements), max_candidates)

    if cache_key in best_station_cache:
        return best_station_cache[cache_key]

    candidates = get_cached_candidate_stations(
        icao=icao,
        lat=lat,
        lon=lon,
        max_candidates=max_candidates
    )

    if not candidates:
        best_station_cache[cache_key] = (None, [])
        return None, []

    scored = []

    for station in candidates:
        available = get_cached_available_elements(station["id"], start, end)
        matched = [e for e in requested_elements if e in available]

        scored.append({
            **station,
            "matched_elements": matched,
            "match_count": len(matched),
        })

    full = [s for s in scored if s["match_count"] == len(requested_elements)]
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


# -----------------------------
# Observations
# -----------------------------
EMPTY_OBS_DF = pd.DataFrame(columns=["referenceTime", "elementId", "value", "unit", "qualityCode"])


def fetch_observations(source_id, start, end, elements):
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
    r = requests.get(url, params=params, auth=(FROST_CLIENT_ID, ""))

    if r.status_code in (404, 412):
        return EMPTY_OBS_DF.copy()

    if not r.ok:
        raise requests.HTTPError(f"Frost error {r.status_code} for {r.url}", response=r)

    data = r.json()

    rows = []
    for item in data.get("data", []):
        ref_time = item.get("referenceTime")
        for obs in item.get("observations", []):
            rows.append({
                "referenceTime": ref_time,
                "elementId": obs.get("elementId"),
                "value": obs.get("value"),
                "unit": obs.get("unit"),
                "qualityCode": obs.get("qualityCode")
            })

    if not rows:
        return EMPTY_OBS_DF.copy()

    df = pd.DataFrame(rows)
    df["referenceTime"] = pd.to_datetime(df["referenceTime"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def fetch_observations_cached(source_id, start, end, elements):
    if not elements:
        return EMPTY_OBS_DF.copy()

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    rounded_start = start.floor("1h")
    rounded_end = end.ceil("1h")

    cache_key = (
        source_id,
        rounded_start.isoformat(),
        rounded_end.isoformat(),
        tuple(sorted(elements))
    )

    if cache_key not in observation_cache:
        observation_cache[cache_key] = fetch_observations(
            source_id=source_id,
            start=rounded_start,
            end=rounded_end,
            elements=elements
        )

    df = observation_cache[cache_key]
    if df.empty:
        return df.copy()

    mask = (df["referenceTime"] >= start) & (df["referenceTime"] <= end)
    return df.loc[mask].copy()


# -----------------------------
# Feature aggregation
# -----------------------------
def aggregate_weather_features(df, prefix):
    out = {
        f"{prefix}_weather_missing": 1,
        f"{prefix}_temp_missing": 1,
        f"{prefix}_wind_missing": 1,
        f"{prefix}_wind_dir_missing": 1,
        f"{prefix}_temp_mean": None,
        f"{prefix}_temp_min": None,
        f"{prefix}_temp_max": None,
        f"{prefix}_temp_std": None,
        f"{prefix}_wind_speed_mean": None,
        f"{prefix}_wind_speed_min": None,
        f"{prefix}_wind_speed_max": None,
        f"{prefix}_wind_speed_std": None,
        f"{prefix}_wind_dir_last": None,
    }

    if df.empty:
        return out

    out[f"{prefix}_weather_missing"] = 0

    temp = df[df["elementId"] == "air_temperature"]["value"].dropna()
    wind = df[df["elementId"] == "wind_speed"]["value"].dropna()
    wind_dir = df[df["elementId"] == "wind_from_direction"]["value"].dropna()

    out[f"{prefix}_temp_missing"] = 0 if len(temp) > 0 else 1
    out[f"{prefix}_wind_missing"] = 0 if len(wind) > 0 else 1
    out[f"{prefix}_wind_dir_missing"] = 0 if len(wind_dir) > 0 else 1

    if len(temp) > 0:
        out[f"{prefix}_temp_mean"] = float(temp.mean())
        out[f"{prefix}_temp_min"] = float(temp.min())
        out[f"{prefix}_temp_max"] = float(temp.max())
        out[f"{prefix}_temp_std"] = float(temp.std()) if len(temp) > 1 else 0.0

    if len(wind) > 0:
        out[f"{prefix}_wind_speed_mean"] = float(wind.mean())
        out[f"{prefix}_wind_speed_min"] = float(wind.min())
        out[f"{prefix}_wind_speed_max"] = float(wind.max())
        out[f"{prefix}_wind_speed_std"] = float(wind.std()) if len(wind) > 1 else 0.0

    if len(wind_dir) > 0:
        out[f"{prefix}_wind_dir_last"] = float(wind_dir.iloc[-1])

    return out


# -----------------------------
# Flight-level weather features
# -----------------------------
def get_weather_features_for_flight(dep_icao, dep_lat, dep_lon, dep_time,
                                    arr_icao, arr_lat, arr_lon, arr_time):
    dep_time = pd.to_datetime(dep_time, unit="s", utc=True, errors="coerce")
    arr_time = pd.to_datetime(arr_time, unit="s", utc=True, errors="coerce")

    result = {}

    # Departure
    if pd.notna(dep_time):
        dep_start = dep_time - pd.Timedelta(minutes=60)
        dep_end = dep_time + pd.Timedelta(minutes=60)

        dep_station, dep_elements = find_best_station_for_elements(
            dep_icao, dep_lat, dep_lon, dep_start, dep_end, REQUESTED_ELEMENTS
        )

        result["dep_station_id"] = dep_station["id"] if dep_station else None
        result["dep_station_distance_km"] = dep_station["distance_km"] if dep_station else None

        if dep_station and dep_elements:
            dep_df = fetch_observations_cached(dep_station["id"], dep_start, dep_end, dep_elements)
            result.update(aggregate_weather_features(dep_df, "dep"))
        else:
            result.update(aggregate_weather_features(EMPTY_OBS_DF, "dep"))
    else:
        result.update(aggregate_weather_features(EMPTY_OBS_DF, "dep"))
        result["dep_station_id"] = None
        result["dep_station_distance_km"] = None

    # Arrival
    if pd.notna(arr_time):
        arr_start = arr_time - pd.Timedelta(minutes=60)
        arr_end = arr_time

        arr_station, arr_elements = find_best_station_for_elements(
            arr_icao, arr_lat, arr_lon, arr_start, arr_end, REQUESTED_ELEMENTS
        )

        result["arr_station_id"] = arr_station["id"] if arr_station else None
        result["arr_station_distance_km"] = arr_station["distance_km"] if arr_station else None

        if arr_station and arr_elements:
            arr_df = fetch_observations_cached(arr_station["id"], arr_start, arr_end, arr_elements)
            result.update(aggregate_weather_features(arr_df, "arr"))
        else:
            result.update(aggregate_weather_features(EMPTY_OBS_DF, "arr"))
    else:
        result.update(aggregate_weather_features(EMPTY_OBS_DF, "arr"))
        result["arr_station_id"] = None
        result["arr_station_distance_km"] = None

    return result


# -----------------------------
# Batch processing
# -----------------------------
def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def process_batch(batch_ids):
    conn = sqlite3.connect(DB_PATH)

    placeholders = ",".join(["?"] * len(batch_ids))
    query = f"""
    SELECT
        flight_id,
        estdepartureairport,
        estarrivalairport,
        dep_time,
        arr_time
    FROM "{FLIGHT_TABLE}"
    WHERE flight_id IN ({placeholders})
    """

    flights = pd.read_sql_query(query, conn, params=batch_ids)
    conn.close()

    weather_rows = []

    for _, row in flights.iterrows():
        try:
            dep_lat, dep_lon = get_airport_coords(row["estdepartureairport"])
            arr_lat, arr_lon = get_airport_coords(row["estarrivalairport"])

            if dep_lat is None or dep_lon is None or arr_lat is None or arr_lon is None:
                weather_rows.append({
                    "flight_id": row["flight_id"],
                    "dep_station_id": None,
                    "dep_station_distance_km": None,
                    "arr_station_id": None,
                    "arr_station_distance_km": None,
                    "dep_weather_missing": 1,
                    "dep_temp_missing": 1,
                    "dep_wind_missing": 1,
                    "dep_wind_dir_missing": 1,
                    "arr_weather_missing": 1,
                    "arr_temp_missing": 1,
                    "arr_wind_missing": 1,
                    "arr_wind_dir_missing": 1,
                    "weather_error": "Missing airport coordinates"
                })
                continue

            features = get_weather_features_for_flight(
                dep_icao=row["estdepartureairport"],
                dep_lat=dep_lat,
                dep_lon=dep_lon,
                dep_time=row["dep_time"],
                arr_icao=row["estarrivalairport"],
                arr_lat=arr_lat,
                arr_lon=arr_lon,
                arr_time=row["arr_time"]
            )

            features["flight_id"] = row["flight_id"]
            features["weather_error"] = None
            weather_rows.append(features)

        except Exception as e:
            weather_rows.append({
                "flight_id": row["flight_id"],
                "dep_station_id": None,
                "dep_station_distance_km": None,
                "arr_station_id": None,
                "arr_station_distance_km": None,
                "dep_weather_missing": 1,
                "dep_temp_missing": 1,
                "dep_wind_missing": 1,
                "dep_wind_dir_missing": 1,
                "arr_weather_missing": 1,
                "arr_temp_missing": 1,
                "arr_wind_missing": 1,
                "arr_wind_dir_missing": 1,
                "weather_error": str(e)
            })

    return pd.DataFrame(weather_rows)


# -----------------------------
# Final merge + engineered features
# -----------------------------
def build_merged_table():
    conn = sqlite3.connect(DB_PATH)

    features_df = pd.read_sql_query(f'SELECT * FROM "{FLIGHT_TABLE}"', conn)
    weather_df = pd.read_sql_query(f'SELECT * FROM "{OUT_WEATHER_TABLE}"', conn)

    merged_df = features_df.merge(weather_df, on="flight_id", how="left")

    # Wind direction sin/cos
    for prefix in ["dep", "arr"]:
        dir_col = f"{prefix}_wind_dir_last"
        if dir_col in merged_df.columns:
            radians = np.radians(merged_df[dir_col])
            merged_df[f"{prefix}_wind_dir_sin"] = np.sin(radians)
            merged_df[f"{prefix}_wind_dir_cos"] = np.cos(radians)
        else:
            merged_df[f"{prefix}_wind_dir_sin"] = np.nan
            merged_df[f"{prefix}_wind_dir_cos"] = np.nan


    merged_df.to_sql(MERGED_TABLE, conn, if_exists="replace", index=False)

    print("\nMerged shape:", merged_df.shape)
    print("\nWind direction columns:")
    print([c for c in merged_df.columns if "wind_dir" in c])

    conn.close()


# -----------------------------
# Main
# -----------------------------
def main(batch_size=500, rebuild_weather_table=False):
    conn = sqlite3.connect(DB_PATH)

    if rebuild_weather_table:
        conn.execute(f'DROP TABLE IF EXISTS "{OUT_WEATHER_TABLE}"')
        conn.commit()
        print(f'Dropped old table: {OUT_WEATHER_TABLE}')

    weather_table_exists = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        conn,
        params=(OUT_WEATHER_TABLE,)
    )

    if weather_table_exists.empty:
        query = f"""
        SELECT flight_id
        FROM "{FLIGHT_TABLE}"
        WHERE estdepartureairport IS NOT NULL
          AND estarrivalairport IS NOT NULL
          AND dep_time IS NOT NULL
          AND arr_time IS NOT NULL
        """
    else:
        query = f"""
        SELECT f.flight_id
        FROM "{FLIGHT_TABLE}" f
        LEFT JOIN "{OUT_WEATHER_TABLE}" w
          ON f.flight_id = w.flight_id
        WHERE f.estdepartureairport IS NOT NULL
          AND f.estarrivalairport IS NOT NULL
          AND f.dep_time IS NOT NULL
          AND f.arr_time IS NOT NULL
          AND w.flight_id IS NULL
        """

    flight_ids = pd.read_sql_query(query, conn)["flight_id"].tolist()
    conn.close()

    print("Flights to process:", len(flight_ids))

    if not flight_ids:
        print("No new flights to process.")
        build_merged_table()
        return

    first_write = weather_table_exists.empty or rebuild_weather_table

    total_batches = math.ceil(len(flight_ids) / batch_size)

    for batch_num, batch_ids in enumerate(chunk_list(flight_ids, batch_size), start=1):
        print(f"\nProcessing batch {batch_num}/{total_batches} with {len(batch_ids)} flights...")

        weather_df = process_batch(batch_ids)

        conn = sqlite3.connect(DB_PATH)
        weather_df.to_sql(
            OUT_WEATHER_TABLE,
            conn,
            if_exists="replace" if first_write else "append",
            index=False
        )
        conn.close()

        first_write = False

        print(f"Batch {batch_num} done. Rows written: {len(weather_df)}")

        if "weather_error" in weather_df.columns:
            n_errors = weather_df["weather_error"].notna().sum()
            print("Rows with weather_error:", n_errors)

        miss_cols = [c for c in [
            "dep_weather_missing",
            "arr_weather_missing",
            "dep_temp_missing",
            "dep_wind_missing",
            "dep_wind_dir_missing",
            "arr_temp_missing",
            "arr_wind_missing",
            "arr_wind_dir_missing",
        ] if c in weather_df.columns]

        if miss_cols:
            print("Missing flags mean in batch:")
            print(weather_df[miss_cols].mean())

    print("\nAll batches processed.")
    build_merged_table()

    conn_check = sqlite3.connect(DB_PATH)

    print("\nRow count in feature table:")
    print(pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{FLIGHT_TABLE}"', conn_check))

    print("\nDistinct flights in feature table:")
    print(pd.read_sql_query(f'SELECT COUNT(DISTINCT flight_id) AS n FROM "{FLIGHT_TABLE}"', conn_check))

    print("\nRow count in weather table:")
    print(pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{OUT_WEATHER_TABLE}"', conn_check))

    print("\nDistinct flights in weather table:")
    print(pd.read_sql_query(f'SELECT COUNT(DISTINCT flight_id) AS n FROM "{OUT_WEATHER_TABLE}"', conn_check))

    conn_check.close()


if __name__ == "__main__":
    main(batch_size=20, rebuild_weather_table=False)