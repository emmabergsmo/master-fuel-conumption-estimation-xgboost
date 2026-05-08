import sqlite3

import numpy as np
import pandas as pd

DB_PATH = "opensky.sqlite"
FLIGHT_TABLE = "flight_phase_features_v2"
RAW_WEATHER_TABLE = "flight_weather_observations_v2"
OUT_WEATHER_TABLE = "flight_weather_features_v3"
MERGED_TABLE = "flight_phase_features_weather_v3"

    
def empty_aggregates(prefix):
    return {
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


def aggregate_weather_features(df, prefix):
    out = empty_aggregates(prefix)

    valid_obs = df[df["elementId"].notna()].copy()
    if valid_obs.empty:
        return out

    out[f"{prefix}_weather_missing"] = 0

    temp = valid_obs[valid_obs["elementId"] == "air_temperature"]["value"].dropna()
    wind = valid_obs[valid_obs["elementId"] == "wind_speed"]["value"].dropna()
    wind_dir = valid_obs[valid_obs["elementId"] == "wind_from_direction"]["value"].dropna()

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
        if "referenceTime" in valid_obs.columns:
            wind_dir_df = valid_obs[valid_obs["elementId"] == "wind_from_direction"].copy()
            wind_dir_df["referenceTime"] = pd.to_datetime(
                wind_dir_df["referenceTime"],
                utc=True,
                errors="coerce",
            )
            wind_dir_df = wind_dir_df.sort_values("referenceTime")
            out[f"{prefix}_wind_dir_last"] = float(wind_dir_df["value"].dropna().iloc[-1])
        else:
            out[f"{prefix}_wind_dir_last"] = float(wind_dir.iloc[-1])

    return out


def station_metadata(df, prefix):
    out = {
        f"{prefix}_station_id": None,
        f"{prefix}_station_distance_km": None,
    }

    if df.empty:
        return out

    if "station_id" in df.columns and df["station_id"].notna().any():
        out[f"{prefix}_station_id"] = df["station_id"].dropna().iloc[0]

    if "station_distance_km" in df.columns and df["station_distance_km"].notna().any():
        out[f"{prefix}_station_distance_km"] = float(df["station_distance_km"].dropna().iloc[0])

    return out


def weather_error(df):
    if "weather_error" not in df.columns:
        return None

    errors = df["weather_error"].dropna().astype(str).unique().tolist()
    return "; ".join(errors) if errors else None


def build_weather_feature_table(raw_df):
    raw_df = raw_df.copy()
    raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce")

    rows = []
    for flight_id, flight_df in raw_df.groupby("flight_id", dropna=False):
        row = {"flight_id": flight_id}

        for prefix in ["dep", "arr"]:
            phase_df = flight_df[flight_df["airport_phase"] == prefix].copy()
            row.update(station_metadata(phase_df, prefix))
            row.update(aggregate_weather_features(phase_df, prefix))

        row["weather_error"] = weather_error(flight_df)
        rows.append(row)

    return pd.DataFrame(rows)


def add_wind_direction_features(df):
    out = df.copy()

    for prefix in ["dep", "arr"]:
        dir_col = f"{prefix}_wind_dir_last"
        if dir_col in out.columns:
            radians = np.radians(out[dir_col])
            out[f"{prefix}_wind_dir_sin"] = np.sin(radians)
            out[f"{prefix}_wind_dir_cos"] = np.cos(radians)
        else:
            out[f"{prefix}_wind_dir_sin"] = np.nan
            out[f"{prefix}_wind_dir_cos"] = np.nan

    return out


def build_merged_table():
    conn = sqlite3.connect(DB_PATH)
    try:
        features_df = pd.read_sql_query(f'SELECT * FROM "{FLIGHT_TABLE}"', conn)
        weather_df = pd.read_sql_query(f'SELECT * FROM "{OUT_WEATHER_TABLE}"', conn)

        merged_df = features_df.merge(weather_df, on="flight_id", how="left")
        merged_df = add_wind_direction_features(merged_df)

        merged_df.to_sql(MERGED_TABLE, conn, if_exists="replace", index=False)

        print("\nMerged shape:", merged_df.shape)
        print("\nWind direction columns:")
        print([c for c in merged_df.columns if "wind_dir" in c])
    finally:
        conn.close()


def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        raw_df = pd.read_sql_query(f'SELECT * FROM "{RAW_WEATHER_TABLE}"', conn)
        if raw_df.empty:
            print(f"No rows found in {RAW_WEATHER_TABLE}.")
            return

        weather_features = build_weather_feature_table(raw_df)
        weather_features.to_sql(OUT_WEATHER_TABLE, conn, if_exists="replace", index=False)

        print("Weather feature table shape:", weather_features.shape)
        print(f"Wrote weather features to table: {OUT_WEATHER_TABLE}")
    finally:
        conn.close()

    build_merged_table()

    conn_check = sqlite3.connect(DB_PATH)
    try:
        print("\nRow count in weather feature table:")
        print(pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{OUT_WEATHER_TABLE}"', conn_check))
        print("\nDistinct flights in weather feature table:")
        print(
            pd.read_sql_query(
                f'SELECT COUNT(DISTINCT flight_id) AS n FROM "{OUT_WEATHER_TABLE}"',
                conn_check,
            )
        )
    finally:
        conn_check.close()


if __name__ == "__main__":
    main()
