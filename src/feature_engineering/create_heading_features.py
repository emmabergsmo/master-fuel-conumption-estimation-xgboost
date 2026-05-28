"""Create heading and wind-component features for flight phases.

This script computes circular mean and final heading values for each flight
phase from raw ADS-B trajectory data. It then merges these heading features
with the existing feature table and derives departure and arrival headwind and
crosswind components using weather wind observations.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "data.sqlite"
TABLE = "adsb_fuel"
IN_FEATURE_TABLE = "flight_phase_features_physics_weather"
OUT_FEATURE_TABLE = "flight_phase_features_physics_weather_heading"

PHASES = {
    1: "takeoff",
    2: "climb",
    3: "cruise",
    4: "descent",
    5: "landing",
}


def circular_mean_deg(series: pd.Series) -> float:
    """Calculate the circular mean of heading angles in degrees."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    # Use circular statistics so headings near 0°/360° are averaged correctly
    radians = np.radians(s.to_numpy(dtype=float))
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))

    angle = np.degrees(np.arctan2(mean_sin, mean_cos))
    return float(angle % 360)


def build_heading_features(db_path=DB_PATH, raw_table=TABLE, limit_flights=None):
    """Build phase-level mean and last-heading features for each flight."""
    conn = sqlite3.connect(db_path)
    try:
        q = f'''
        SELECT DISTINCT flight_id
        FROM "{raw_table}"
        WHERE flight_id IS NOT NULL
        '''
        if limit_flights:
            q += f" LIMIT {int(limit_flights)}"

        fids = pd.read_sql(q, conn)["flight_id"].tolist()
        rows = []

        for fid in fids:
            df = pd.read_sql(
                f'''
                SELECT flight_id, phase_id, heading, postime
                FROM "{raw_table}"
                WHERE flight_id = ?
                ORDER BY postime
                ''',
                conn,
                params=(fid,)
            )

            if df.empty:
                continue

            row = {"flight_id": fid}

            # Build heading features separately for each labeled flight phase
            for phase_id, phase_name in PHASES.items():
                phase_df = df[df["phase_id"] == phase_id].copy()

                if phase_df.empty or phase_df["heading"].dropna().empty:
                    row[f"mean_heading_{phase_name}"] = np.nan
                    row[f"last_heading_{phase_name}"] = np.nan
                else:
                    heading_clean = pd.to_numeric(phase_df["heading"], errors="coerce").dropna()
                    row[f"mean_heading_{phase_name}"] = circular_mean_deg(heading_clean)
                    row[f"last_heading_{phase_name}"] = float(heading_clean.iloc[-1])

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    finally:
        conn.close()


def add_wind_components(df, prefix, heading_col):
    """Calculate headwind and crosswind components relative to a heading column."""
    wind_speed_col = f"{prefix}_wind_speed_mean"
    wind_dir_col = f"{prefix}_wind_dir_last"

    required = [wind_speed_col, wind_dir_col, heading_col]
    if not all(c in df.columns for c in required):
        return df

    # Convert wind direction relative to flight heading before decomposing wind
    rel = np.radians(df[wind_dir_col] - df[heading_col])

    # Headwind is positive for wind opposing the aircraft direction
    df[f"{prefix}_headwind"] = -df[wind_speed_col] * np.cos(rel)
    df[f"{prefix}_crosswind"] = np.abs(df[wind_speed_col] * np.sin(rel))
    return df


def main(limit_flights=None):
    """Merge heading and wind-component features into the feature table."""
    print("Building heading features...")
    heading_df = build_heading_features(limit_flights=limit_flights)
    print("Heading feature shape:", heading_df.shape)

    conn = sqlite3.connect(DB_PATH)

    print("Reading base table...")
    base_df = pd.read_sql_query(f'SELECT * FROM "{IN_FEATURE_TABLE}"', conn)
    print("Base shape:", base_df.shape)

    print("Merging heading features...")
    merged_df = base_df.merge(heading_df, on="flight_id", how="left")

    print("Adding wind components...")
    merged_df = add_wind_components(merged_df, "dep", "mean_heading_takeoff")
    merged_df = add_wind_components(merged_df, "arr", "mean_heading_landing")

    print("Output shape:", merged_df.shape)

    sample_cols = [
        "flight_id",
        "mean_heading_takeoff",
        "mean_heading_landing",
        "dep_wind_speed_mean",
        "dep_wind_dir_last",
        "dep_headwind",
        "dep_crosswind",
        "arr_wind_speed_mean",
        "arr_wind_dir_last",
        "arr_headwind",
        "arr_crosswind",
    ]
    sample_cols = [c for c in sample_cols if c in merged_df.columns]

    print("\nExample:")
    print(merged_df[sample_cols].head())

    merged_df.to_sql(OUT_FEATURE_TABLE, conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nWrote table: {OUT_FEATURE_TABLE}")


if __name__ == "__main__":
    main()