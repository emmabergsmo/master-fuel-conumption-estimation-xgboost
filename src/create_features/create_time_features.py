import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "../opensky.sqlite"
TABLE_IN = "flight_phase_features_physics_weather"
TABLE_OUT = "flight_phase_features_physics_weather"


def add_time_parts_and_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for prefix in ["dep", "arr"]:
        time_col = f"{prefix}_time"
        if time_col not in df.columns:
            continue

        dt = pd.to_datetime(df[time_col], unit="s", utc=True, errors="coerce")

        # Extract time parts
        df[f"{prefix}_hour_utc"] = dt.dt.hour
        df[f"{prefix}_month"] = dt.dt.month
        df[f"{prefix}_dayofweek"] = dt.dt.dayofweek

        # Cyclical features
        df[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * df[f"{prefix}_hour_utc"] / 24)
        df[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * df[f"{prefix}_hour_utc"] / 24)

        df[f"{prefix}_month_sin"] = np.sin(2 * np.pi * df[f"{prefix}_month"] / 12)
        df[f"{prefix}_month_cos"] = np.cos(2 * np.pi * df[f"{prefix}_month"] / 12)

        df[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * df[f"{prefix}_dayofweek"] / 7)
        df[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * df[f"{prefix}_dayofweek"] / 7)

    return df


def main():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(f'SELECT * FROM "{TABLE_IN}"', conn)
    print("Input shape:", df.shape)

    df_out = add_time_parts_and_cyclical_features(df)

    # Not needed 
    cols_to_drop = [
        "dep_hour_utc", "dep_month", "dep_dayofweek",
        "arr_hour_utc", "arr_month", "arr_dayofweek",
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in df_out.columns]
    df_out = df_out.drop(columns=existing_drop_cols)

    print("Output shape:", df_out.shape)

    new_cols = [c for c in df_out.columns if c not in df.columns]
    print("\nNew columns:")
    print(new_cols)

    sample_cols = [
        "dep_time",
        "dep_hour_sin", "dep_hour_cos",
        "dep_month_sin", "dep_month_cos",
        "dep_dow_sin", "dep_dow_cos",
        "arr_time",
        "arr_hour_sin", "arr_hour_cos",
        "arr_dow_sin", "arr_dow_cos",
    ]
    sample_cols = [c for c in sample_cols if c in df_out.columns]

    print("\nExample:")
    print(df_out[sample_cols].head())

    df_out.to_sql(TABLE_OUT, conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nWrote table: {TABLE_OUT}")


if __name__ == "__main__":
    main()