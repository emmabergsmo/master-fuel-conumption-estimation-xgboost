import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "opensky.sqlite"
TABLE = "adsb_fuel"         
OUT_FEATURE_TABLE = "flight_phase_features"  

# Phase IDs we model (as defined in set_flight_phases.py)
PHASES = {
    1: "takeoff",
    2: "climb",
    3: "cruise",
    4: "descent",
    5: "landing",
}

FUEL_COLS = [
    "takeoff_fuel", "climb_fuel", "cruise_fuel", "descent_fuel", "landing_fuel",
    "block_fuel", "taxi_out_fuel", "taxi_in_fuel"
]

META_COLS = [
    "flight_id", "icao24", "callsign", "aircraft_type_icao_code",
    "estdepartureairport", "estarrivalairport", "great_circle_distance"
]

POINT_COLS = [
    "postime", "lon", "lat", "geoaltitude", "velocity", "heading", "vertrate", "phase_id"
]


def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in meters."""
    R = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def heading_delta_deg(h1, h2):
    """Smallest absolute heading delta in degrees (0..180), vectorized."""
    d = (h2 - h1 + 180) % 360 - 180
    return np.abs(d)


def compute_point_deltas(df_flight: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-row:
      - dt_s: time delta to next point (seconds); last row dt_s=0
      - dist_m: distance to next point (meters); last row dist_m=0
      - dh_m: altitude change to next point (meters); last row dh_m=0
      - dhead_deg: abs heading change to next point (deg); last row 0
    Works with mixed sampling (10s + 60s) automatically.
    """
    d = df_flight.sort_values("postime").reset_index(drop=True).copy()

    t = d["postime"].to_numpy(dtype=float)
    lon = d["lon"].to_numpy(dtype=float)
    lat = d["lat"].to_numpy(dtype=float)
    alt = d["geoaltitude"].to_numpy(dtype=float)
    head = d["heading"].to_numpy(dtype=float)

    # next arrays (shift -1)
    t2 = np.roll(t, -1); lon2 = np.roll(lon, -1); lat2 = np.roll(lat, -1)
    alt2 = np.roll(alt, -1); head2 = np.roll(head, -1)

    dt = t2 - t
    dt[-1] = 0.0
    # clip dt in case of duplicates or out-of-order weirdness
    dt = np.clip(dt, 0.0, 600.0)

    dist = haversine_m(lon, lat, lon2, lat2)
    dist[-1] = 0.0
    # if any NaNs
    dist = np.nan_to_num(dist, nan=0.0)

    dh = alt2 - alt
    dh[-1] = 0.0
    dh = np.nan_to_num(dh, nan=0.0)

    dhead = heading_delta_deg(head, head2)
    dhead[-1] = 0.0
    dhead = np.nan_to_num(dhead, nan=0.0)

    d["dt_s"] = dt
    d["dist_m"] = dist
    d["dh_m"] = dh
    d["dhead_deg"] = dhead
    return d


def aggregate_phase_features(df_flight: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one-row dataframe for this flight_id with wide phase features + labels.
    """
    # ensure required columns
    missing = {"flight_id", "postime", "phase_id", "lon", "lat", "geoaltitude", "velocity", "heading", "vertrate"} - set(df_flight.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # compute deltas
    d = compute_point_deltas(df_flight)

    # keep only modeled phases; unknowns are dropped
    d = d[d["phase_id"].isin(PHASES.keys())].copy()
    if d.empty:
        return pd.DataFrame()

    # helper metrics
    d["alt_gain_m"] = np.clip(d["dh_m"], 0, None)
    d["alt_loss_m"] = np.clip(-d["dh_m"], 0, None)
    d["abs_vr_mps"] = np.abs(d["vertrate"].astype(float))

    # aggregate per phase
    g = d.groupby("phase_id", as_index=False).agg(
        time_s=("dt_s", "sum"),
        dist_m=("dist_m", "sum"),
        alt_gain_m=("alt_gain_m", "sum"),
        alt_loss_m=("alt_loss_m", "sum"),
        mean_alt_m=("geoaltitude", "mean"),
        max_alt_m=("geoaltitude", "max"),
        mean_gs_mps=("velocity", "mean"),
        p95_gs_mps=("velocity", lambda x: np.nanpercentile(x, 95)),
        std_gs_mps=("velocity", "std"),
        mean_vr_mps=("vertrate", "mean"),
        p95_abs_vr_mps=("abs_vr_mps", lambda x: np.nanpercentile(x, 95)),
        turn_sum_deg=("dhead_deg", "sum"),
        n_points=("postime", "count"),
    )

     # Build wide phase features explicitly: metric_phase (e.g., time_s_takeoff)
    out = {}
    for _, row in g.iterrows():
        p = PHASES[int(row["phase_id"])]
        for metric in [
            "time_s","dist_m","alt_gain_m","alt_loss_m","mean_alt_m","max_alt_m",
            "mean_gs_mps","p95_gs_mps","std_gs_mps","mean_vr_mps","p95_abs_vr_mps",
            "turn_sum_deg","n_points"
        ]:
            out[f"{metric}_{p}"] = float(row[metric]) if pd.notna(row[metric]) else np.nan

    # flight-level totals (from the labeled phases only)
    out["time_s_modeled"] = float(d["dt_s"].sum())
    out["dist_m_modeled"] = float(d["dist_m"].sum())

    # path ratio
    gcd = df_flight["great_circle_distance"].iloc[0] if "great_circle_distance" in df_flight.columns else np.nan
    out["great_circle_distance"] = float(gcd) * 1000 if pd.notna(gcd) else np.nan
    if pd.notna(out["great_circle_distance"]) and out["great_circle_distance"] > 0:
        out["path_ratio"] = out["dist_m_modeled"] / out["great_circle_distance"]
    else:
        out["path_ratio"] = np.nan

    # metadata (take first non-null per flight)
    meta = {}
    for c in META_COLS:
        if c in df_flight.columns:
            v = df_flight[c].dropna().iloc[0] if df_flight[c].notna().any() else None
            meta[c] = v

    # labels (fuel per phase) – same for whole flight, grab first non-null
    labels = {}
    for c in FUEL_COLS:
        if c in df_flight.columns:
            v = df_flight[c].dropna().iloc[0] if df_flight[c].notna().any() else np.nan
            labels[c] = float(v) if pd.notna(v) else np.nan

    row = {**meta, **out, **labels}
    return pd.DataFrame([row])


def build_feature_table(db_path=DB_PATH, table=TABLE, limit_flights=None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    try:
        # get flight_ids
        q = f'''
        SELECT DISTINCT flight_id
        FROM "{table}"
        WHERE estdepartureairport IS NOT NULL
        AND estarrivalairport IS NOT NULL
        AND estdepartureairport != estarrivalairport
        '''
        if limit_flights:
            q += f" LIMIT {int(limit_flights)}"
        fids = pd.read_sql(q, conn)["flight_id"].tolist()

        rows = []
        for fid in fids:
            df = pd.read_sql(
                f'''
                SELECT {", ".join(META_COLS + POINT_COLS + FUEL_COLS)}
                FROM "{table}"
                WHERE flight_id = ?
                ORDER BY postime
                ''',
                conn,
                params=(fid,)
            )

            if df.empty or df["phase_id"].isna().all():
                continue

            feat = aggregate_phase_features(df)
            if not feat.empty:
                rows.append(feat)

        if not rows:
            return pd.DataFrame()

        feat_all = pd.concat(rows, ignore_index=True)

        # fill missing phase metrics with 0 (common for tree models)
        phase_metric_cols = [c for c in feat_all.columns if any(c.endswith("_"+p) for p in PHASES.values())]
        feat_all[phase_metric_cols] = feat_all[phase_metric_cols].fillna(0.0)

        # create total label
        if all(c in feat_all.columns for c in ["takeoff_fuel","climb_fuel","cruise_fuel","descent_fuel","landing_fuel"]):
            feat_all["total_fuel_modeled"] = feat_all[
            ["takeoff_fuel","climb_fuel","cruise_fuel","descent_fuel","landing_fuel"]
              ].sum(axis=1)

        return feat_all

    finally:
        conn.close()


def write_features_to_sqlite(df_features: pd.DataFrame, db_path=DB_PATH, out_table=OUT_FEATURE_TABLE):
    conn = sqlite3.connect(db_path)
    try:
        df_features.to_sql(out_table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{out_table}_flight_id ON "{out_table}"(flight_id)')
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    df_features = build_feature_table(limit_flights=None)  
    print("Feature table shape:", df_features.shape)
    print(df_features.head(3).T)

    # write back to sqlite
    write_features_to_sqlite(df_features)
    print(f"Wrote features to table: {OUT_FEATURE_TABLE}")
