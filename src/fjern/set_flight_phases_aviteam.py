"""Assign AviTEAM-compatible flight phase labels to trajectory points.

This script labels each ADS-B trajectory point with a flight phase using
heuristics based on geometric altitude, ground speed, and vertical rate.
The labels are intended for the AviTEAM-ready dataset and distinguish
takeoff, climb, cruise, descent, landing, and unknown phases.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "opensky.sqlite"
TABLE = "adsb_fuel_v2" 

PHASE_NAMES = {
    1: "takeoff",
    2: "climb",
    3: "cruise",
    4: "descent",
    5: "landing",
    6: "unknown",
}

def ensure_columns(conn: sqlite3.Connection, table: str):
    """Add phase label columns to the SQLite table if they do not exist."""
    cols = pd.read_sql(f'PRAGMA table_info("{table}")', conn)["name"].tolist()
    if "phase_id" not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN phase_id INTEGER')
    if "phase_name" not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN phase_name TEXT')
    conn.commit()

def majority_filter(phases: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth phase labels using a centered rolling majority filter."""
    s = pd.Series(phases)
    def mode_or_center(x):
        vc = x.value_counts()
        if len(vc) == 0:
            return x.iloc[len(x)//2]
        # mode; tie -> center
        if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
            return x.iloc[len(x)//2]
        return vc.index[0]
    return (
        s.rolling(window, center=True, min_periods=1)
         .apply(mode_or_center, raw=False)
         .astype(int)
         .to_numpy()
    )

def enforce_min_dwell(phases: np.ndarray, min_dwell: int = 2) -> np.ndarray:
    """Replace phase segments shorter than `min_dwell` with a neighboring phase."""
    phases = phases.copy()
    n = len(phases)
    i = 0
    while i < n:
        j = i + 1
        while j < n and phases[j] == phases[i]:
            j += 1
        if (j - i) < min_dwell:
            left = phases[i-1] if i-1 >= 0 else None
            right = phases[j] if j < n else None
            # prefer non-unknown neighbor
            cand = None
            for c in (left, right):
                if c is not None and c != 6:
                    cand = c
                    break
            if cand is None:
                cand = left if left is not None else (right if right is not None else phases[i])
            phases[i:j] = cand
        i = j
    return phases

def label_phases_one_flight(df: pd.DataFrame) -> pd.DataFrame:
    """Assign numeric flight phase labels to one ordered flight trajectory."""
    d = df.sort_values("postime").reset_index().copy()
    original_index_col = "index"

    alt = d["alt_geom"].astype(float).to_numpy()
    gs = d["gs"].astype(float).to_numpy()
    vr = d["baro_rate"].astype(float).to_numpy()

    vr_s = pd.Series(vr).rolling(3, center=True, min_periods=1).median().to_numpy()

    alt_dep = alt[0]
    alt_arr = alt[-1]
    rel_dep = alt - alt_dep
    rel_arr = alt - alt_arr

    TAKEOFF_MAX_REL_ALT_M = 500.0
    LANDING_MAX_REL_ALT_M = 900.0

    CLIMB_VR_MIN_MPS = 2.5
    DESC_VR_MAX_MPS = -2.5
    CRUISE_VR_ABS_MPS = 1.0
    CRUISE_MIN_ALT_M = 6100.0

    MIN_AIRBORNE_GS_MPS = 30.0

    phases = np.full(len(d), 6, dtype=int)

    for i in range(len(d)):
        a0 = rel_dep[i]
        a1 = rel_arr[i]
        v = vr_s[i]
        a = alt[i]
        g = gs[i]

        if 0 <= a0 <= TAKEOFF_MAX_REL_ALT_M and v > 0.5 and g >= MIN_AIRBORNE_GS_MPS:
            phases[i] = 1
            continue

        if 0 <= a1 <= LANDING_MAX_REL_ALT_M and v < -0.5 and g >= 20.0:
            phases[i] = 5
            continue

        if v >= CLIMB_VR_MIN_MPS:
            phases[i] = 2
            continue

        if v <= DESC_VR_MAX_MPS:
            phases[i] = 4
            continue

        if a >= CRUISE_MIN_ALT_M and abs(v) <= CRUISE_VR_ABS_MPS:
            phases[i] = 3
            continue

        if a >= 4500.0 and abs(v) <= 1.5:
            phases[i] = 3

    phases = majority_filter(phases, window=5)
    phases = enforce_min_dwell(phases, min_dwell=2)

    out = d[[original_index_col]].copy()
    out["flight_phase_new"] = phases  # <-- NUMERISK
    return out

def write_phases_to_db(db_path: str, table: str):
    """Compute phase labels for each flight and write them back to SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        ensure_columns(conn, table)

        # speed indexes
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_fid_time ON "{table}"(flight_id, postime)')
        conn.commit()

        fids = pd.read_sql(f'SELECT DISTINCT flight_id FROM "{table}"', conn)["flight_id"].tolist()
        cur = conn.cursor()

        for fid in fids:
            df = pd.read_sql(
                f'''
                SELECT flight_id, postime, geoaltitude, velocity, vertrate
                FROM "{table}"
                WHERE flight_id = ?
                ORDER BY postime
                ''',
                conn,
                params=(fid,)
            )
            if df.empty or len(df) < 5:
                continue

            upd = label_phases_one_flight(df)

            cur.executemany(
                f'UPDATE "{table}" SET phase_id = ?, phase_name = ? WHERE flight_id = ? AND postime = ?',
                [(int(r.phase_id), str(r.phase_name), str(r.flight_id), int(r.postime))
                for r in upd.itertuples(index=False)]

            )
            conn.commit()

    finally:
        conn.close()

def main():
    """Assign flight phases and write them to SQLite."""
    write_phases_to_db(DB_PATH, TABLE)
    print("Done: phase_id + phase_name written.")

if __name__ == "__main__":
    main()
