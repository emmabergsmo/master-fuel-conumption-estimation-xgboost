import sqlite3
import numpy as np
import pandas as pd

"""
This script labels each ADS-B point with a flight phase (takeoff, climb, cruise, descent, landing, unknown) based on heuristics using altitude, vertical rate, and ground speed. The resulting phase_id and phase_name are written back to the same table in the SQLite database.
"""

PHASE_NAMES = {
    1: "takeoff",
    2: "climb",
    3: "cruise",
    4: "descent",
    5: "landing",
    6: "unknown",
}

def ensure_columns(conn: sqlite3.Connection, table: str):
    cols = pd.read_sql(f'PRAGMA table_info("{table}")', conn)["name"].tolist()
    if "phase_id" not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN phase_id INTEGER')
    if "phase_name" not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN phase_name TEXT')
    conn.commit()

def majority_filter(phases: np.ndarray, window: int = 5) -> np.ndarray:
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
    """
    SI units:
      geoaltitude: meters
      velocity: m/s
      vertrate: m/s
      postime: seconds
    """
    d = df.sort_values("postime").reset_index(drop=True).copy()

    alt = d["geoaltitude"].astype(float).to_numpy()
    gs  = d["velocity"].astype(float).to_numpy()
    vr  = d["vertrate"].astype(float).to_numpy()

    # smooth vertical rate (median) to reduce noise
    vr_s = pd.Series(vr).rolling(3, center=True, min_periods=1).median().to_numpy()

    # relative altitude vs start/end (proxy for "near departure/arrival")
    alt_dep = alt[0]
    alt_arr = alt[-1]
    rel_dep = alt - alt_dep
    rel_arr = alt - alt_arr

    # thresholds 
    TAKEOFF_MAX_REL_ALT_M = 450.0   
    LANDING_MAX_REL_ALT_M = 450.0     

    CLIMB_VR_MIN_MPS = 2.5            
    DESC_VR_MAX_MPS  = -2.5
    CRUISE_VR_ABS_MPS = 1.0          
    CRUISE_MIN_ALT_M  = 6100.0        

    # A tiny speed gate can help reduce false takeoff/landing at very low gs
    MIN_AIRBORNE_GS_MPS = 30.0      

    phases = np.full(len(d), 6, dtype=int)

    for i in range(len(d)):
        a0 = rel_dep[i]
        a1 = rel_arr[i]
        v = vr_s[i]
        a = alt[i]
        g = gs[i]

        # takeoff: early, climbing, moving fast
        if 0 <= a0 <= TAKEOFF_MAX_REL_ALT_M and v > 0.5 and g >= MIN_AIRBORNE_GS_MPS:
            phases[i] = 1
            continue

        # landing: late, descending, moving (still) reasonably
        if 0 <= a1 <= LANDING_MAX_REL_ALT_M and v < -0.5 and g >= 20.0:
            phases[i] = 5
            continue

        # climb / descent
        if v >= CLIMB_VR_MIN_MPS:
            phases[i] = 2
            continue
        if v <= DESC_VR_MAX_MPS:
            phases[i] = 4
            continue

        # cruise
        if a >= CRUISE_MIN_ALT_M and abs(v) <= CRUISE_VR_ABS_MPS:
            phases[i] = 3
            continue

        # fallback cruise-ish
        if a >= 4500.0 and abs(v) <= 1.5:
            phases[i] = 3

    # stabilize: majority + minimum dwell
    phases = majority_filter(phases, window=5)
    phases = enforce_min_dwell(phases, min_dwell=2)

    out = d[["flight_id", "postime"]].copy()
    out["phase_id"] = phases
    out["phase_name"] = out["phase_id"].map(PHASE_NAMES).fillna("unknown")
    return out

def write_phases_to_db(db_path: str, table: str):
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

if __name__ == "__main__":
    DB_PATH = "opensky.sqlite"
    TABLE = "adsb_fuel_v2" 
    write_phases_to_db(DB_PATH, TABLE)
    print("Done: phase_id + phase_name written.")
