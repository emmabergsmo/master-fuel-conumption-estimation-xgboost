import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "../opensky.sqlite"
IN_FEATURE_TABLE = "flight_phase_features_weather_heading_v2"
OUT_FEATURE_TABLE = "flight_phase_features_physics_weather"

PHASES = ["takeoff", "climb", "cruise", "descent", "landing"]
G = 9.81

# ISA / standard atmosphere constants
RHO0 = 1.225      # kg/m^3 at sea level
T0 = 288.15       # K
L = 0.0065        # K/m
EXP = 4.2561      # exponent for density approximation in troposphere
H_TROPO = 11000.0 # m


def safe_div(a, b):
    """
    Elementwise safe division.
    Returns 0.0 where denominator is 0 or NaN.
    """
    a = pd.Series(a, copy=False).astype(float)
    b = pd.Series(b, copy=False).astype(float)

    out = np.zeros(len(a), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def air_density_isa(alt_m):
    """
    Approximate air density [kg/m^3] from altitude [m]
    using a simple ISA-style troposphere model.

    Clipped to [0, 11000] m to stay in a stable regime.
    """
    h = pd.Series(alt_m, copy=False).astype(float).clip(lower=0.0, upper=H_TROPO)
    rho = RHO0 * np.power(1.0 - (L * h / T0), EXP)
    rho = np.where(np.isfinite(rho), rho, RHO0)
    return rho


def ensure_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required phase columns exist.
    Missing columns are created and filled with 0.0.
    """
    d = df.copy()

    required_prefixes = [
        "alt_gain_m",
        "alt_loss_m",
        "dist_m",
        "time_s",
        "mean_gs_mps",
        "mean_alt_m",
        "turn_sum_deg",
    ]

    for phase in PHASES:
        for prefix in required_prefixes:
            col = f"{prefix}_{phase}"
            if col not in d.columns:
                d[col] = 0.0

    return d


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-inspired no-leakage features to the existing flight feature table.

    New features per phase:
      - spec_pe_<phase>
      - spec_ke_<phase>
      - gradient_<phase>
      - turn_rate_degps_<phase>
      - dyn_press_<phase>
      - drag_work_<phase>
    """
    d = ensure_phase_columns(df)

    for phase in PHASES:
        alt_gain = d[f"alt_gain_m_{phase}"]
        alt_loss = d[f"alt_loss_m_{phase}"]
        dist = d[f"dist_m_{phase}"]
        time_s = d[f"time_s_{phase}"]
        mean_gs = d[f"mean_gs_mps_{phase}"]
        mean_alt = d[f"mean_alt_m_{phase}"]
        turn_sum = d[f"turn_sum_deg_{phase}"]

        # 1) Specific potential energy proxy
        if phase in ["takeoff", "climb"]:
            d[f"spec_pe_{phase}"] = G * alt_gain
        elif phase == "cruise":
            d[f"spec_pe_{phase}"] = G * (alt_gain + alt_loss)
        else:  # descent, landing
            d[f"spec_pe_{phase}"] = G * alt_loss

        # 2) Specific kinetic energy proxy
        d[f"spec_ke_{phase}"] = 0.5 * np.square(mean_gs)

        # 3) Gradient
        if phase in ["takeoff", "climb"]:
            d[f"gradient_{phase}"] = safe_div(alt_gain, dist)
        elif phase == "cruise":
            d[f"gradient_{phase}"] = safe_div(alt_gain + alt_loss, dist)
        else:  # descent, landing
            d[f"gradient_{phase}"] = safe_div(alt_loss, dist)

        # 4) Turn rate
        d[f"turn_rate_degps_{phase}"] = safe_div(turn_sum, time_s)

        # 5) Dynamic pressure proxy: q = 0.5 * rho(h) * v^2
        rho = air_density_isa(mean_alt)
        d[f"dyn_press_{phase}"] = 0.5 * rho * np.square(mean_gs)

        # 6) Drag-work proxy: q * distance
        d[f"drag_work_{phase}"] = d[f"dyn_press_{phase}"] * dist

    return d


def build_physics_feature_table(
    db_path=DB_PATH,
    in_table=IN_FEATURE_TABLE
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f'SELECT * FROM "{in_table}"', conn)
        if df.empty:
            return pd.DataFrame()

        df_out = add_physics_features(df)
        return df_out

    finally:
        conn.close()


def write_features_to_sqlite(
    df_features: pd.DataFrame,
    db_path=DB_PATH,
    out_table=OUT_FEATURE_TABLE
):
    conn = sqlite3.connect(db_path)
    try:
        df_features.to_sql(out_table, conn, if_exists="replace", index=False)
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{out_table}_flight_id '
            f'ON "{out_table}"(flight_id)'
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    df_features = build_physics_feature_table()
    print("Physics feature table shape:", df_features.shape)
    print(df_features.head(3).T)

    write_features_to_sqlite(df_features)
    print(f"Wrote physics-augmented features to table: {OUT_FEATURE_TABLE}")