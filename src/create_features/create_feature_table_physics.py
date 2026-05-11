"""Add physics-inspired features to the flight feature table.

This script augments phase-level flight features with physics-based proxy
variables such as specific potential energy, specific kinetic energy, climb or
descent gradient, turn rate, dynamic pressure, and drag-work proxy. The output
table is used for XGBoost models that include physics-inspired predictors.
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "opensky.sqlite"
IN_FEATURE_TABLE = "flight_phase_features_weather_heading_v2"
OUT_FEATURE_TABLE = "flight_phase_features_physics_weather"

PHASES = ["takeoff", "climb", "cruise", "descent", "landing"]
G = 9.81

# ISA / standard-atmosphere constants
RHO0 = 1.225        # Air density at sea level [kg/m^3]
T0 = 288.15         # Sea-level temperature [K]
L = 0.0065          # Temperature lapse rate [K/m]
EXP = 4.2561        # Density exponent for the troposphere approximation
H_TROPO = 11000.0   # Troposphere height limit [m]


def safe_div(a, b):
    """Divide elementwise and return zero where the denominator is invalid."""
    a = pd.Series(a, copy=False).astype(float)
    b = pd.Series(b, copy=False).astype(float)

    out = np.zeros(len(a), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def air_density_isa(alt_m):
    """Estimate air density from altitude using a simplified ISA atmosphere model."""
    h = pd.Series(alt_m, copy=False).astype(float).clip(lower=0.0, upper=H_TROPO)
    rho = RHO0 * np.power(1.0 - (L * h / T0), EXP)
    rho = np.where(np.isfinite(rho), rho, RHO0)
    return rho


def ensure_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing phase feature columns and fill them with zeros."""
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
    """Add phase-level physics-inspired features to a feature table."""
    d = ensure_phase_columns(df)

    for phase in PHASES:
        alt_gain = d[f"alt_gain_m_{phase}"]
        alt_loss = d[f"alt_loss_m_{phase}"]
        dist = d[f"dist_m_{phase}"]
        time_s = d[f"time_s_{phase}"]
        mean_gs = d[f"mean_gs_mps_{phase}"]
        mean_alt = d[f"mean_alt_m_{phase}"]
        turn_sum = d[f"turn_sum_deg_{phase}"]

        # Potential-energy proxy based on altitude gain or loss during the phase
        if phase in ["takeoff", "climb"]:
            d[f"spec_pe_{phase}"] = G * alt_gain
        elif phase == "cruise":
            d[f"spec_pe_{phase}"] = G * (alt_gain + alt_loss)
        else: 
            d[f"spec_pe_{phase}"] = G * alt_loss

        # Kinetic-energy proxy based on mean ground speed
        d[f"spec_ke_{phase}"] = 0.5 * np.square(mean_gs)

        # Vertical gradient relative to horizontal distance traveled
        if phase in ["takeoff", "climb"]:
            d[f"gradient_{phase}"] = safe_div(alt_gain, dist)
        elif phase == "cruise":
            d[f"gradient_{phase}"] = safe_div(alt_gain + alt_loss, dist)
        else: 
            d[f"gradient_{phase}"] = safe_div(alt_loss, dist)

        # Average turning intensity during the phase
        d[f"turn_rate_degps_{phase}"] = safe_div(turn_sum, time_s)

        # Dynamic-pressure proxy using ISA-estimated air density
        rho = air_density_isa(mean_alt)
        d[f"dyn_press_{phase}"] = 0.5 * rho * np.square(mean_gs)

        # Drag-work proxy combining dynamic pressure and traveled distance
        d[f"drag_work_{phase}"] = d[f"dyn_press_{phase}"] * dist

    return d


def build_physics_feature_table(
    db_path=DB_PATH,
    in_table=IN_FEATURE_TABLE
) -> pd.DataFrame:
    """Read a feature table from SQLite and add physics-inspired features."""
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
    """Write the physics-augmented feature table to SQLite."""
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


def main():
    """Build and write the physics-augmented feature table."""
    df_features = build_physics_feature_table()
    print("Physics feature table shape:", df_features.shape)
    print(df_features.head(3).T)

    write_features_to_sqlite(df_features)
    print(f"Wrote physics-augmented features to table: {OUT_FEATURE_TABLE}")


if __name__ == "__main__":
    main()