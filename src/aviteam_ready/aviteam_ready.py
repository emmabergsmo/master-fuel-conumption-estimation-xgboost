"""Create an AviTEAM-compatible SQLite table from ADS-B data.

This script renames and selects columns from the ADS-B fuel table so that the
data structure matches the format expected by AviTEAM. The output table keeps
trajectory, airport, flight, and phase information, and adds AviTEAM status
columns such as 'ground', 'completed', 'completed_adv', and 'r'.
"""

import sqlite3

DB_PATH = "opensky_updated.sqlite"
IN_TABLE = "adsb_fuel_v2"
OUT_TABLE = "aviteam_ready_phase"


def main():
    """Create an AviTEAM-compatible table from ADS-B fuel data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"DROP TABLE IF EXISTS {OUT_TABLE};")

    cursor.execute(f"""
        CREATE TABLE {OUT_TABLE} AS
        SELECT 
            icao24 AS hex,
            estdepartureairport AS port_o,
            estarrivalairport AS port_d,
            callsign AS flight,
            aircraft_type_icao_code AS t,
            lon,
            lat,
            postime,
            baroaltitude AS alt_baro,
            geoaltitude AS alt_geom,
            velocity AS gs,
            heading AS track,
            vertrate AS baro_rate,
            squawk,
            flight_id,
            phase_id,
            phase_name,
            FALSE AS ground,
            FALSE AS completed,
            FALSE AS completed_adv,
            NULL as r
        FROM {IN_TABLE};
    """)

    conn.commit()
    conn.close()
    print(f"Created table: {OUT_TABLE}")

if __name__ == "__main__":
    main()