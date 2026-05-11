"""Retrieve ADS-B state vectors and attach recorded fuel values.

This script extends the ADS-B retrieval step by joining each flight with fuel
consumption values from the matched flight table. It samples the first and last
two minutes of each flight at 10-second resolution and the middle of the flight
at 60-second resolution, giving denser coverage during takeoff and landing.
"""

import sqlite3
from trino_client import get_trino_connection
from decimal import Decimal

# Convert Decimal values from SQLite/Trino-compatible data to floats before insertion
sqlite3.register_adapter(Decimal, float)

DB_PATH = "opensky.sqlite"
IN_TABLE = "norwegian_flights_2022_with_type_fuel_v2"
TRINO_STATE_TABLE = "state_vectors_data4" 
OUT_TABLE = "adsb_fuel_v2"

TIME_START = 1671663600  # 2022-12-21 00:00:00
TIME_END   = 1672531200  # 2023-01-01 00:00:00

FLIGHT_CHUNK_SIZE = 50       
COMMIT_EVERY_CHUNKS = 10     


DDL = f"""
CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
    icao24 TEXT,
    estdepartureairport TEXT,
    estarrivalairport TEXT,
    callsign TEXT,
    aircraft_type_icao_code TEXT,
    lon REAL,
    lat REAL,
    postime INTEGER,
    baroaltitude REAL,
    geoaltitude REAL,
    velocity REAL,
    heading REAL,
    vertrate REAL,
    squawk TEXT, 
    block_fuel REAL,
    taxi_out_fuel REAL,
    takeoff_fuel REAL,
    climb_fuel REAL,
    cruise_fuel REAL,
    descent_fuel REAL,
    landing_fuel REAL,
    taxi_in_fuel REAL,
    great_circle_distance REAL,
    UNIQUE (icao24, postime)
);
"""


INSERT_SQL = f"""
INSERT OR IGNORE INTO {OUT_TABLE} (
  icao24, estdepartureairport, estarrivalairport, callsign, aircraft_type_icao_code,
  lon, lat, postime, baroaltitude, geoaltitude, velocity, heading, vertrate, squawk, 
  block_fuel, taxi_out_fuel, takeoff_fuel, climb_fuel, cruise_fuel, descent_fuel, 
  landing_fuel, taxi_in_fuel, great_circle_distance
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""



def trino_sql_literal(v) -> str:
    """Return a SQL-safe literal value for use in Trino VALUES clauses."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).strip().replace("'", "''")
    return f"'{s}'"


def hour_floor(ts: int) -> int:
    """Round a Unix timestamp down to the start of its hour."""
    return ts - (ts % 3600)


GLOBAL_START_H = hour_floor(TIME_START)
GLOBAL_END_H   = hour_floor(TIME_END)


def chunked(seq, n):
    """Yield successive chunks of size 'n' from a sequence."""
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    """Retrieve ADS-B observations with fuel labels and store them in SQLite."""
    sqlite_conn = sqlite3.connect(DB_PATH)

    # Improve SQLite write performance for large batch inserts
    sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    sqlite_conn.executescript(DDL)

    read_cur = sqlite_conn.cursor()
    write_cur = sqlite_conn.cursor()

    trino_conn = get_trino_connection()
    trino_cur = trino_conn.cursor()

    flights = list(read_cur.execute(
        f"""
        SELECT
          icao24,
          estdepartureairport,
          estarrivalairport,
          callsign,
          aircraft_type_icao,
          firstseen,
          lastseen,
          block_fuel,
          taxi_out_fuel,
          takeoff_fuel,
          climb_fuel,
          cruise_fuel,
          descent_fuel,
          landing_fuel,
          taxi_in_fuel,
          great_circle_distance
        FROM {IN_TABLE}
        WHERE icao24 IS NOT NULL
          AND firstseen IS NOT NULL
          AND lastseen IS NOT NULL
          AND NOT (lastseen < {TIME_START} OR firstseen >= {TIME_END})
        """
    ))

    print("Flights to process in window:", len(flights))

    total_points = 0
    chunk_count = 0

    try:
        for flight_chunk in chunked(flights, FLIGHT_CHUNK_SIZE):
            # Build a temporary VALUES table for this chunk of flights in the Trino query
            values_rows = []

            for (
                icao24, dep, arr, callsign, ac_type,firstseen, lastseen,
                block_fuel, taxi_out_fuel, takeoff_fuel, climb_fuel,
                cruise_fuel, descent_fuel, landing_fuel, taxi_in_fuel,
                gcd,
            ) in flight_chunk:
                firstseen = int(firstseen)
                lastseen = int(lastseen)

                # Clip each flight to the selected time window
                start_t = max(firstseen, TIME_START)
                end_t = min(lastseen, TIME_END)
                if start_t >= end_t:
                    continue

                start_h = hour_floor(start_t)
                end_h = hour_floor(end_t)

                values_rows.append(
                    "(" + ",".join([
                        trino_sql_literal(icao24),
                        trino_sql_literal(dep),
                        trino_sql_literal(arr),
                        trino_sql_literal(callsign),
                        trino_sql_literal(ac_type),
                        str(start_t),
                        str(end_t),
                        str(start_h),
                        str(end_h),
                        trino_sql_literal(block_fuel),
                        trino_sql_literal(taxi_out_fuel),
                        trino_sql_literal(takeoff_fuel),
                        trino_sql_literal(climb_fuel),
                        trino_sql_literal(cruise_fuel),
                        trino_sql_literal(descent_fuel),
                        trino_sql_literal(landing_fuel),
                        trino_sql_literal(taxi_in_fuel),
                        trino_sql_literal(gcd),
                    ]) + ")"
                )

            if not values_rows:
                continue

            trino_sql = f"""
            WITH flights(
                icao24, dep, arr, callsign, ac_type,
                start_t, end_t, start_h, end_h,
                block_fuel, taxi_out_fuel, takeoff_fuel, climb_fuel,
                cruise_fuel, descent_fuel, landing_fuel, taxi_in_fuel, gcd
            ) AS (
                VALUES {",".join(values_rows)}
            ),

            -- =========================
            -- 1) 10-seconds sampling for takeoff and landing phases
            -- =========================
            dense AS (
                SELECT
                    f.icao24,
                    f.dep AS estdepartureairport,
                    f.arr AS estarrivalairport,
                    f.callsign,
                    f.ac_type AS aircraft_type_icao_code,

                    max_by(sv.lon, sv.time)           AS lon,
                    max_by(sv.lat, sv.time)           AS lat,
                    (sv.time - (sv.time % 10))        AS postime,
                    max_by(sv.baroaltitude, sv.time)  AS baroaltitude,
                    max_by(sv.geoaltitude, sv.time)   AS geoaltitude,
                    max_by(sv.velocity, sv.time)      AS velocity,
                    max_by(sv.heading, sv.time)       AS heading,
                    max_by(sv.vertrate, sv.time)      AS vertrate,
                    max_by(sv.squawk, sv.time)        AS squawk,

                    f.block_fuel,
                    f.taxi_out_fuel,
                    f.takeoff_fuel,
                    f.climb_fuel,
                    f.cruise_fuel,
                    f.descent_fuel,
                    f.landing_fuel,
                    f.taxi_in_fuel,
                    f.gcd AS great_circle_distance

                FROM {TRINO_STATE_TABLE} sv
                JOIN flights f
                    ON sv.icao24 = f.icao24
                AND sv.time BETWEEN f.start_t AND f.end_t

                WHERE sv.hour BETWEEN {GLOBAL_START_H} AND {GLOBAL_END_H}
                AND (
                        sv.time <  f.start_t + 120
                    OR sv.time >= f.end_t   - 120
                )

                GROUP BY
                    f.icao24, f.dep, f.arr, f.callsign, f.ac_type,
                    f.block_fuel, f.taxi_out_fuel, f.takeoff_fuel, f.climb_fuel,
                    f.cruise_fuel, f.descent_fuel, f.landing_fuel, f.taxi_in_fuel,
                    f.gcd,
                    (sv.time - (sv.time % 10))
            ),

            -- =========================
            -- 2) 60-seconds sampling 
            -- =========================
            coarse AS (
                SELECT
                    f.icao24,
                    f.dep AS estdepartureairport,
                    f.arr AS estarrivalairport,
                    f.callsign,
                    f.ac_type AS aircraft_type_icao_code,

                    max_by(sv.lon, sv.time)           AS lon,
                    max_by(sv.lat, sv.time)           AS lat,
                    (sv.time - (sv.time % 60))        AS postime,
                    max_by(sv.baroaltitude, sv.time)  AS baroaltitude,
                    max_by(sv.geoaltitude, sv.time)   AS geoaltitude,
                    max_by(sv.velocity, sv.time)      AS velocity,
                    max_by(sv.heading, sv.time)       AS heading,
                    max_by(sv.vertrate, sv.time)      AS vertrate,
                    max_by(sv.squawk, sv.time)        AS squawk,

                    f.block_fuel,
                    f.taxi_out_fuel,
                    f.takeoff_fuel,
                    f.climb_fuel,
                    f.cruise_fuel,
                    f.descent_fuel,
                    f.landing_fuel,
                    f.taxi_in_fuel,
                    f.gcd AS great_circle_distance

                FROM {TRINO_STATE_TABLE} sv
                JOIN flights f
                    ON sv.icao24 = f.icao24
                AND sv.time BETWEEN f.start_t AND f.end_t

                WHERE sv.hour BETWEEN {GLOBAL_START_H} AND {GLOBAL_END_H}
                AND sv.time >= f.start_t + 120
                AND sv.time <  f.end_t   - 120

                GROUP BY
                    f.icao24, f.dep, f.arr, f.callsign, f.ac_type,
                    f.block_fuel, f.taxi_out_fuel, f.takeoff_fuel, f.climb_fuel,
                    f.cruise_fuel, f.descent_fuel, f.landing_fuel, f.taxi_in_fuel,
                    f.gcd,
                    (sv.time - (sv.time % 60))
            )

            SELECT * FROM dense
            UNION ALL
            SELECT * FROM coarse
            ORDER BY icao24, postime
            """

            trino_cur.execute(trino_sql)
            rows = trino_cur.fetchall()

            if rows:
                write_cur.executemany(INSERT_SQL, rows)
                total_points += len(rows)

            chunk_count += 1
            if chunk_count % COMMIT_EVERY_CHUNKS == 0:
                sqlite_conn.commit()

            print(f"Chunks processed: {chunk_count} | Points inserted: {total_points}")

        sqlite_conn.commit()
        # Create indexes to speed up later joins, filtering, and lookups
        write_cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_icao24 ON {OUT_TABLE}(icao24);")
        write_cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_postime ON {OUT_TABLE}(postime);")
        write_cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_icao24_postime ON {OUT_TABLE}(icao24, postime);")
        sqlite_conn.commit()

        print(f"Done. Chunks processed: {chunk_count} | Total points inserted: {total_points}")
        print(f"Wrote {OUT_TABLE} to {DB_PATH}")

    finally:
        sqlite_conn.close()
        trino_conn.close()


if __name__ == "__main__":
    main()
