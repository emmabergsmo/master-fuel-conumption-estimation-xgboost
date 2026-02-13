import sqlite3
from trino_client import get_trino_connection

DB_PATH = "opensky.sqlite"
FLIGHTS_TABLE = "norwegian_flights_2022_with_type"
TRINO_STATE_TABLE = "state_vectors_data4" 
OUT_TABLE = "test_1min"

# Window
TIME_START = 1640991600  # 2022-09-15 00:00:00
TIME_END   = 1641510000  # 2022-10-01 00:00:00

# Performance knobs
FLIGHT_CHUNK_SIZE = 50       # flights per Trino query
COMMIT_EVERY_CHUNKS = 10     # commit every N chunks


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
    squawk TEXT
);
"""

INSERT_SQL = f"""
INSERT OR IGNORE INTO {OUT_TABLE} (
  icao24, estdepartureairport, estarrivalairport, callsign, aircraft_type_icao_code,
  lon, lat, postime, baroaltitude, geoaltitude, velocity, heading, vertrate, squawk
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


# Helpers
def trino_sql_literal(v) -> str:
    """Safe Trino SQL literal (avoids prepared statements)."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).strip().replace("'", "''")
    return f"'{s}'"


def hour_floor(ts: int) -> int:
    """Your dataset's 'hour' is a unix timestamp rounded down to the hour start."""
    return ts - (ts % 3600)


def chunked(seq, n):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# Main
def main():
    sqlite_conn = sqlite3.connect(DB_PATH)
    sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    sqlite_conn.executescript(DDL)

    read_cur = sqlite_conn.cursor()
    write_cur = sqlite_conn.cursor()

    trino_conn = get_trino_connection()
    trino_cur = trino_conn.cursor()

    # Overlap filter: include any flight that overlaps the window
    flights = list(read_cur.execute(
        f"""
        SELECT
          icao24,
          estdepartureairport,
          estarrivalairport,
          callsign,
          aircraft_type_icao,
          firstseen,
          lastseen
        FROM {FLIGHTS_TABLE}
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
            values_rows = []

            for icao24, dep, arr, callsign, ac_type, firstseen, lastseen in flight_chunk:
                firstseen = int(firstseen)
                lastseen = int(lastseen)

                # Clamp flight interval to the window
                start_t = max(firstseen, TIME_START)
                end_t = min(lastseen, TIME_END)
                if start_t >= end_t:
                    continue

                # IMPORTANT: hour is unix timestamp floored to the hour
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
                    ]) + ")"
                )

            if not values_rows:
                continue

            trino_sql = f"""
            WITH flights(icao24, dep, arr, callsign, ac_type, start_t, end_t, start_h, end_h) AS (
              VALUES {",".join(values_rows)}
            )
            SELECT
              f.icao24,
              f.dep AS estdepartureairport,
              f.arr AS estarrivalairport,
              f.callsign,
              f.ac_type AS aircraft_type_icao_code,

              max_by(sv.lon, sv.time)           AS lon,
              max_by(sv.lat, sv.time)           AS lat,
              max_by(sv.time, sv.time)          AS postime,
              max_by(sv.baroaltitude, sv.time)  AS baroaltitude,
              max_by(sv.geoaltitude, sv.time)   AS geoaltitude,
              max_by(sv.velocity, sv.time)      AS velocity,
              max_by(sv.heading, sv.time)       AS heading,
              max_by(sv.vertrate, sv.time)      AS vertrate,
              max_by(sv.squawk, sv.time)        AS squawk
            FROM {TRINO_STATE_TABLE} sv
            JOIN flights f
              ON sv.icao24 = f.icao24
             AND sv.time BETWEEN f.start_t AND f.end_t
             AND sv.hour BETWEEN f.start_h AND f.end_h
            GROUP BY
              f.icao24, f.dep, f.arr, f.callsign, f.ac_type,
              (sv.time - (sv.time % 60))
            ORDER BY f.icao24, postime
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

        # Build indexes AFTER inserts (much faster)
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
