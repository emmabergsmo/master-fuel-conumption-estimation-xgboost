import sqlite3

DB_PATH = "opensky.sqlite"
TABLE = "norwegian_flights_2022_points_1min"  # <-- change this if needed
GAP_SECONDS = 3600  # <-- change this if needed (e.g. 45*60)

def ensure_column(conn):
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({TABLE})").fetchall()]
    if "flight_id" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN flight_id TEXT;")
        conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    ensure_column(conn)
    cur = conn.cursor()

    # This:
    # 1) orders points per (icao24,callsign)
    # 2) flags a new flight when gap > GAP_SECONDS (or first row)
    # 3) creates a running flight_seq number per (icao24,callsign)
    # 4) finds first_postime per (icao24,callsign,flight_seq)
    # 5) updates every row with: icao24-callsign-mm-dd-hh-mm using first_postime
    sql = f"""
    WITH ordered AS (
      SELECT
        rowid AS rid,
        icao24,
        callsign,
        postime,
        LAG(postime) OVER (
          PARTITION BY icao24, callsign
          ORDER BY postime
        ) AS prev_postime
      FROM {TABLE}
      WHERE icao24 IS NOT NULL
        AND callsign IS NOT NULL
        AND postime IS NOT NULL
    ),
    flagged AS (
      SELECT
        rid,
        icao24,
        callsign,
        postime,
        CASE
          WHEN prev_postime IS NULL THEN 1
          WHEN (postime - prev_postime) > {GAP_SECONDS} THEN 1
          ELSE 0
        END AS is_new_flight
      FROM ordered
    ),
    grouped AS (
      SELECT
        rid,
        icao24,
        callsign,
        postime,
        SUM(is_new_flight) OVER (
          PARTITION BY icao24, callsign
          ORDER BY postime
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS flight_seq
      FROM flagged
    ),
    firsts AS (
      SELECT
        icao24,
        callsign,
        flight_seq,
        MIN(postime) AS first_postime
      FROM grouped
      GROUP BY icao24, callsign, flight_seq
    ),
    to_update AS (
      SELECT
        g.rid,
        g.icao24,
        g.callsign,
        f.first_postime
      FROM grouped g
      JOIN firsts f
        ON g.icao24 = f.icao24
       AND g.callsign = f.callsign
       AND g.flight_seq = f.flight_seq
    )
    UPDATE {TABLE}
    SET flight_id =
      to_update.icao24 || '-' || to_update.callsign || '-' ||
      strftime('%m-%d-%H-%M', to_update.first_postime, 'unixepoch')
    FROM to_update
    WHERE {TABLE}.rowid = to_update.rid;
    """

    cur.executescript(sql)
    conn.commit()

    # Optional: index for speed later
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_flight_id ON {TABLE}(flight_id);")
    conn.commit()

    conn.close()
    print("Done: flight_id assigned per flight segment.")

if __name__ == "__main__":
    main()
