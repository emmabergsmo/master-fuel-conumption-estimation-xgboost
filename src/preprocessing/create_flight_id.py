"""Create unique flight identifiers for ADS-B trajectory segments.

This script adds a 'flight_id' column to an ADS-B SQLite table and assigns
one identifier per continuous flight segment. Segments are separated by large
time gaps within each '(icao24, callsign)' group. The generated identifier
uses the aircraft ICAO24, callsign, and first timestamp of the segment.
"""

import sqlite3

DB_PATH = "opensky.sqlite"
TABLE = "adsb_fuel_v2"  

GAP_SECONDS = 3600  


def ensure_column(conn):
    """Add the 'flight_id' column to the ADS-B table if it does not exist."""
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({TABLE})").fetchall()]
    if "flight_id" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN flight_id TEXT;")
        conn.commit()

def main():
    """Assign flight IDs to continuous ADS-B segments and index the result."""
    conn = sqlite3.connect(DB_PATH)

    # Improve SQLite write performance for large batch inserts
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    ensure_column(conn)
    cur = conn.cursor()

    # 1) Order ADS-B points within each (icao24, callsign) group
    # 2) Mark a new flight when the time gap exceeds GAP_SECONDS
    # 3) Create a cumulative flight sequence number per aircraft/callsign
    # 4) Find the first timestamp for each detected flight segment
    # 5) Generate and assign a unique flight_id for every row

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
    # Create index to speed up later joins, filtering, and lookups
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_flight_id ON {TABLE}(flight_id);")
    conn.commit()

    conn.close()
    print("Done: flight_id assigned per flight segment.")

if __name__ == "__main__":
    main()
