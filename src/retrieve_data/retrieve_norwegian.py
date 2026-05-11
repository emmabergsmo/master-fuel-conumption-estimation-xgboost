"""Retrieve Norwegian Air Shuttle flights from OpenSky for 2022.

This script queries the OpenSky Trino 'flights_data4' table for flights with
Norwegian-related callsigns and stores the result in a local SQLite table for 
later preprocessing.
"""

import json
import sqlite3
from trino_client import get_trino_connection

DB_PATH = "opensky.sqlite"
SOURCE_TABLE = "flights_data4"
DEST_TABLE = "norwegian_flights_2022"

YEAR_START = 1640995200   # 2022-01-01 00:00:00
YEAR_END = 1672531200     # 2023-01-01 00:00:00

NORWEGIAN_CALLSIGN_PREFIXES = ("NOZ", "NSZ", "NAX", "NRS")


callsign_filter = " OR ".join(
    f"TRIM(callsign) LIKE '{prefix}%%'"
    for prefix in NORWEGIAN_CALLSIGN_PREFIXES
)

DDL = f"""
DROP TABLE IF EXISTS {DEST_TABLE};

CREATE TABLE {DEST_TABLE} (
    day INTEGER,
    icao24 TEXT,
    callsign TEXT,
    estdepartureairport TEXT,
    estarrivalairport TEXT,
    firstseen INTEGER,
    lastseen INTEGER
);

CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_firstseen ON {DEST_TABLE}(firstseen);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_icao24 ON {DEST_TABLE}(icao24);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_callsign ON {DEST_TABLE}(callsign);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_day ON {DEST_TABLE}(day);
"""


SELECT_SQL = f"""
SELECT
  day,
  icao24,
  callsign,
  estdepartureairport,
  estarrivalairport,
  firstseen,
  lastseen
FROM {SOURCE_TABLE}
WHERE firstseen >= {YEAR_START}
  AND firstseen < {YEAR_END}
  AND day >= {YEAR_START}
  AND day < {YEAR_END}
  AND (
    {callsign_filter}
  )
"""

INSERT_SQL = f"""
INSERT OR IGNORE INTO {DEST_TABLE} VALUES (?,?,?,?,?,?,?)
"""


def to_sqlite_value(v):
    """Convert Trino values to SQLite-compatible values."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    return v


def main():
    """Download Norwegian flights from Trino and write them to SQLite."""
    trino_conn = get_trino_connection()
    sqlite_conn = sqlite3.connect(DB_PATH)

    # Improve SQLite write performance for large batch inserts
    sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    sqlite_conn.executescript(DDL)

    cur = trino_conn.cursor()
    buf = []
    inserted = 0

    try:
        print("Downloading Norwegian flights from flights_data4 for 2022 (NOZ*/NSZ*/NAX*/NRS*) ...")
        cur.execute(SELECT_SQL)

        while True:
            # Fetch rows in chunks to avoid loading the full result set into memory
            rows = cur.fetchmany(5000)
            if not rows:
                break

            for r in rows:
                buf.append(tuple(to_sqlite_value(v) for v in r))

            if len(buf) >= 10000:
                sqlite_conn.executemany(INSERT_SQL, buf)
                sqlite_conn.commit()
                inserted += len(buf)
                buf.clear()
                print(f"Inserted {inserted} rows...")

        if buf:
            sqlite_conn.executemany(INSERT_SQL, buf)
            sqlite_conn.commit()
            inserted += len(buf)

        print(
            f"Done. Inserted {inserted} rows into {DB_PATH} table {DEST_TABLE}.")

    finally:
        sqlite_conn.close()
        trino_conn.close()


if __name__ == "__main__":
    main()
