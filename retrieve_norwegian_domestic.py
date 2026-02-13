import json
import sqlite3
from trino_client import get_trino_connection

DB_PATH = "opensky.sqlite"
SOURCE_TABLE = "flights_data4"
DEST_TABLE = "norwegian_domestic_flights_2022"

YEAR_START = 1640995200   # 2022-01-01 00:00:00
YEAR_END = 1672531200     # 2023-01-01 00:00:00

DDL = f"""
DROP TABLE IF EXISTS {DEST_TABLE};

CREATE TABLE {DEST_TABLE} (
    day INTEGER,
    icao24 TEXT,
    callsign TEXT,
    estdepartureairport TEXT,
    estarrivalairport TEXT,
    firstseen INTEGER,
    lastseen INTEGER,
    track ARRAY
);

CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_firstseen ON {DEST_TABLE}(firstseen);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_icao24 ON {DEST_TABLE}(icao24);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_callsign ON {DEST_TABLE}(callsign);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_day ON {DEST_TABLE}(day);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_dep ON {DEST_TABLE}(estdepartureairport);
CREATE INDEX IF NOT EXISTS idx_{DEST_TABLE}_arr ON {DEST_TABLE}(estarrivalairport);
"""

# IMPORTANT: no "?" params -> avoids prepared statements / EXECUTE IMMEDIATE path
# Domestic Norway filter: EN* -> EN*
SELECT_SQL = f"""
SELECT
  day,
  icao24,
  callsign,
  estdepartureairport,
  estarrivalairport,
  firstseen,
  lastseen,
  track
FROM {SOURCE_TABLE}
WHERE firstseen >= {YEAR_START}
  AND firstseen < {YEAR_END}
  AND estdepartureairport IS NOT NULL
  AND estarrivalairport IS NOT NULL
  AND estdepartureairport LIKE 'EN%%'
  AND estarrivalairport LIKE 'EN%%'
  AND (
    TRIM(callsign) LIKE 'NOZ%%'
    OR TRIM(callsign) LIKE 'NSZ%%'
    OR TRIM(callsign) LIKE 'NAX%%'
    OR TRIM(callsign) LIKE 'NRS%%'
  )
"""

INSERT_SQL = f"""
INSERT OR IGNORE INTO {DEST_TABLE} VALUES (?,?,?,?,?,?,?,?)
"""


def to_sqlite_value(v):
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
    trino_conn = get_trino_connection()
    sqlite_conn = sqlite3.connect(DB_PATH)

    sqlite_conn.execute("PRAGMA journal_mode=WAL;")
    sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
    sqlite_conn.executescript(DDL)

    cur = trino_conn.cursor()
    buf = []
    inserted = 0

    try:
        print("Downloading domestic Norwegian flights from flights_data4 for 2022 (EN*->EN*, NOZ/NSZ/NAX/NRS) ...")
        cur.execute(SELECT_SQL)

        while True:
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
            f"Done. Inserted ~{inserted} rows into {DB_PATH} table {DEST_TABLE}.")

    finally:
        sqlite_conn.close()
        trino_conn.close()


if __name__ == "__main__":
    main()
