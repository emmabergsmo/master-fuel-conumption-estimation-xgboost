"""Export the ADS-B fuel table from SQLite to CSV.

This script reads a table from the SQLite database and writes it to a 
CSV file in chunks. Chunked export avoids loading the full table into
memory at once.
"""

import sqlite3
import pandas as pd

DB_PATH = "opensky_updated.sqlite"
TABLE = "adsb_fuel_v2"
CSV_OUTPUT_PATH = "norwegian_fuel_v2.csv"

chunksize = 100000 


def main():
    """Export a SQLite table to CSV in chunks."""
    conn = sqlite3.connect(DB_PATH)

    query = f"SELECT * FROM {TABLE}"

    with open(CSV_OUTPUT_PATH, "w", newline="") as f:
        for i, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=chunksize)):

            print(f"Writing chunk {i}...")

            chunk.to_csv(
                f,
                index=False,
                header=(i == 0)
            )

    conn.close()
    print("CSV export complete.")

if __name__ == "__main__":
    main()