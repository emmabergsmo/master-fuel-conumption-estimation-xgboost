import sqlite3
import pandas as pd

sqlite_db_path = "opensky_updated.sqlite"
csv_output_path = "norwegian_fuel_v2.csv"

conn = sqlite3.connect(sqlite_db_path)

query = "SELECT * FROM adsb_fuel_v2"

chunksize = 100000  # number of rows per chunk

with open(csv_output_path, "w", newline="") as f:
    for i, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=chunksize)):

        print(f"Writing chunk {i}...")

        chunk.to_csv(
            f,
            index=False,
            header=(i == 0)  # header only for first chunk
        )

conn.close()

print("CSV export complete.")