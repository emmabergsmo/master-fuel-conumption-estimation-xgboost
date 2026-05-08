import sqlite3
import pandas as pd

sqlite_db_path = "opensky_updated.sqlite"
h5_output_path = "ready_v2_phase.h5"

conn = sqlite3.connect(sqlite_db_path)

# Finn alle unike hex-koder
hex_codes = pd.read_sql_query(
    "SELECT DISTINCT hex FROM aviteam_ready_phase;",
    conn
)["hex"].tolist()

print(f"Found {len(hex_codes)} hex-codes")

with pd.HDFStore(h5_output_path, mode='w', complevel=9, complib='blosc') as store:

    for hex_code in hex_codes:
        print(f"Lagrer {hex_code}...")

        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM aviteam_ready_phase
            WHERE hex = '{hex_code}';
            """,
            conn
        )

        store.put(
            f"ready_{hex_code}",
            df,
            format="table"
        )

conn.close()

print("ready_v2_phase.h5 created with all keys.")
