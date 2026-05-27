"""Export AviTEAM-ready flight data from SQLite to HDF5.

This script reads the AviTEAM-ready SQLite table and writes one HDF5 dataset
per aircraft hex code. The resulting HDF5 file can be used as input for
AviTEAM or downstream AviTEAM comparison workflows.
"""

import sqlite3
import pandas as pd

DB_PATH = "opensky_updated.sqlite"
TABLE = "aviteam_ready_phase"
H5_OUTPUT_PATH = "ready_v2_phase.h5"


def main():
    """Export AviTEAM-ready flight data grouped by aircraft hex code."""
    conn = sqlite3.connect(DB_PATH)

    hex_codes = pd.read_sql_query(
        f"SELECT DISTINCT hex FROM {TABLE};",
        conn
    )["hex"].tolist()

    print(f"Found {len(hex_codes)} hex-codes")

    with pd.HDFStore(
        H5_OUTPUT_PATH, 
        mode='w', 
        complevel=9, 
        complib='blosc'
    ) as store:

        for hex_code in hex_codes:
            print(f"Saving {hex_code}...")

            df = pd.read_sql_query(
                f"""
                SELECT *
                FROM {TABLE}
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
    print("H5 export complete.")

if __name__ == "__main__":
    main()
