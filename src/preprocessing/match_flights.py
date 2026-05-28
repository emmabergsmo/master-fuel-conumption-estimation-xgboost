"""Match OpenSky flights with Norwegian aircraft type and fuel records.

This script matches domestic Norwegian OpenSky flights to Norwegian CSV rows
using departure time, airport pair, and flight number. In addition to aircraft
type, it copies recorded fuel values for block, taxi, takeoff, climb, cruise,
descent, landing, and great-circle distance into the output SQLite table.
"""

import re
import sqlite3
import pandas as pd
from dateutil import parser
import airportsdata

DB_PATH = "data.sqlite"
IN_TABLE = "norwegian_domestic_flights_2022"  
CSV_PATH = "norwegian_data.csv"
OUT_TABLE = "norwegian_flights_2022_matched"

TIME_WINDOW_SECONDS = int(1.5 * 3600)  

FLIGHTNUM_PREFIXES = ("DY", "D8")
CALLSIGN_PREFIXES = ("NOZ", "NAX", "NSZ", "NRS")


airports = airportsdata.load()

# Build IATA-to-ICAO mapping for matching CSV airport codes with OpenSky airports
IATA_TO_ICAO = {}
duplicates = {}

for icao, info in airports.items():
    iata = (info.get("iata") or "").strip().upper()
    if not iata:
        continue

    if iata in IATA_TO_ICAO and IATA_TO_ICAO[iata] != icao:
        duplicates.setdefault(iata, []).append(icao)
    else:
        IATA_TO_ICAO[iata] = icao


def strip_prefixes(s: str, prefixes: tuple[str, ...]) -> str:
    """Remove known airline prefixes from a flight number or callsign."""
    if s is None:
        return ""
    s = str(s).strip().upper()
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):].strip()
    return s


def clean_numeric(val):
    """Convert CSV numeric values with comma decimals to floats."""
    if pd.isna(val):
        return None
    s = str(val).strip().replace(" ", "").replace(",", ".")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def tail_has_letters_after_prefix(s: str, prefixes: tuple[str, ...], tail_len: int = 5) -> bool:
    """Return whether the suffix after the airline prefix contains letters."""
    s2 = strip_prefixes(s, prefixes)
    if not s2:
        return True
    tail = s2[-tail_len:]
    return bool(re.search(r"[A-Z]", tail))


def extract_flight_number_digits(s: str, prefixes: tuple[str, ...], max_len: int = 4) -> str | None:
    """Extract the trailing 3- or 4-digit flight number after removing a prefix."""
    s2 = strip_prefixes(s, prefixes)
    m = re.search(r"(\d+)$", s2)
    if not m:
        return None
    digits = m.group(1)[-max_len:]
    if len(digits) == 4:
        return digits
    if len(digits) == 3:
        return digits
    return None


def clean_callsign(s):
    """Strip whitespace from a callsign and preserve missing values."""
    if s is None:
        return None
    return str(s).strip()


def parse_timestamp_to_epoch(ts_str):
    """Parse a timestamp string and return Unix epoch seconds."""
    dt = parser.parse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
    return int(dt.timestamp())


def iata_to_icao(iata):
    """Convert an IATA airport code to an ICAO airport code."""
    if iata is None or pd.isna(iata):
        return None
    iata = str(iata).strip().upper()
    if not iata:
        return None
    return IATA_TO_ICAO.get(iata)


def main():
    """Match Norwegian CSV rows to OpenSky flights and write fuel-enriched output."""
    con = sqlite3.connect(DB_PATH)

    opensky_df = pd.read_sql_query(
        f"""
        SELECT
          day,
          icao24,
          callsign,
          estdepartureairport,
          estarrivalairport,
          firstseen,
          lastseen
        FROM {IN_TABLE}
        WHERE callsign IS NOT NULL
        """,
        con,
    )

    # Normalize OpenSky callsigns so they can be compared with CSV flight numbers
    opensky_df["callsign_clean"] = opensky_df["callsign"].map(clean_callsign)
    opensky_df["callsign_num"] = opensky_df["callsign_clean"].map(
        lambda s: extract_flight_number_digits(s, CALLSIGN_PREFIXES)
    )
    opensky_df["callsign_has_letters_tail"] = opensky_df["callsign_clean"].map(
        lambda s: tail_has_letters_after_prefix(s, CALLSIGN_PREFIXES, tail_len=5)
    )

    opensky_df = opensky_df.sort_values("firstseen").reset_index(drop=True)

    csv_df = pd.read_csv(CSV_PATH, sep=";")

    CSV_COL_FLTNUM = "Displayed flight number"
    CSV_COL_TIME = "Actual departure date"
    CSV_COL_TYPE = "Aircraft type ICAO code"

    CSV_COL_DEP_IATA = "Actual departure airport IATA"
    CSV_COL_ARR_IATA = "Actual arrival airport IATA"
    CSV_COL_BLOCK_FUEL = "Block fuel [AVG]"
    CSV_COL_TAXI_OUT_FUEL = "TAXI-OUT fuel burn [AVG]"
    CSV_COL_TAKEOFF_FUEL = "TAKEOFF fuel burn [AVG]"
    CSV_COL_CLIMB_FUEL = "CLIMB fuel burn [AVG]"
    CSV_COL_CRUISE_FUEL = "CRUISE fuel burn [AVG]"
    CSV_COL_DESCENT_FUEL = "DESCENT fuel burn [AVG]"
    CSV_COL_LANDING_FUEL = "LANDING fuel burn [AVG]"
    CSV_COL_TAXI_IN_FUEL = "TAXI-IN fuel burn [AVG]"
    CSV_COL_GCD = "Flight great circle distance [AVG]"

    csv_df["flightnum"] = csv_df[CSV_COL_FLTNUM].astype(str).str.strip()
    csv_df["flightnum_num"] = csv_df["flightnum"].map(
        lambda s: extract_flight_number_digits(s, FLIGHTNUM_PREFIXES)
    )
    csv_df["flightnum_has_letters_tail"] = csv_df["flightnum"].map(
        lambda s: tail_has_letters_after_prefix(s, FLIGHTNUM_PREFIXES, tail_len=5)
    )

    csv_df["aircraft_type_icao"] = csv_df[CSV_COL_TYPE].astype(str).str.strip()
    csv_df["event_time_epoch"] = csv_df[CSV_COL_TIME].map(parse_timestamp_to_epoch)

    csv_df["dep_iata"] = csv_df[CSV_COL_DEP_IATA].astype(str).str.strip().str.upper()
    csv_df["arr_iata"] = csv_df[CSV_COL_ARR_IATA].astype(str).str.strip().str.upper()
    csv_df["dep_icao"] = csv_df["dep_iata"].map(iata_to_icao)
    csv_df["arr_icao"] = csv_df["arr_iata"].map(iata_to_icao)

    fuel_columns = [
    CSV_COL_BLOCK_FUEL,
    CSV_COL_TAXI_OUT_FUEL,
    CSV_COL_TAKEOFF_FUEL,
    CSV_COL_CLIMB_FUEL,
    CSV_COL_CRUISE_FUEL,
    CSV_COL_DESCENT_FUEL,
    CSV_COL_LANDING_FUEL,
    CSV_COL_TAXI_IN_FUEL,
    CSV_COL_GCD,
]

    for col in fuel_columns:
        csv_df[col] = csv_df[col].map(clean_numeric)


    matches = []
    unmatched = 0

    for _, row in csv_df.iterrows():
        t0 = int(row["event_time_epoch"])
        flightnum = row["flightnum"]
        ac_type = row["aircraft_type_icao"]

        dep_icao = row["dep_icao"]
        arr_icao = row["arr_icao"]

        best = None
        match_rule = None
        num = row["flightnum_num"]

        # Keep only OpenSky candidates close in time to the CSV departure time
        cand = opensky_df[
            (opensky_df["firstseen"] >= t0 - TIME_WINDOW_SECONDS)
            & (opensky_df["firstseen"] <= t0 + TIME_WINDOW_SECONDS)
        ].copy()

        if cand.empty:
            unmatched += 1
            continue

        # Require matching airport pair when available, then prefer matching flight number
        if dep_icao and arr_icao:
            cand = cand[
                (cand["estdepartureairport"] == dep_icao) &
                (cand["estarrivalairport"] == arr_icao)
            ].copy()
        elif dep_icao:
            cand = cand[cand["estdepartureairport"] == dep_icao].copy()
        elif arr_icao:
            cand = cand[cand["estarrivalairport"] == arr_icao].copy()

        if cand.empty:
            unmatched += 1
            continue

        cand["dt"] = (cand["firstseen"] - t0).abs()

        if num:
            cand["flightnum_match"] = (cand["callsign_num"] == num).astype(int)
            cand = cand.sort_values(["flightnum_match", "dt"], ascending=[False, True])

            if cand.iloc[0]["flightnum_match"] == 1:
                match_rule = "airports_required_digits_preferred"
            else:
                match_rule = "airports_required_time_only"
        else:
            cand = cand.sort_values("dt")
            match_rule = "airports_required_time_only"

        best = cand.iloc[0]
                    
        if best is None:
            unmatched += 1
            continue

        # Store matched OpenSky metadata together with aircraft type and fuel values
        matches.append(
            {
                "day": int(best["day"]) if pd.notna(best["day"]) else None,
                "csv_timestamp": row[CSV_COL_TIME],
                "icao24": best["icao24"],
                "callsign": best["callsign_clean"],
                "displayed_flight_number": flightnum,
                "estdepartureairport": best["estdepartureairport"],
                "estarrivalairport": best["estarrivalairport"],
                "csv_dep_iata": row["dep_iata"],
                "csv_arr_iata": row["arr_iata"],
                "firstseen": int(best["firstseen"]),
                "lastseen": int(best["lastseen"]) if pd.notna(best["lastseen"]) else None,
                "aircraft_type_icao": ac_type,
                ""

                "block_fuel": row[CSV_COL_BLOCK_FUEL],
                "taxi_out_fuel": row[CSV_COL_TAXI_OUT_FUEL],
                "takeoff_fuel": row[CSV_COL_TAKEOFF_FUEL],
                "climb_fuel": row[CSV_COL_CLIMB_FUEL],
                "cruise_fuel": row[CSV_COL_CRUISE_FUEL],
                "descent_fuel": row[CSV_COL_DESCENT_FUEL],
                "landing_fuel": row[CSV_COL_LANDING_FUEL],
                "taxi_in_fuel": row[CSV_COL_TAXI_IN_FUEL],
                "great_circle_distance": row[CSV_COL_GCD],

                "time_diff_seconds": int(abs(int(best["firstseen"]) - t0)),
                "match_rule": match_rule,
            }
        )

    match_df = pd.DataFrame(matches)

    print(f"Matched rows: {len(match_df)}")
    print(f"Unmatched rows: {unmatched}")

    match_df.to_sql(OUT_TABLE, con, if_exists="replace", index=False)

    cur = con.cursor()
    # Create indexes to speed up later joins, filtering, and lookups
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_icao24 ON {OUT_TABLE}(icao24);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_callsign ON {OUT_TABLE}(callsign);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_firstseen ON {OUT_TABLE}(firstseen);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_flightnum ON {OUT_TABLE}(displayed_flight_number);")
    con.commit()
    con.close()

    print(f"Wrote {OUT_TABLE} to {DB_PATH}")


if __name__ == "__main__":
    main()
