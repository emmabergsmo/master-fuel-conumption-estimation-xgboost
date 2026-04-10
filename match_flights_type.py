import re
import sqlite3
import pandas as pd
from dateutil import parser
import airportsdata

DB_PATH = "opensky.sqlite"
OPENSKY_TABLE = "norwegian_domestic_flights_2022"  # or norwegian_flights_2022
CSV_PATH = "norwegian_data.csv"
OUT_TABLE = "norwegian_flights_2022_with_type"

TIME_WINDOW_SECONDS = int(1.5 * 3600)  # ±1.5 hours

# Prefixes to strip before extracting identifiers
CSV_PREFIXES = ("DY", "D8")
CALLSIGN_PREFIXES = ("NOZ", "NAX", "NSZ", "NRS")

airports = airportsdata.load()

# Build IATA -> ICAO lookup
IATA_TO_ICAO = {}
for icao, info in airports.items():
    i = info.get("iata")
    if i:
        IATA_TO_ICAO[i.upper()] = icao


def strip_prefixes(s: str, prefixes: tuple[str, ...]) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):].strip()
    return s


def tail_has_letters_after_prefix(s: str, prefixes: tuple[str, ...], tail_len: int = 5) -> bool:
    """
    After removing prefix, check whether the last 4-5 characters contain any letters.
    If yes -> airport-based matching.
    """
    s2 = strip_prefixes(s, prefixes)
    if not s2:
        return True
    tail = s2[-tail_len:]
    # If there are any A-Z letters in the tail, treat as "has letters"
    return bool(re.search(r"[A-Z]", tail))


def extract_flight_number_digits(s: str, prefixes: tuple[str, ...], max_len: int = 4) -> str | None:
    """
    Strip prefix, then extract trailing digits (3-4 digits).
    Returns None if not 3-4 trailing digits.
    """
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
    if s is None:
        return None
    return str(s).strip()


def parse_timestamp_to_epoch(ts_str):
    # Assumption: UTC for naive timestamps
    dt = parser.parse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)  # UTC tzinfo
    return int(dt.timestamp())


def iata_to_icao(iata):
    if iata is None or pd.isna(iata):
        return None
    iata = str(iata).strip().upper()
    if not iata:
        return None
    return IATA_TO_ICAO.get(iata)


def main():
    con = sqlite3.connect(DB_PATH)

    # ---- Load OpenSky data from SQLite ----
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
        FROM {OPENSKY_TABLE}
        WHERE callsign IS NOT NULL
        """,
        con,
    )

    opensky_df["callsign_clean"] = opensky_df["callsign"].map(clean_callsign)
    opensky_df["callsign_num"] = opensky_df["callsign_clean"].map(
        lambda s: extract_flight_number_digits(s, CALLSIGN_PREFIXES)
    )
    opensky_df["callsign_has_letters_tail"] = opensky_df["callsign_clean"].map(
        lambda s: tail_has_letters_after_prefix(s, CALLSIGN_PREFIXES, tail_len=5)
    )

    opensky_df = opensky_df.sort_values("firstseen").reset_index(drop=True)

    # ---- Load CSV ----
    csv_df = pd.read_csv(CSV_PATH, sep=";")

    CSV_COL_FLTNUM = "Displayed flight number"
    CSV_COL_TIME = "Actual departure date"
    CSV_COL_TYPE = "Aircraft type ICAO code"

    # Airport columns (IATA) 
    CSV_COL_ARR_IATA = "Actual arrival airport IATA"
    CSV_COL_DEP_IATA = "Actual departure airport IATA"

    csv_df["flightnum"] = csv_df[CSV_COL_FLTNUM].astype(str).str.strip()
    csv_df["flightnum_num"] = csv_df["flightnum"].map(
        lambda s: extract_flight_number_digits(s, CSV_PREFIXES)
    )
    csv_df["flightnum_has_letters_tail"] = csv_df["flightnum"].map(
        lambda s: tail_has_letters_after_prefix(s, CSV_PREFIXES, tail_len=5)
    )

    csv_df["aircraft_type_icao"] = csv_df[CSV_COL_TYPE].astype(str).str.strip()
    csv_df["event_time_epoch"] = csv_df[CSV_COL_TIME].map(parse_timestamp_to_epoch)

    csv_df["dep_iata"] = csv_df[CSV_COL_DEP_IATA].astype(str).str.strip().str.upper()
    csv_df["arr_iata"] = csv_df[CSV_COL_ARR_IATA].astype(str).str.strip().str.upper()
    csv_df["dep_icao"] = csv_df["dep_iata"].map(iata_to_icao)
    csv_df["arr_icao"] = csv_df["arr_iata"].map(iata_to_icao)

    # ---- Matching ----
    matches = []
    unmatched = 0

    for _, row in csv_df.iterrows():
        t0 = int(row["event_time_epoch"])
        flightnum = row["flightnum"]
        ac_type = row["aircraft_type_icao"]

        dep_icao = row["dep_icao"]
        arr_icao = row["arr_icao"]
        num = row["flightnum_num"]  


        # 1) time window candidates
        cand = opensky_df[
            (opensky_df["firstseen"] >= t0 - TIME_WINDOW_SECONDS)
            & (opensky_df["firstseen"] <= t0 + TIME_WINDOW_SECONDS)
        ].copy()

        if cand.empty:
            unmatched += 1
            continue

        # HARD FILTER on airports
        if dep_icao:
            cand = cand[cand["estdepartureairport"] == dep_icao].copy()

        if arr_icao:
            cand = cand[cand["estarrivalairport"] == arr_icao].copy()

        if cand.empty:
            unmatched += 1
            continue

        # Calculate time diff
        cand["dt"] = (cand["firstseen"] - t0).abs()

       
        if num:
            cand["flightnum_match"] = (cand["callsign_num"] == num).astype(int)

            cand = cand.sort_values(
                ["flightnum_match", "dt"],
                ascending=[False, True]
            )

            if cand.iloc[0]["flightnum_match"] == 1:
                match_rule = "airports_required_digits_preferred"
            else:
                match_rule = "airports_required_time_only"
        else:
            cand = cand.sort_values("dt")
            match_rule = "airports_required_time_only"

        best = cand.iloc[0]

        matches.append(
            {
                # OpenSky
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
                # Diagnostics
                "time_diff_seconds": int(abs(int(best["firstseen"]) - t0)),
                "match_rule": match_rule,
            }
        )

    match_df = pd.DataFrame(matches)

    print(f"Matched rows: {len(match_df)}")
    print(f"Unmatched rows: {unmatched}")

    # ---- Write output table to SQLite ----
    match_df.to_sql(OUT_TABLE, con, if_exists="replace", index=False)

    cur = con.cursor()
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_icao24 ON {OUT_TABLE}(icao24);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_callsign ON {OUT_TABLE}(callsign);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_firstseen ON {OUT_TABLE}(firstseen);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_flightnum ON {OUT_TABLE}(displayed_flight_number);")
    con.commit()
    con.close()

    print(f"Wrote {OUT_TABLE} to {DB_PATH}")


if __name__ == "__main__":
    main()
