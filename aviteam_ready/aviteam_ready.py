import sqlite3

db_path = "../opensky_updated.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS aviteam_ready;")

cursor.execute("""
    CREATE TABLE aviteam_ready_phase AS
    SELECT 
        hex,
        departure AS port_o,
        arrival AS port_d,
        callsign AS flight,
        t,
        lon,
        lat,
        postime,
        alt_baro,
        alt_geom,
        gs,
        track,
        baro_rate,
        squawk,
        flight_id,
        phase_id,
        phase_name,
        FALSE AS ground,
        FALSE AS completed,
        FALSE AS completed_adv,
        NULL as r
    FROM adsb_fuel_v2;
""")

conn.commit()
conn.close()

print("Tabellen opprettet.")
