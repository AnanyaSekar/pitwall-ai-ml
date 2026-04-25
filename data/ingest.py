import fastf1
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os

fastf1.Cache.enable_cache("data/cache")

cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def fetch_and_save(year, gp_name):
    print(f"Fetching {gp_name} {year}...")
    session = fastf1.get_session(year, gp_name, "R")
    session.load()
    laps = session.laps[["Driver","LapNumber","LapTime",
                          "Compound","TyreLife","Stint"]].dropna()
    batch = db.batch()
    count = 0
    for _, row in laps.iterrows():
        doc_ref = db.collection("laps").document()
        batch.set(doc_ref, {
            "driver": row.Driver,
            "lap": int(row.LapNumber),
            "lap_time_s": row.LapTime.total_seconds(),
            "compound": row.Compound,
            "tyre_life": int(row.TyreLife),
            "stint": int(row.Stint),
            "year": year,
            "gp": gp_name
        })
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    print(f"Saved {count} laps for {gp_name} {year} to Firestore!")

if __name__ == "__main__":
    mkdir_cmd = "mkdir -p data/cache"
    os.system(mkdir_cmd)
    races = [
        (2023, "Bahrain"),
        (2023, "Saudi Arabia"),
        (2023, "Australia"),
        (2023, "Monaco"),
        (2024, "Bahrain"),
        (2024, "Monaco"),
    ]
    for year, gp in races:
        try:
            fetch_and_save(year, gp)
        except Exception as e:
            print(f"Error {gp} {year}: {e}")
