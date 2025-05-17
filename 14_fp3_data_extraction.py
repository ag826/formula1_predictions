import os
import fastf1
import pandas as pd
from datetime import datetime

# Create the cache directory if it doesn't exist
os.makedirs("f1_cache", exist_ok=True)
os.makedirs("RAW_DATA/FP3_DATA", exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Define session types
session_types = ["FP3"]

# Initialize containers
all_results = []
all_weather = []
all_track_status = []
all_laps = []
all_corners = []
all_event_metadata = []

# ---------------------------------------------------
# Sourcing Race Data
# ---------------------------------------------------

# Year range
for year in range(2023, 2026):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    race_numbers = schedule["RoundNumber"].dropna().astype(int)

    for rnd in race_numbers:
        for stype in session_types:
            try:
                session = fastf1.get_session(year, rnd, stype)
                session.load()

                # 1. session.weather_data
                if session.weather_data is not None:
                    df = session.weather_data.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_weather.append(df)

                # 2. session.track_status
                if session.track_status is not None:
                    df = session.track_status.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_track_status.append(df)

                # 3. session.laps
                laps = session.laps
                if laps is not None and not laps.empty:
                    df = laps.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_laps.append(df)

            except Exception as e:
                print(f"⚠️ Skipped {year} Round {rnd} Session {stype}: {e}")

# Save to CSV files
if all_weather:
    pd.concat(all_weather).to_csv("RAW_DATA/FP3_DATA/FP3_weather_data.csv", index=False)
if all_track_status:
    pd.concat(all_track_status).to_csv("RAW_DATA/FP3_DATA/FP3_track_status.csv", index=False)
if all_laps:
    pd.concat(all_laps).to_csv("RAW_DATA/FP3_DATA/FP3_lap_data.csv", index=False)

print("✅ All requested FP3 session data saved to CSV.") 