
import os
import fastf1
import pandas as pd
from datetime import datetime

# Create the cache directory if it doesn't exist
os.makedirs("f1_cache", exist_ok=True)
os.makedirs("RAW_DATA/RACE_DATA", exist_ok=True)  # Ensure output folder exists

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Define session types
session_types = ["R"]  # You can add more types: ["FP1", "FP2", "FP3", "Q", "SQ", "SS", "R"]

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

                # 1. session.results
                if session.results is not None:
                    df = session.results.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_results.append(df)

                # 2. session.weather_data
                if session.weather_data is not None:
                    df = session.weather_data.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_weather.append(df)

                # 3. session.track_status
                if session.track_status is not None:
                    df = session.track_status.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_track_status.append(df)

                # 4. session.laps
                laps = session.laps
                if laps is not None and not laps.empty:
                    df = laps.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_laps.append(df)

                # 5. session.get_circuit_info().corners
                corners = session.get_circuit_info().corners
                if corners is not None:
                    df = corners.copy()
                    df["RACEYEAR"] = year
                    df["RACENUMBER"] = rnd
                    df["SESSIONTYPE"] = stype
                    all_corners.append(df)

                # 6. session.event metadata
                try:
                    event_data = pd.DataFrame([dict(session.event)])  # Convert to dict first, then wrap in a list for DataFrame
                    event_data["RACEYEAR"] = year
                    event_data["RACENUMBER"] = rnd
                    event_data["SESSIONTYPE"] = stype
                    all_event_metadata.append(event_data)
                except Exception as e:
                    print(f"⚠️ Failed to extract event metadata for {year} Round {rnd} Session {stype}: {e}")

            except Exception as e:
                print(f"⚠️ Skipped {year} Round {rnd} Session {stype}: {e}")

# Save to CSV files
if all_results:
    pd.concat(all_results).to_csv("RAW_DATA/RACE_DATA/R_race_results.csv", index=False)
if all_weather:
    pd.concat(all_weather).to_csv("RAW_DATA/RACE_DATA/R_weather_data.csv", index=False)
if all_track_status:
    pd.concat(all_track_status).to_csv("RAW_DATA/RACE_DATA/R_track_status.csv", index=False)
if all_laps:
    pd.concat(all_laps).to_csv("RAW_DATA/RACE_DATA/R_lap_data.csv", index=False)
if all_corners:
    pd.concat(all_corners).to_csv("RAW_DATA/RACE_DATA/R_track_structure.csv", index=False)
if all_event_metadata:
    pd.concat(all_event_metadata).to_csv("RAW_DATA/All_session_event_data.csv", index=False)

print("✅ All requested session data saved to CSV.")