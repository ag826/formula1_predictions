import os
import fastf1

# Create the directory if it doesn't exist
os.makedirs("f1_cache", exist_ok=True)

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()

session_2024.laps.columns