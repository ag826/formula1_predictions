import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil import parser

##################################################################################################################
# WEATHER DATA
##################################################################################################################

full_weather_data=pd.read_csv("FP1_DATA/FP1_weather_data.csv")

agg_weather_data = full_weather_data.groupby(['RACEYEAR', 'RACENUMBER']).agg({
    'AirTemp': ['mean', 'min', 'max'],
    'Humidity': ['mean', 'min', 'max'],
    'Pressure': ['mean', 'min', 'max'],
    'Rainfall': ['max'],  # 1 if rainfall occurred at any point, else 0
    'TrackTemp': ['mean', 'min', 'max'],
    'WindDirection': ['mean'],
    'WindSpeed': ['mean', 'max'],
    'SESSIONTYPE': 'first'  # just one value per group
}).reset_index()

agg_weather_data.columns = [
    col[0] if col[0] in ['RACEYEAR', 'RACENUMBER'] else 'FP1_' + '_'.join(col).strip('_')
    for col in agg_weather_data.columns
]

agg_weather_data = agg_weather_data.drop("FP1_SESSIONTYPE_first", axis=1)

final_fp1_data=agg_weather_data

##################################################################################################################
# DRIVING CHARACTERISTICS
##################################################################################################################

# ---------------------------------------------------------------------------------------------------------------- Avg sector 1/2/3 speed

full_lap_data=pd.read_csv("FP1_DATA/FP1_lap_data.csv")

full_lap_data["Sector1Time"] = pd.to_timedelta(full_lap_data["Sector1Time"], errors='coerce')
full_lap_data["Sector1Time"] = full_lap_data["Sector1Time"].dt.total_seconds() * 1000

full_lap_data["Sector2Time"] = pd.to_timedelta(full_lap_data["Sector2Time"], errors='coerce')
full_lap_data["Sector2Time"] = full_lap_data["Sector2Time"].dt.total_seconds() * 1000

full_lap_data["Sector3Time"] = pd.to_timedelta(full_lap_data["Sector3Time"], errors='coerce')
full_lap_data["Sector3Time"] = full_lap_data["Sector3Time"].dt.total_seconds() * 1000

avg_sector_times = (
    full_lap_data
    .groupby(['RACEYEAR', 'RACENUMBER', 'Driver'], as_index=False)
    .agg(
        AvgSector1Time_ms=('Sector1Time', 'mean'),
        AvgSector2Time_ms=('Sector2Time', 'mean'),
        AvgSector3Time_ms=('Sector3Time', 'mean')
    )
) 

# ---------------------------------------------------------------------------------------------------------------- Avg pit time per race and number of pitstops

full_lap_data["PitOutTime"] = pd.to_timedelta(full_lap_data["PitOutTime"], errors='coerce')
full_lap_data["PitOutTime"] = full_lap_data["PitOutTime"].dt.total_seconds() * 1000

full_lap_data["PitInTime"] = pd.to_timedelta(full_lap_data["PitInTime"], errors='coerce')
full_lap_data["PitInTime"] = full_lap_data["PitInTime"].dt.total_seconds() * 1000

full_lap_data["LapTime"] = pd.to_timedelta(full_lap_data["LapTime"], errors='coerce')
full_lap_data["LapTime"] = full_lap_data["LapTime"].dt.total_seconds() * 1000

full_lap_data = full_lap_data.sort_values(['RACEYEAR', 'RACENUMBER', 'Driver', 'Stint', 'LapNumber'])

# Get last lap of each stint (contains PitInTime)
last_laps = full_lap_data.groupby(['RACEYEAR', 'RACENUMBER', 'Driver', 'Stint'], as_index=False).last()

# Get first lap of next stint (contains PitOutTime)
first_laps = full_lap_data.groupby(['RACEYEAR', 'RACENUMBER', 'Driver', 'Stint'], as_index=False).first()

# Shift first_laps to align next stint's PitOutTime with current stint's PitInTime
first_laps_shifted = first_laps.copy()
first_laps_shifted['Stint'] = first_laps_shifted['Stint'] - 1
first_laps_shifted = first_laps_shifted.rename(columns={'PitOutTime': 'NextStint_PitOutTime'})

# Merge PitInTime from last_laps with PitOutTime from next stint
pit_merged = pd.merge(
    last_laps,
    first_laps_shifted[['RACEYEAR', 'RACENUMBER', 'Driver', 'Stint', 'NextStint_PitOutTime']],
    on=['RACEYEAR', 'RACENUMBER', 'Driver', 'Stint'],
    how='inner'
)

# Calculate pit duration in milliseconds
pit_merged['PitDuration_ms'] = pit_merged['NextStint_PitOutTime']- pit_merged['PitInTime']

# Final output
pit_times_aligned = pit_merged[[
    'RACEYEAR', 'RACENUMBER', 'Driver', 'Stint', 'NextStint_PitOutTime', 'PitInTime', 'PitDuration_ms'
]]

pit_summary = (
    pit_times_aligned
    .groupby(['RACEYEAR', 'RACENUMBER', 'Driver'], as_index=False)
    .agg(
        AvgPitStopDuration_ms=('PitDuration_ms', 'mean'),
        TotalPitStops=('PitDuration_ms', 'count')
    )
)

final_fp1_data=final_fp1_data.merge(pit_summary, left_on=['RACEYEAR', 'RACENUMBER'], right_on=['RACEYEAR', 'RACENUMBER'], how="right" )


# if driver does not pit at all (DNS, DNF) then there will be no values for pit, so row mismatch with sector data


# ---------------------------------------------------------------------------------------------------------------- Avg tyre life SML / Avg speed on SML tyres

# Step 1: Group and aggregate tyre data by Race-Year, Race-Number, Driver, and Compound
tyre_stats = (
    full_lap_data
    .dropna(subset=['Compound', 'Stint', 'TyreLife', 'SpeedFL', 'LapTime'])
    .groupby(['RACEYEAR', 'RACENUMBER', 'Driver', 'Compound'], as_index=False)
    .agg(
        MaxStint=('Stint', 'max'),
        AvgTyreLife=('TyreLife', 'mean'),
        AvgSpeedOnTyre=('SpeedFL', 'mean'),
        FastestSpeedOnTyre=('SpeedFL', 'max'),
        AvgLapTimeOnTyre=('LapTime', 'mean'),  # LapTime is already in milliseconds
        FastestLapTimeOnTyre=('LapTime', 'min')  # LapTime is already in milliseconds
    )
)

# Step 2: Pivot the data to get compounds as columns
tyre_stats_pivot = tyre_stats.pivot(index=['RACEYEAR', 'RACENUMBER', 'Driver'], columns='Compound')

# Step 3: Flatten MultiIndex columns with clear naming
tyre_stats_pivot.columns = [f"FP1_{stat}_{compound}" for stat, compound in tyre_stats_pivot.columns]

# (Optional) Reset index if you want a flat DataFrame
tyre_stats_pivot = tyre_stats_pivot.reset_index()

final_fp1_data=final_fp1_data.merge(tyre_stats_pivot, left_on=['RACEYEAR', 'RACENUMBER', 'Driver'], right_on=['RACEYEAR', 'RACENUMBER', 'Driver'], how="left" )


##################################################################################################################
# TRACK STATUS DATA
##################################################################################################################

track_status_data=pd.read_csv("FP1_DATA/FP1_track_status.csv")

agg_track_status = (
    track_status_data.groupby(["RACEYEAR", "RACENUMBER", "Message"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

agg_track_status=agg_track_status[["RACEYEAR","RACENUMBER","Red","SCDeployed","VSCDeployed","Yellow"]]
agg_track_status.columns=["RACEYEAR","RACENUMBER","FP1_Red","FP1_SCDeployed","FP1_VSCDeployed","FP1_Yellow"]
final_fp1_data=final_fp1_data.merge(agg_track_status, on=['RACEYEAR', 'RACENUMBER'], how="left")


final_fp1_data.to_csv("final_fp1_data.csv")