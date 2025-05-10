# EDGE CASES:
# - Merge on event type ((R,Q,FP1,2,3)) also



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


full_race_results=pd.read_csv("RACE_DATA/R_race_results.csv")
# full_race_results.head()

full_race_results=full_race_results[['DriverNumber', 'BroadcastName', 'Abbreviation',
       'DriverId', 'TeamName', 'TeamId', 'CountryCode', 'Position',
       'ClassifiedPosition', 'GridPosition', 'Time',
       'Status', 'Points', 'RACEYEAR', 'RACENUMBER']]

full_race_results["Time"] = pd.to_timedelta(full_race_results["Time"], errors='coerce')
full_race_results["Time"] = full_race_results["Time"].dt.total_seconds() * 1000


# ---------------------------------------------------
# Race Winners and benchmark pace
# ---------------------------------------------------

# Filtering for only completed: EDGE CASE (those who got disqualified)
everyone_who_completed=full_race_results[full_race_results["Status"]=="Finished"]

# Filter for race winner: EDGE CASE (Those whose classification changed after race win)
race_winner=full_race_results[(full_race_results["Status"]=="Finished") & (full_race_results["Position"]==1)]

race_winner_time=race_winner[["RACEYEAR","RACENUMBER","Time"]]
race_winner_time.columns=["RACEYEAR","RACENUMBER","WinnerTime"]

race_winner_driverid=race_winner[["RACEYEAR","RACENUMBER","DriverId"]]
race_winner_driverid.columns=["RACEYEAR","RACENUMBER","WinnerDriver"]

race_pace=full_race_results.merge(race_winner_time, on=["RACEYEAR","RACENUMBER"], how="left")
race_pace=race_pace.merge(race_winner_driverid, on=["RACEYEAR","RACENUMBER"], how="left")
race_pace["Racepace"]=race_pace["Time"]/race_pace["WinnerTime"]

race_pace["IsWinnerFlag"]=(race_pace["Time"]==race_pace["WinnerTime"]).astype(int)

# ---------------------------------------------------
# WEATHER DATA
# ---------------------------------------------------

full_weather_data=pd.read_csv("RACE_DATA/R_weather_data.csv")

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
    col[0] if col[0] in ['RACEYEAR', 'RACENUMBER'] else 'RACE_' + '_'.join(col).strip('_')
    for col in agg_weather_data.columns
]

final_race_data=race_pace.merge(agg_weather_data, on=['RACEYEAR', 'RACENUMBER'], how="left")


# ---------------------------------------------------
# DRIVING CHARACTERISTICS
# ---------------------------------------------------

# -------------- Avg sector 1/2/3 speed

full_lap_data=pd.read_csv("RACE_DATA/R_lap_data.csv")

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

# -------------- Avg pit time per race and number of pitstops

full_lap_data["PitOutTime"] = pd.to_timedelta(full_lap_data["PitOutTime"], errors='coerce')
full_lap_data["PitOutTime"] = full_lap_data["PitOutTime"].dt.total_seconds() * 1000

full_lap_data["PitInTime"] = pd.to_timedelta(full_lap_data["PitInTime"], errors='coerce')
full_lap_data["PitInTime"] = full_lap_data["PitInTime"].dt.total_seconds() * 1000

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

final_race_data=final_race_data.merge(pit_summary, left_on=['RACEYEAR', 'RACENUMBER', 'Abbreviation'], right_on=['RACEYEAR', 'RACENUMBER', 'Driver'], how="left" )


# if driver does not pit at all (DNS, DNF) then there will be no values for pit, so row mismatch with sector data


# -------------- Avg tyre life SML / Avg speed on SML tyres

tyre_stats = (
    full_lap_data
    .dropna(subset=['Compound'])
    .groupby(['RACEYEAR', 'RACENUMBER', 'Driver', 'Compound'], as_index=False)
    .agg(
        AvgTyreLife=('TyreLife', 'mean'),
        AvgSpeedOnTyre=('SpeedFL', 'mean'),
        AvgLapTimeOnTyre=('LapTime', lambda x: pd.to_timedelta(x, errors='coerce').dt.total_seconds().mean())
    )
)

# Pivot tyre stats so each tyre type becomes a set of columns
tyre_stats_pivot = tyre_stats.pivot(index=['RACEYEAR', 'RACENUMBER', 'Driver'], columns='Compound')
tyre_stats_pivot.columns = [f"RACE_{stat}_{compound}" for stat, compound in tyre_stats_pivot.columns]

# Reset index for easier access and readability
tyre_stats_pivot = tyre_stats_pivot.reset_index()

final_race_data=final_race_data.merge(tyre_stats_pivot, left_on=['RACEYEAR', 'RACENUMBER', 'Abbreviation'], right_on=['RACEYEAR', 'RACENUMBER', 'Driver'], how="left" )


# ---------------------------------------------------
# TRACK STATUS DATA
# ---------------------------------------------------

track_status_data=pd.read_csv("RACE_DATA/R_track_status.csv")

agg_track_status = (
    track_status_data.groupby(["RACEYEAR", "RACENUMBER", "Message"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

agg_track_status=agg_track_status[["RACEYEAR","RACENUMBER","Red","SCDeployed","VSCDeployed","Yellow"]]
agg_track_status.columns=["RACEYEAR","RACENUMBER","RACE_Red","RACE_SCDeployed","RACE_VSCDeployed","RACE_Yellow"]
final_race_data=final_race_data.merge(agg_track_status, on=['RACEYEAR', 'RACENUMBER'], how="left")


# ----------------------------------------------------------------------------------------------------------------Add race time of day here (?)

# ----------------------------------------------------------------------------------------------------------------Add track information here (?)

full_track_corners=pd.read_csv("RACE_DATA/R_track_structure.csv")
full_track_corners['CornerCategory'] = full_track_corners['Angle'].abs().apply(lambda x: 'slow' if x > 90 else 'medium' if x > 60 else 'fast')

agg_track_corners = full_track_corners.pivot_table(
    index=['RACEYEAR', 'RACENUMBER'],
    columns='CornerCategory',
    values='Angle',  # or any column (we just need a count)
    aggfunc='count',
    fill_value=0
).reset_index()

agg_track_corners['TotalCorners'] = agg_track_corners[['slow', 'medium', 'fast']].sum(axis=1)

final_race_data=final_race_data.merge(agg_track_corners, on=['RACEYEAR', 'RACENUMBER'], how="left")

# ---------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------

correlation_matrix = final_race_data.corr(numeric_only=True)
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


target_var = 'Position'
# Compute correlation matrix

correlation_matrix = final_race_data.corr(numeric_only=True)
# Extract only the row/column for the target variable (excluding self-correlation)

target_corr = correlation_matrix[target_var].drop(labels=[target_var])

# Plot as a horizontal heatmap
plt.figure(figsize=(35, 2))
sns.heatmap(target_corr.to_frame().T, annot=True, cmap='coolwarm', fmt='.2f')
plt.title(f'Correlation of {target_var} vs All Other Variables')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


final_race_data.to_csv("final_data.csv")