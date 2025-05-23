import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


final_race_data= pd.read_csv("PROCESSED_DATA/final_race_data.csv")
final_quali_data= pd.read_csv("PROCESSED_DATA/final_quali_data.csv")
final_fp1_data= pd.read_csv("PROCESSED_DATA/final_fp1_data.csv")
final_fp2_data= pd.read_csv("PROCESSED_DATA/final_fp2_data.csv")
final_fp3_data= pd.read_csv("PROCESSED_DATA/final_fp3_data.csv")

##################################################################################################################
# FINAL CLEAN UPS
##################################################################################################################

# ---------------------------------------------------------------------------------------------------------------- Final Cleaning Race Data

if 'Unnamed: 0' in final_race_data.columns:
    final_race_data = final_race_data.drop('Unnamed: 0', axis=1)

# ---------------------------------------------------------------------------------------------------------------- Final Cleaning Quali Data

if 'Unnamed: 0' in final_quali_data.columns:
    final_quali_data = final_quali_data.drop('Unnamed: 0', axis=1)  

# Drop Position column as it's not needed for prediction
if 'Position' in final_quali_data.columns:
    final_quali_data = final_quali_data.drop('Position', axis=1)

# ---------------------------------------------------------------------------------------------------------------- Final Cleaning FP1

fp1_columns = {col: f'FP1_{col}' for col in final_fp1_data.columns if not col.startswith('FP1_')}
final_fp1_data.rename(columns=fp1_columns, inplace=True)

if 'FP1_Unnamed: 0' in final_fp1_data.columns:
    final_fp1_data = final_fp1_data.drop('FP1_Unnamed: 0', axis=1)

# Remove UNKNOWN and TEST_UNKNOWN columns from FP1
unknown_cols_fp1 = [col for col in final_fp1_data.columns if 'UNKNOWN' in col]
final_fp1_data = final_fp1_data.drop(columns=unknown_cols_fp1)

# ---------------------------------------------------------------------------------------------------------------- Final Cleaning FP2

fp2_columns = {col: f'FP2_{col}' for col in final_fp2_data.columns if not col.startswith('FP2_')}
final_fp2_data.rename(columns=fp2_columns, inplace=True)

if 'FP2_Unnamed: 0' in final_fp2_data.columns:
    final_fp2_data = final_fp2_data.drop('FP2_Unnamed: 0', axis=1)

# Remove UNKNOWN and TEST_UNKNOWN columns from FP2
unknown_cols_fp2 = [col for col in final_fp2_data.columns if 'UNKNOWN' in col]
final_fp2_data = final_fp2_data.drop(columns=unknown_cols_fp2)

# ---------------------------------------------------------------------------------------------------------------- Final Cleaning FP3

fp3_columns = {col: f'FP3_{col}' for col in final_fp3_data.columns if not col.startswith('FP3_')}
final_fp3_data.rename(columns=fp3_columns, inplace=True)

if 'FP3_Unnamed: 0' in final_fp3_data.columns:
    final_fp3_data = final_fp3_data.drop('FP3_Unnamed: 0', axis=1)

# Remove UNKNOWN and TEST_UNKNOWN columns from FP3
unknown_cols_fp3 = [col for col in final_fp3_data.columns if 'UNKNOWN' in col]
final_fp3_data = final_fp3_data.drop(columns=unknown_cols_fp3)

##################################################################################################################
# MEGA JOIN
##################################################################################################################

final_full_data=final_race_data.merge(final_fp1_data, left_on=["RACEYEAR","RACENUMBER","Abbreviation"], right_on=["FP1_RACEYEAR","FP1_RACENUMBER","FP1_Driver"],how="left")
final_full_data=final_full_data.merge(final_fp2_data, left_on=["RACEYEAR","RACENUMBER","Abbreviation"], right_on=["FP2_RACEYEAR","FP2_RACENUMBER","FP2_Driver"],how="left")
final_full_data=final_full_data.merge(final_fp3_data, left_on=["RACEYEAR","RACENUMBER","Abbreviation"], right_on=["FP3_RACEYEAR","FP3_RACENUMBER","FP3_Driver"],how="left")
final_full_data=final_full_data.merge(final_quali_data, left_on=["RACEYEAR","RACENUMBER","Abbreviation"], right_on=["RACEYEAR","RACENUMBER","Abbreviation"],how="left")

cols_to_min = [
    "FP1_FastestFullLapTime_ms", "FP1_FastestSector1Time_ms", "FP1_FastestSector2Time_ms", "FP1_FastestSector3Time_ms",
    "FP2_FastestFullLapTime_ms", "FP2_FastestSector1Time_ms", "FP2_FastestSector2Time_ms", "FP2_FastestSector3Time_ms",
    "FP3_FastestFullLapTime_ms", "FP3_FastestSector1Time_ms", "FP3_FastestSector2Time_ms", "FP3_FastestSector3Time_ms",
    "Quali_FastestSector1Time_ms", "Quali_FastestSector2Time_ms", "Quali_FastestSector3Time_ms",
    "Quali_Q1_ms", "Quali_Q2_ms", "Quali_Q3_ms",
]

fastest_in_every_session=final_full_data.groupby(["RACEYEAR","RACENUMBER"])[cols_to_min].min().reset_index().rename(columns=lambda col: f"OverallSession_{col}" if col not in ["RACEYEAR", "RACENUMBER"] else col)

final_full_data=final_full_data.merge(fastest_in_every_session, left_on=["RACEYEAR","RACENUMBER"], right_on=["RACEYEAR","RACENUMBER"],how="left")

for col in cols_to_min:
    final_full_data[f"ComparativePace_{col}"] = final_full_data[col] / final_full_data[f"OverallSession_{col}"]

##################################################################################################################
# FINAL DATA SET(S)
##################################################################################################################

columns_to_drop = []

for col in cols_to_min:
    columns_to_drop.append(col)
    columns_to_drop.append(f"OverallSession_{col}")

# Clean up duplicate columns
columns_to_drop.extend([
    # Duplicate driver/race identifiers
    "FP1_RACEYEAR", "FP1_RACENUMBER", "FP1_Driver",
    "FP2_RACEYEAR", "FP2_RACENUMBER", "FP2_Driver",
    "FP3_RACEYEAR", "FP3_RACENUMBER", "FP3_Driver",
    "Driver_x", "Driver_y","Time","WinnerTime","WinnerDriver"
])

# Drop unwanted columns
final_full_data = final_full_data.drop(columns=columns_to_drop, errors='ignore')

# Save without index
final_full_data.to_csv("PROCESSED_DATA/final_full_data.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------- Only Free Practice Predictors

columns_to_drop = [
    "ClassifiedPosition",
    "GridPosition",
    "Time",
    "Status",
    "Points",
    "WinnerTime",
    "WinnerDriver",
    "Racepace",
    "RACE_AirTemp_mean",
    "RACE_AirTemp_min",
    "RACE_AirTemp_max",
    "RACE_Humidity_mean",
    "RACE_Humidity_min",
    "RACE_Humidity_max",
    "RACE_Pressure_mean",
    "RACE_Pressure_min",
    "RACE_Pressure_max",
    "RACE_Rainfall_max",
    "RACE_TrackTemp_mean",
    "RACE_TrackTemp_min",
    "RACE_TrackTemp_max",
    "RACE_WindDirection_mean",
    "RACE_WindSpeed_mean",
    "RACE_WindSpeed_max",
    "RACE_MaxStint_HARD",
    "RACE_MaxStint_INTERMEDIATE",
    "RACE_MaxStint_MEDIUM",
    "RACE_MaxStint_SOFT",
    "RACE_MaxStint_WET",
    "RACE_AvgTyreLife_HARD",
    "RACE_AvgTyreLife_INTERMEDIATE",
    "RACE_AvgTyreLife_MEDIUM",
    "RACE_AvgTyreLife_SOFT",
    "RACE_AvgTyreLife_WET",
    "RACE_AvgSpeedOnTyre_HARD",
    "RACE_AvgSpeedOnTyre_INTERMEDIATE",
    "RACE_AvgSpeedOnTyre_MEDIUM",
    "RACE_AvgSpeedOnTyre_SOFT",
    "RACE_AvgSpeedOnTyre_WET",
    "RACE_AvgLapTimeOnTyre_HARD",
    "RACE_AvgLapTimeOnTyre_INTERMEDIATE",
    "RACE_AvgLapTimeOnTyre_MEDIUM",
    "RACE_AvgLapTimeOnTyre_SOFT",
    "RACE_AvgLapTimeOnTyre_WET",
    "RACE_Red",
    "RACE_SCDeployed",
    "RACE_VSCDeployed",
    "RACE_Yellow"
]

final_full_data_only_fp_predictors = final_full_data.drop(columns=columns_to_drop)

final_full_data_only_fp_predictors.to_csv("PROCESSED_DATA/final_full_data_only_fp_predictors.csv")


##################################################################################################################
# Correlation Matrix
##################################################################################################################

# ---------------------------------------------------------------------------------------------------------------- Full data Correlation Matrix

# correlation_matrix = final_full_data.corr(numeric_only=True)
# plt.figure(figsize=(100, 100))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

# ---------------------------------------------------------------------------------------------------------------- Only Free Practice Predictors Correlation Matrix

# correlation_matrix = final_full_data_only_fp_predictors.corr(numeric_only=True)
# plt.figure(figsize=(100, 100))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()  





