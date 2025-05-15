import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# ========== 1. Load and Prepare Data ==========
df = pd.read_csv("final_full_data_only_fp_predictors.csv")

# Select relevant columns
selected_columns = [
    # Basic identifiers
    'RACEYEAR', 'RACENUMBER', 'DriverId', 'TeamId',
    
    # Track and event info
    'Country', 'Location', 'EventFormat', 'fast', 'medium', 'slow', 'TotalCorners',
    'Session1TimeOfDay', 'Session2TimeOfDay', 'Session3TimeOfDay',
    
    # FP1 Performance and Weather
    'FP1_AvgPitStopDuration_ms', 'FP1_TotalPitStops',
    # Weather data including extremes
    'FP1_AirTemp_mean', 'FP1_AirTemp_min', 'FP1_AirTemp_max',
    'FP1_Humidity_mean', 'FP1_Humidity_min', 'FP1_Humidity_max',
    'FP1_Pressure_mean', 'FP1_Pressure_min', 'FP1_Pressure_max',
    'FP1_Rainfall_max',
    'FP1_TrackTemp_mean', 'FP1_TrackTemp_min', 'FP1_TrackTemp_max',
    'FP1_WindDirection_mean',
    'FP1_WindSpeed_mean', 'FP1_WindSpeed_max',
    
    # Track conditions
    'FP1_Red', 'FP1_SCDeployed', 'FP1_VSCDeployed', 'FP1_Yellow',
    
    # FP1 Tire Performance
    'FP1_MaxStint_HARD', 'FP1_MaxStint_INTERMEDIATE', 'FP1_MaxStint_MEDIUM', 'FP1_MaxStint_SOFT',
    'FP1_AvgTyreLife_HARD', 'FP1_AvgTyreLife_INTERMEDIATE', 'FP1_AvgTyreLife_MEDIUM', 'FP1_AvgTyreLife_SOFT',
    'FP1_AvgLapTimeOnTyre_HARD', 'FP1_AvgLapTimeOnTyre_INTERMEDIATE', 'FP1_AvgLapTimeOnTyre_MEDIUM', 'FP1_AvgLapTimeOnTyre_SOFT',
    'FP1_FastestLapTimeOnTyre_HARD', 'FP1_FastestLapTimeOnTyre_INTERMEDIATE', 'FP1_FastestLapTimeOnTyre_MEDIUM', 'FP1_FastestLapTimeOnTyre_SOFT',
    
    # FP2 (same pattern as FP1)
    'FP2_AvgPitStopDuration_ms', 'FP2_TotalPitStops',
    'FP2_AirTemp_mean', 'FP2_AirTemp_min', 'FP2_AirTemp_max',
    'FP2_Humidity_mean', 'FP2_Humidity_min', 'FP2_Humidity_max',
    'FP2_Pressure_mean', 'FP2_Pressure_min', 'FP2_Pressure_max',
    'FP2_Rainfall_max',
    'FP2_TrackTemp_mean', 'FP2_TrackTemp_min', 'FP2_TrackTemp_max',
    'FP2_WindDirection_mean',
    'FP2_WindSpeed_mean', 'FP2_WindSpeed_max',
    'FP2_Red', 'FP2_SCDeployed', 'FP2_VSCDeployed', 'FP2_Yellow',
    'FP2_MaxStint_HARD', 'FP2_MaxStint_INTERMEDIATE', 'FP2_MaxStint_MEDIUM', 'FP2_MaxStint_SOFT',
    'FP2_AvgTyreLife_HARD', 'FP2_AvgTyreLife_INTERMEDIATE', 'FP2_AvgTyreLife_MEDIUM', 'FP2_AvgTyreLife_SOFT',
    'FP2_AvgLapTimeOnTyre_HARD', 'FP2_AvgLapTimeOnTyre_INTERMEDIATE', 'FP2_AvgLapTimeOnTyre_MEDIUM', 'FP2_AvgLapTimeOnTyre_SOFT',
    'FP2_FastestLapTimeOnTyre_HARD', 'FP2_FastestLapTimeOnTyre_INTERMEDIATE', 'FP2_FastestLapTimeOnTyre_MEDIUM', 'FP2_FastestLapTimeOnTyre_SOFT',
    
    # FP3 (same pattern as FP1/FP2)
    'FP3_AvgPitStopDuration_ms', 'FP3_TotalPitStops',
    'FP3_AirTemp_mean', 'FP3_AirTemp_min', 'FP3_AirTemp_max',
    'FP3_Humidity_mean', 'FP3_Humidity_min', 'FP3_Humidity_max',
    'FP3_Pressure_mean', 'FP3_Pressure_min', 'FP3_Pressure_max',
    'FP3_Rainfall_max',
    'FP3_TrackTemp_mean', 'FP3_TrackTemp_min', 'FP3_TrackTemp_max',
    'FP3_WindDirection_mean',
    'FP3_WindSpeed_mean', 'FP3_WindSpeed_max',
    'FP3_Red', 'FP3_SCDeployed', 'FP3_VSCDeployed', 'FP3_Yellow',
    'FP3_MaxStint_HARD', 'FP3_MaxStint_INTERMEDIATE', 'FP3_MaxStint_MEDIUM', 'FP3_MaxStint_SOFT',
    'FP3_AvgTyreLife_HARD', 'FP3_AvgTyreLife_INTERMEDIATE', 'FP3_AvgTyreLife_MEDIUM', 'FP3_AvgTyreLife_SOFT',
    'FP3_AvgLapTimeOnTyre_HARD', 'FP3_AvgLapTimeOnTyre_INTERMEDIATE', 'FP3_AvgLapTimeOnTyre_MEDIUM', 'FP3_AvgLapTimeOnTyre_SOFT',
    'FP3_FastestLapTimeOnTyre_HARD', 'FP3_FastestLapTimeOnTyre_INTERMEDIATE', 'FP3_FastestLapTimeOnTyre_MEDIUM', 'FP3_FastestLapTimeOnTyre_SOFT',
    
    # Target variable
    'IsWinnerFlag'
]

# Filter DataFrame to only include selected columns
df = df[selected_columns]

X = df.drop(columns=['IsWinnerFlag'])
y = df['IsWinnerFlag']