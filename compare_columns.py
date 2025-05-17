import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load all datasets
final_race_data = pd.read_csv("PROCESSED_DATA/final_race_data.csv")
final_quali_data = pd.read_csv("PROCESSED_DATA/final_quali_data.csv")
final_fp1_data = pd.read_csv("PROCESSED_DATA/final_fp1_data.csv")
final_fp2_data = pd.read_csv("PROCESSED_DATA/final_fp2_data.csv")
final_fp3_data = pd.read_csv("PROCESSED_DATA/final_fp3_data.csv")

# Get columns for each dataset
race_cols = set(final_race_data.columns)
quali_cols = set(final_quali_data.columns)
fp1_cols = set(final_fp1_data.columns)
fp2_cols = set(final_fp2_data.columns)
fp3_cols = set(final_fp3_data.columns)

# Find columns unique to each dataset
unique_to_race = race_cols - (quali_cols | fp1_cols | fp2_cols | fp3_cols)
unique_to_quali = quali_cols - (race_cols | fp1_cols | fp2_cols | fp3_cols)
unique_to_fp1 = fp1_cols - (race_cols | quali_cols | fp2_cols | fp3_cols)
unique_to_fp2 = fp2_cols - (race_cols | quali_cols | fp1_cols | fp3_cols)
unique_to_fp3 = fp3_cols - (race_cols | quali_cols | fp1_cols | fp2_cols)

# Find common columns across all datasets
common_cols = race_cols & quali_cols & fp1_cols & fp2_cols & fp3_cols

print("Columns unique to Race data:")
print("\n".join(sorted(unique_to_race)))
print("\nColumns unique to Qualifying data:")
print("\n".join(sorted(unique_to_quali)))
print("\nColumns unique to FP1 data:")
print("\n".join(sorted(unique_to_fp1)))
print("\nColumns unique to FP2 data:")
print("\n".join(sorted(unique_to_fp2)))
print("\nColumns unique to FP3 data:")
print("\n".join(sorted(unique_to_fp3)))
print("\nCommon columns across all datasets:")
print("\n".join(sorted(common_cols)))

# Check for sector time columns in each dataset
def get_sector_columns(df):
    return sorted([col for col in df.columns if 'Sector' in col])

print("\nSector columns in Race data:")
print("\n".join(get_sector_columns(final_race_data)))

print("\nSector columns in Qualifying data:")
print("\n".join(get_sector_columns(final_quali_data)))

print("\nSector columns in FP1 data:")
print("\n".join(get_sector_columns(final_fp1_data)))

print("\nSector columns in FP2 data:")
print("\n".join(get_sector_columns(final_fp2_data)))

print("\nSector columns in FP3 data:")
print("\n".join(get_sector_columns(final_fp3_data))) 