# GRID POSITION DELTA NEEDS TO BE NORMALIZED BY NUMBER OF RACES COMPLETEDPER DRIVER


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

##################################################################################################################
# DRIVER POINTS OVER TIME
##################################################################################################################


# Load the data
df = pd.read_csv("PROCESSED_DATA/final_full_data.csv")

# Filter necessary columns
df_points = df[['RACEYEAR', 'RACENUMBER', 'DriverId', 'Points']].copy()

# Sort by year, race number
df_points = df_points.sort_values(['DriverId', 'RACEYEAR', 'RACENUMBER'])

# Calculate cumulative points per driver
df_points['CumulativePoints'] = df_points.groupby('DriverId')['Points'].cumsum()

# Create a combined race identifier for x-axis
df_points['Race'] = df_points['RACEYEAR'].astype(str) + '-' + df_points['RACENUMBER'].astype(str)

# Pivot for plotting: rows = race, columns = driver, values = cumulative points
pivot = df_points.pivot_table(index='Race', columns='DriverId', values='CumulativePoints')

# Extract year and race number as integers
race_index_df = pivot.index.to_series().str.split('-', expand=True).astype(int)
race_index_df.columns = ['Year', 'RaceNum']
# Get sorted order
sorted_order = race_index_df.sort_values(['Year', 'RaceNum']).index
# Reindex pivot table
pivot = pivot.loc[sorted_order]

# Plot
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Use colorblind-friendly colormap
colors = plt.cm.tab20.colors  # This is a colorblind-friendly colormap
for i, (driver, data) in enumerate(pivot.items()):
    ax.plot(range(len(data.index)), data.values, marker='o', label=driver, color=colors[i % len(colors)])

# Set x-tick labels to show (Year, RaceNumber)
ax.set_xticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.index, rotation=90)

plt.title('Cumulative Points per Driver (by Race)')
plt.xlabel('Race (Year-Number)')
plt.ylabel('Cumulative Points')
plt.legend(title='Driver', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('ANALYTICS/DRIVER_POINTS_OVER_TIME.png')
plt.show()


##################################################################################################################
# TEAM POINTS OVER TIME
##################################################################################################################

# Filter necessary columns for team points
df_team = df[['RACEYEAR', 'RACENUMBER', 'TeamName', 'Points']].copy()

# Sort by year, race number
df_team = df_team.sort_values(['TeamName', 'RACEYEAR', 'RACENUMBER'])

# Calculate cumulative points per team
df_team['CumulativePoints'] = df_team.groupby('TeamName')['Points'].cumsum()

# Create a combined race identifier for x-axis
df_team['Race'] = df_team['RACEYEAR'].astype(str) + '-' + df_team['RACENUMBER'].astype(str)

# Pivot for plotting: rows = race, columns = team, values = cumulative points
pivot_team = df_team.pivot_table(index='Race', columns='TeamName', values='CumulativePoints')

# Extract year and race number as integers
race_index_df_team = pivot_team.index.to_series().str.split('-', expand=True).astype(int)
race_index_df_team.columns = ['Year', 'RaceNum']
sorted_order_team = race_index_df_team.sort_values(['Year', 'RaceNum']).index
pivot_team = pivot_team.loc[sorted_order_team]

# Plot
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Use colorblind-friendly colormap
colors = plt.cm.tab20.colors  # This is a colorblind-friendly colormap
for i, (team, data) in enumerate(pivot_team.items()):
    ax.plot(range(len(data.index)), data.values, marker='o', label=team, color=colors[i % len(colors)])

# Set x-tick labels to show (Year, RaceNumber)
ax.set_xticks(range(len(pivot_team.index)))
ax.set_xticklabels(pivot_team.index, rotation=90)

plt.title('Cumulative Points per Team (by Race)')
plt.xlabel('Race (Year-Number)')
plt.ylabel('Cumulative Points')
plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('ANALYTICS/TEAM_POINTS_OVER_TIME.png')
plt.show()


##################################################################################################################
# DRIVER: AVERAGE GRID POSITION VS FINAL POSITION
##################################################################################################################

# (Assume 'Grid Position' is qualifying, 'Position' is race result)
df_avg = df[['DriverId', 'BroadcastName', 'GridPosition', 'Position']].copy()

# Remove rows with missing positions
df_avg = df_avg.dropna(subset=['GridPosition', 'Position'])

# Group by driver and calculate mean positions
avg_positions = df_avg.groupby(['DriverId', 'BroadcastName']).agg({
    'GridPosition': 'mean',
    'Position': 'mean'
}).reset_index()

# Sort by average race position (lower is better)
avg_positions = avg_positions.sort_values('GridPosition')

fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(avg_positions))

# Plot grid position and race position as dots
ax.scatter(avg_positions['GridPosition'], y, color='tab:blue', label='Avg Grid Position', s=100, marker='o')
ax.scatter(avg_positions['Position'], y, color='tab:orange', label='Avg Race Position', s=100, marker='D')

# Draw lines between grid and race position for each driver
for i, row in enumerate(avg_positions.itertuples()):
    ax.plot([row.GridPosition, row.Position], [i, i], color='gray', linestyle='--', alpha=0.7)

ax.set_yticks(y)
ax.set_yticklabels(avg_positions['BroadcastName'])
ax.set_xlabel('Average Position (Lower is Better)')
ax.set_title('Average Grid (Qualifying) vs Final Race Position per Driver')
ax.legend()
ax.invert_xaxis()  # Lower (better) positions to the right
plt.tight_layout()
plt.savefig('ANALYTICS/DRIVER_AVG_GRID_VS_FINAL_POSITION_DOTPLOT.png')
plt.show()



##################################################################################################################
# Average top speed on tyre compounds per race
##################################################################################################################

tyre_compounds = ['HARD', 'MEDIUM', 'SOFT', 'INTERMEDIATE', 'WET']
speed_cols = [f'RACE_FastestSpeedOnTyre_{comp}' for comp in tyre_compounds]

# Replace 0s with NaN so they're excluded from mean calculation
df[speed_cols] = df[speed_cols].replace(0, np.nan)

# Group by race and calculate mean only for non-NaN (non-zero) values
grouped = df.groupby('Location')[speed_cols].mean().reset_index()

# Melt to long format for plotting
melted = grouped.melt(id_vars=['Location'], 
                      value_vars=speed_cols,
                      var_name='TyreCompound',
                      value_name='FastestTopSpeed')

# Clean TyreCompound column
melted['TyreCompound'] = melted['TyreCompound'].str.replace('RACE_FastestSpeedOnTyre_', '')

# Remove rows where FastestTopSpeed is NaN (no valid data for that compound in that race)
melted = melted.dropna(subset=['FastestTopSpeed'])

# Sort by speed range (largest difference between fastest and slowest compound per location)
race_ranges = melted.groupby('Location')['FastestTopSpeed'].agg(lambda x: x.max() - x.min()).sort_values(ascending=False)
races = race_ranges.index.tolist()

# Reorder melted data according to the sorted races
melted['Location'] = pd.Categorical(melted['Location'], categories=races, ordered=True)
melted = melted.sort_values('Location')

# Get y positions for the sorted races
y = np.arange(len(races))

# Prepare color and marker maps for tyre compounds
color_map = {
    'HARD': 'tab:blue', 
    'MEDIUM': 'tab:orange', 
    'SOFT': 'tab:green', 
    'INTERMEDIATE': 'tab:red', 
    'WET': 'tab:purple'
}

marker_map = {
    'HARD': 'o', 
    'MEDIUM': 'D', 
    'SOFT': 's', 
    'INTERMEDIATE': '^', 
    'WET': 'X'
}

fig, ax = plt.subplots(figsize=(14, 10))

labeled_compounds = set()

# Plot dots for each tyre compound at y-position corresponding to the race
for i, race in enumerate(races):
    race_data = melted[melted['Location'] == race]
    for _, row in race_data.iterrows():
        compound = row['TyreCompound']
        # Only add label if this compound hasn't been labeled yet
        label = compound if compound not in labeled_compounds else ""
        if compound not in labeled_compounds:
            labeled_compounds.add(compound)
            
        ax.scatter(row['FastestTopSpeed'], i, 
                   color=color_map.get(compound, 'black'), 
                   marker=marker_map.get(compound, 'o'), 
                   s=100, 
                   label=label)

    # Optional: draw lines connecting dots for this race
    speeds = race_data['FastestTopSpeed'].values
    ax.plot(speeds, [i]*len(speeds), color='gray', linestyle='--', alpha=0.5)

# Set y-ticks to race names
ax.set_yticks(y)
ax.set_yticklabels(races)

ax.set_xlabel('Fastest Top Speed')
ax.set_title('Fastest Top Speed per Tyre Compound Across Locations (Sorted by Speed Range)')

# Handle legend without duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title='Tyre Compound')

plt.tight_layout()
plt.savefig('ANALYTICS/FASTEST_TOP_SPEED_PER_TYRE_COMPOUND_ACROSS_LOCATIONS.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Print the speed ranges for analysis
print("\nSpeed Ranges by Location (largest to smallest):")
print("=" * 50)
for location in races:
    location_data = melted[melted['Location'] == location]
    speed_range = location_data['FastestTopSpeed'].max() - location_data['FastestTopSpeed'].min()
    min_speed = location_data['FastestTopSpeed'].min()
    max_speed = location_data['FastestTopSpeed'].max()
    print(f"{location:<20} | Range: {speed_range:6.1f} km/h | Min: {min_speed:6.1f} | Max: {max_speed:6.1f}")


##################################################################################################################
# Weather and tyre compound correlation
##################################################################################################################

x_cols = [
    "RACE_AirTemp_mean", "RACE_AirTemp_min", "RACE_AirTemp_max",
    "RACE_Humidity_mean", "RACE_Humidity_min", "RACE_Humidity_max",
    "RACE_Pressure_mean", "RACE_Pressure_min", "RACE_Pressure_max",
    "RACE_TrackTemp_mean", "RACE_TrackTemp_min", "RACE_TrackTemp_max",
    "RACE_WindDirection_mean", "RACE_WindSpeed_mean", "RACE_WindSpeed_max"
]

y_cols = [
    "RACE_TotalPitStops","RACE_MaxStint_SOFT", "RACE_AvgTyreLife_SOFT", "RACE_AvgSpeedOnTyre_SOFT",
    "RACE_FastestSpeedOnTyre_SOFT", "RACE_AvgLapTimeOnTyre_SOFT", "RACE_FastestLapTimeOnTyre_SOFT",
    
    "RACE_MaxStint_MEDIUM", "RACE_AvgTyreLife_MEDIUM", "RACE_AvgSpeedOnTyre_MEDIUM",
    "RACE_FastestSpeedOnTyre_MEDIUM", "RACE_AvgLapTimeOnTyre_MEDIUM", "RACE_FastestLapTimeOnTyre_MEDIUM",
    
    "RACE_MaxStint_HARD", "RACE_AvgTyreLife_HARD", "RACE_AvgSpeedOnTyre_HARD",
    "RACE_FastestSpeedOnTyre_HARD", "RACE_AvgLapTimeOnTyre_HARD", "RACE_FastestLapTimeOnTyre_HARD",
    
    # "RACE_MaxStint_INTERMEDIATE", "RACE_AvgTyreLife_INTERMEDIATE", "RACE_AvgSpeedOnTyre_INTERMEDIATE",
    # "RACE_FastestSpeedOnTyre_INTERMEDIATE", "RACE_AvgLapTimeOnTyre_INTERMEDIATE", "RACE_FastestLapTimeOnTyre_INTERMEDIATE",
    
    # "RACE_MaxStint_WET", "RACE_AvgTyreLife_WET", "RACE_AvgSpeedOnTyre_WET",
    # "RACE_FastestSpeedOnTyre_WET", "RACE_AvgLapTimeOnTyre_WET", "RACE_FastestLapTimeOnTyre_WET"
]

correlation_matrix = pd.DataFrame(index=x_cols, columns=y_cols)
for x in x_cols:
    for y in y_cols:
        correlation_matrix.loc[x, y] = df[[x, y]].corr().iloc[0, 1]
correlation_matrix = correlation_matrix.astype(float)

plt.figure(figsize=(len(y_cols) * 0.7, len(x_cols) * 0.7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Selected Weather Features and Tyre Performance')
plt.tight_layout()
plt.savefig('ANALYTICS/WEATHER_AND_TYRE_COMPOUND_CORRELATION.png')
plt.show()

##################################################################################################################
# RACE INCIDENTS PER TRACK
##################################################################################################################

# Step 1: Group by Location and calculate averages
avg_flags_by_location = df.groupby('Location')[["RACE_Red", "RACE_SCDeployed", "RACE_VSCDeployed", "RACE_Yellow"]].mean()

# Step 2: Count number of races per location
race_counts = (df['Location'].value_counts().rename('RaceCount'))/20

# Step 3: Merge average flags with race counts
avg_flags_by_location = avg_flags_by_location.merge(race_counts, left_index=True, right_index=True)

# Step 4: Create a 'Total' column for sorting
avg_flags_by_location['Total'] = avg_flags_by_location[["RACE_Red", "RACE_SCDeployed", "RACE_VSCDeployed", "RACE_Yellow"]].sum(axis=1)

# Step 5: Sort by total incidents
avg_flags_by_location = avg_flags_by_location.sort_values('Total', ascending=False)

# Step 6: Plotting
fig, ax = plt.subplots(figsize=(15, 10))

# Stacked bar components
ax.bar(avg_flags_by_location.index, avg_flags_by_location['RACE_Red'], label='Red Flags', color='tab:red')
ax.bar(avg_flags_by_location.index, avg_flags_by_location['RACE_Yellow'],
       bottom=avg_flags_by_location['RACE_Red'], label='Yellow Flags', color='gold')
ax.bar(avg_flags_by_location.index, avg_flags_by_location['RACE_SCDeployed'],
       bottom=avg_flags_by_location['RACE_Red'] + avg_flags_by_location['RACE_Yellow'],
       label='Safety Car', color='gray')
ax.bar(avg_flags_by_location.index, avg_flags_by_location['RACE_VSCDeployed'],
       bottom=avg_flags_by_location['RACE_Red'] + avg_flags_by_location['RACE_Yellow'] + avg_flags_by_location['RACE_SCDeployed'],
       label='Virtual Safety Car', color='tab:blue')

# Step 7: Add total race count as annotation
for i, (loc, row) in enumerate(avg_flags_by_location.iterrows()):
    ax.text(i, row['Total'] + 0.1, f"{int(row['RaceCount'])} races", ha='center', va='bottom', fontsize=9, rotation=90)

# Final touches
ax.set_title('Average Incidents per Race by Location (with Race Counts)', fontsize=16)
ax.set_ylabel('Average Incident Count per Race')
ax.set_xticks(range(len(avg_flags_by_location)))
ax.set_xticklabels(avg_flags_by_location.index, rotation=45, ha='right')
ax.legend(title='Incident Type')
max_y = avg_flags_by_location['Total'].max()
ax.set_ylim(0, max_y + 1)

plt.tight_layout()
plt.savefig('ANALYTICS/RACE_INCIDENTS_PER_TRACK.png')
plt.show()

##################################################################################################################
# TRACK STRUCTURE
##################################################################################################################

# Get distinct track corner data
track_structure = df[["Location", "fast", "medium", "slow"]].drop_duplicates()
track_structure = track_structure.sort_values(by='Location').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15, 10))

# Stacked bar chart
ax.bar(range(len(track_structure)), track_structure['fast'], label='Fast Corners (* < 60)', color='#88CCEE')     # Muted Blue
ax.bar(range(len(track_structure)), track_structure['medium'],
       bottom=track_structure['fast'], label='Medium Corners (60 < * < 90)', color='#DDCC77')  # Muted Orange
ax.bar(range(len(track_structure)), track_structure['slow'],
       bottom=track_structure['fast'] + track_structure['medium'],
       label='Slow Corners (* > 90)', color='#CC6677')   # Muted Red

# Labels and final formatting
ax.set_title('Corner Type Distribution per Track', fontsize=16)
ax.set_ylabel('Number of Corners')
ax.set_xticks(range(len(track_structure)))
ax.set_xticklabels(track_structure['Location'], rotation=45, ha='right')
ax.legend(title='Corner Type')
ax.set_ylim(0, (track_structure[['fast', 'medium', 'slow']].sum(axis=1).max()) + 2)

plt.tight_layout()
plt.savefig('ANALYTICS/TRACK_STRUCTURE.png')
plt.show()

##################################################################################################################
# DRIVER CORRELATION WITH TRACK STRUCTURE
##################################################################################################################

# Step 1: One-hot encode 'Location' column

location_dummies = pd.get_dummies(df['Location'], prefix='Loc')

# Step 2: Normalize numeric features 'slow', 'medium', 'fast'
numeric_features = df[['slow', 'medium', 'fast']]
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns, index=df.index)

# Step 3: Combine normalized numeric features and location dummies with BroadcastName
features = pd.concat([df[['DriverId']], numeric_scaled, location_dummies], axis=1)

# Step 4: Group by 'BroadcastName' and compute mean
driver_features = features.groupby('DriverId').mean().T  # transpose for heatmap layout

# Step 5: Plot heatmap with values shown
plt.figure(figsize=(20, len(driver_features) * 0.5))
sns.heatmap(driver_features, cmap='coolwarm', linewidths=0.5,
            cbar_kws={'label': 'Average Value'}, annot=True, fmt='.2f')
plt.title('Drivers vs Track Features and Locations')
plt.ylabel('Track Features and Locations')
plt.xlabel('Drivers')
plt.tight_layout()
plt.savefig('ANALYTICS/DRIVER_CORRELATION_WITH_TRACK_STRUCTURE.png')
plt.show()

##################################################################################################################
# TEAM CORRELATION WITH TRACK STRUCTURE
##################################################################################################################

location_dummies = pd.get_dummies(df['Location'], prefix='Loc')

# Step 2: Normalize numeric features 'slow', 'medium', 'fast'
numeric_features = df[['slow', 'medium', 'fast']]
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns, index=df.index)

# Step 3: Combine normalized numeric features and location dummies with BroadcastName
features = pd.concat([df[['TeamName']], numeric_scaled, location_dummies], axis=1)

# Step 4: Group by 'BroadcastName' and compute mean
driver_features = features.groupby('TeamName').mean().T  # transpose for heatmap layout

# Step 5: Plot heatmap with values shown
plt.figure(figsize=(10, len(driver_features) * 0.5))
sns.heatmap(driver_features, cmap='coolwarm', linewidths=0.5,
            cbar_kws={'label': 'Average Value'}, annot=True, fmt='.2f')
plt.title('Teams vs Track Features and Locations')
plt.ylabel('Track Features and Locations')
plt.xlabel('Teams')
plt.tight_layout()
plt.savefig('ANALYTICS/TEAM_CORRELATION_WITH_TRACK_STRUCTURE.png')
plt.show()
