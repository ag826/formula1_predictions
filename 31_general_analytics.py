# GRID POSITION DELTA NEEDS TO BE NORMALIZED BY NUMBER OF RACES COMPLETEDPER DRIVER


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
speed_cols = [f'RACE_AvgSpeedOnTyre_{comp}' for comp in tyre_compounds]

# Fill NaNs with 0 if needed
df[speed_cols] = df[speed_cols].fillna(0)

# Create a Race ID (e.g. "2023-5")
df['RaceID'] = df['RACEYEAR'].astype(str) + '-' + df['RACENUMBER'].astype(str)

# Group by race and take mean of the speeds for each tyre compound
grouped = df.groupby('RaceID')[speed_cols].mean().reset_index()

# Melt to long format for plotting
melted = grouped.melt(id_vars=['RaceID'], 
                      value_vars=speed_cols,
                      var_name='TyreCompound',
                      value_name='AvgTopSpeed')

# Clean TyreCompound column
melted['TyreCompound'] = melted['TyreCompound'].str.replace('RACE_AvgSpeedOnTyre_', '')

# Plotting
# plt.figure(figsize=(15, 6))

# sns.scatterplot(data=melted, x='RaceID', y='AvgTopSpeed', hue='TyreCompound', s=100)

# plt.xticks(rotation=90)
# plt.xlabel('Race (Year-RaceNumber)')
# plt.ylabel('Average Top Speed')
# plt.title('Average Top Speed by Tyre Compound for Each Race')
# plt.legend(title='Tyre Compound')
# plt.tight_layout()
# plt.show()


melted = melted.sort_values(['RaceID'])

# Get unique races and y positions
races = melted['RaceID'].unique()
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

# Plot dots for each tyre compound at y-position corresponding to the race
for i, race in enumerate(races):
    race_data = melted[melted['RaceID'] == race]
    for _, row in race_data.iterrows():
        ax.scatter(row['AvgTopSpeed'], i, 
                   color=color_map.get(row['TyreCompound'], 'black'), 
                   marker=marker_map.get(row['TyreCompound'], 'o'), 
                   s=100, 
                   label=row['TyreCompound'] if i == 0 else "")  # label only once for legend

    # Optional: draw lines connecting dots for this race
    speeds = race_data['AvgTopSpeed'].values
    ax.plot(speeds, [i]*len(speeds), color='gray', linestyle='--', alpha=0.5)

# Set y-ticks to race names
ax.set_yticks(y)
ax.set_yticklabels(races)

ax.set_xlabel('Average Top Speed')
ax.set_title('Average Top Speed per Tyre Compound Across Races Ordered by Year and Number')

# Handle legend without duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title='Tyre Compound')

plt.tight_layout()
plt.show()