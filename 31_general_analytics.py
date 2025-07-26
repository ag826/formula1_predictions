# GRID POSITION DELTA NEEDS TO BE NORMALIZED BY NUMBER OF RACES COMPLETEDPER DRIVER


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import colorsys  # For HSL color manipulation
from matplotlib.colors import to_rgb  # To convert matplotlib colors to RGB tuple

# Load the data
df = pd.read_csv("PROCESSED_DATA/final_full_data.csv")

##################################################################################################################
# DRIVER POINTS OVER TIME - CUMULATIVE
##################################################################################################################

# Filter necessary columns
df_points = df[["RACEYEAR", "RACENUMBER", "DriverNameMapped", "Points"]].copy()

# Sort by year, race number
df_points = df_points.sort_values(["DriverNameMapped", "RACEYEAR", "RACENUMBER"])

# Calculate cumulative points per driver
df_points["CumulativePoints"] = df_points.groupby("DriverNameMapped")["Points"].cumsum()

# Create a combined race identifier for x-axis
df_points["Race"] = (
    df_points["RACEYEAR"].astype(str) + "-" + df_points["RACENUMBER"].astype(str)
)

# Pivot for plotting: rows = race, columns = driver, values = cumulative points
pivot = df_points.pivot_table(
    index="Race", columns="DriverNameMapped", values="CumulativePoints"
)

# Extract year and race number as integers
race_index_df = pivot.index.to_series().str.split("-", expand=True).astype(int)
race_index_df.columns = ["Year", "RaceNum"]
# Get sorted order
sorted_order = race_index_df.sort_values(["Year", "RaceNum"]).index
# Reindex pivot table
pivot = pivot.loc[sorted_order]

# Plot
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Use colorblind-friendly colormap
colors = plt.cm.tab20.colors  # This is a colorblind-friendly colormap
for i, (driver, data) in enumerate(pivot.items()):
    ax.plot(
        range(len(data.index)),
        data.values,
        marker="o",
        label=driver,
        color=colors[i % len(colors)],
        markersize=3,
    )

# Set x-tick labels to show (Year, RaceNumber)
ax.set_xticks(range(len(pivot.index)))
ax.set_xticklabels(pivot.index, rotation=90)

plt.title("Cumulative Points per Driver (by Race)")
plt.xlabel("Race (Year-Number)")
plt.ylabel("Cumulative Points")
plt.legend(title="Driver", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("ANALYTICS/DRIVER_POINTS_OVER_TIME.png")
plt.show()


##################################################################################################################
# DRIVER POINTS OVER TIME - BROKEN BY YEAR
##################################################################################################################


# --- DRIVER COLORS ---
drivers = sorted(df["DriverNameMapped"].dropna().unique())
driver_color_map = {driver: plt.cm.tab20(i % 20) for i, driver in enumerate(drivers)}

years = sorted(df["RACEYEAR"].unique())

# --- DRIVER POINTS ---
df_points = df[["RACEYEAR", "RACENUMBER", "DriverNameMapped", "Points"]].copy()
df_points = df_points.sort_values(["DriverNameMapped", "RACEYEAR", "RACENUMBER"])
df_points["CumulativePoints"] = df_points.groupby(["DriverNameMapped", "RACEYEAR"])[
    "Points"
].cumsum()

# --- Build shared x-axis: one entry per race, with year and race number ---
race_tuples = sorted(df[["RACEYEAR", "RACENUMBER"]].drop_duplicates().values.tolist())
race_labels = [f"{year}-{race}" for year, race in race_tuples]
race_index_map = {(year, race): idx for idx, (year, race) in enumerate(race_tuples)}

# --- Find year boundaries for vertical lines ---
year_boundaries = []
prev_year = race_tuples[0][0]
for idx, (year, race) in enumerate(race_tuples):
    if year != prev_year:
        year_boundaries.append(idx)
        prev_year = year

fig, ax_driver = plt.subplots(1, 1, figsize=(18, 8), sharex=True)

# --- Plotting Driver Lines, Avoiding Duplicate Labels ---
plotted_drivers = set()

for driver in drivers:
    x_points = []
    y_points = []
    for year, race in race_tuples:
        sub = df_points[
            (df_points["DriverNameMapped"] == driver)
            & (df_points["RACEYEAR"] == year)
            & (df_points["RACENUMBER"] == race)
        ]
        if not sub.empty:
            x_points.append(race_index_map[(year, race)])
            y_points.append(sub.iloc[0]["CumulativePoints"])
    # Split by year boundaries
    boundaries = [0] + year_boundaries + [len(race_tuples)]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        indices = [j for j, x in enumerate(x_points) if start <= x < end]
        if indices:
            seg_x = [x_points[j] for j in indices]
            seg_y = [y_points[j] for j in indices]
            ax_driver.plot(
                seg_x,
                seg_y,
                marker="o",
                label=driver if driver not in plotted_drivers else None,
                color=driver_color_map[driver],
                linewidth=2,
                markersize=4,
            )
    plotted_drivers.add(driver)

# --- Axis & Title ---
ax_driver.set_ylabel("Cumulative Points (Driver)")
ax_driver.set_title("Driver Points Over Time (Points Reset Each Year)")
ax_driver.set_xticks(range(len(race_labels)))
ax_driver.set_xticklabels(race_labels, rotation=90)

# --- Draw vertical lines for year boundaries ---
for boundary in year_boundaries:
    ax_driver.axvline(
        boundary - 0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7
    )
    ax_driver.text(
        boundary,
        ax_driver.get_ylim()[1],
        f"{race_tuples[boundary][0]}",
        color="black",
        fontsize=12,
        ha="center",
        va="bottom",
    )

# --- Clean legend (unique and sorted alphabetically) ---
handles, labels = ax_driver.get_legend_handles_labels()
by_label = dict(sorted(zip(labels, handles), key=lambda x: (x[0] is None, x[0])))
ax_driver.legend(
    by_label.values(),
    by_label.keys(),
    title="Driver",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=8,
)

plt.xlabel("Race (Year-Number)")
plt.tight_layout()
plt.savefig("ANALYTICS/DRIVER_POINTS_OVER_TIME_BY_YEAR.png")
plt.show()

##################################################################################################################
# TEAM POINTS OVER TIME - CUMULATIVE
##################################################################################################################

# Filter necessary columns for team points
df_team = df[["RACEYEAR", "RACENUMBER", "TeamNameMapped", "Points"]].copy()

# Sort by year, race number
df_team = df_team.sort_values(["TeamNameMapped", "RACEYEAR", "RACENUMBER"])

# Calculate cumulative points per team
df_team["CumulativePoints"] = df_team.groupby("TeamNameMapped")["Points"].cumsum()

# Create a combined race identifier for x-axis
df_team["Race"] = (
    df_team["RACEYEAR"].astype(str) + "-" + df_team["RACENUMBER"].astype(str)
)

# Pivot for plotting: rows = race, columns = team, values = cumulative points
pivot_team = df_team.pivot_table(
    index="Race", columns="TeamNameMapped", values="CumulativePoints"
)

# Extract year and race number as integers
race_index_df_team = (
    pivot_team.index.to_series().str.split("-", expand=True).astype(int)
)
race_index_df_team.columns = ["Year", "RaceNum"]
sorted_order_team = race_index_df_team.sort_values(["Year", "RaceNum"]).index
pivot_team = pivot_team.loc[sorted_order_team]

# Plot
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Use colorblind-friendly colormap
colors = plt.cm.tab20.colors  # This is a colorblind-friendly colormap
for i, (team, data) in enumerate(pivot_team.items()):
    ax.plot(
        range(len(data.index)),
        data.values,
        marker="o",
        label=team,
        color=colors[i % len(colors)],
        markersize=3,
    )

# Set x-tick labels to show (Year, RaceNumber)
ax.set_xticks(range(len(pivot_team.index)))
ax.set_xticklabels(pivot_team.index, rotation=90)

plt.title("Cumulative Points per Team (by Race)")
plt.xlabel("Race (Year-Number)")
plt.ylabel("Cumulative Points")
plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("ANALYTICS/TEAM_POINTS_OVER_TIME.png")
plt.show()


##################################################################################################################
# TEAM POINTS OVER TIME - BROKEN BY YEAR
##################################################################################################################

# --- TEAM COLORS ---
teams = sorted(df["TeamNameMapped"].dropna().unique())
team_color_map = {team: plt.cm.tab20(i % 20) for i, team in enumerate(teams)}

years = sorted(df["RACEYEAR"].unique())

# --- TEAM POINTS ---
df_team_points = df[["RACEYEAR", "RACENUMBER", "TeamNameMapped", "Points"]].copy()
df_team_points = (
    df_team_points.groupby(["RACEYEAR", "RACENUMBER", "TeamNameMapped"])
    .sum()
    .reset_index()
)
df_team_points = df_team_points.sort_values(
    ["TeamNameMapped", "RACEYEAR", "RACENUMBER"]
)
df_team_points["CumulativePoints"] = df_team_points.groupby(
    ["TeamNameMapped", "RACEYEAR"]
)["Points"].cumsum()

# --- Build shared x-axis: one entry per race, with year and race number ---
race_tuples = sorted(df[["RACEYEAR", "RACENUMBER"]].drop_duplicates().values.tolist())
race_labels = [f"{year}-{race}" for year, race in race_tuples]
race_index_map = {(year, race): idx for idx, (year, race) in enumerate(race_tuples)}

# --- Find year boundaries for vertical lines ---
year_boundaries = []
prev_year = race_tuples[0][0]
for idx, (year, race) in enumerate(race_tuples):
    if year != prev_year:
        year_boundaries.append(idx)
        prev_year = year

fig, ax_team = plt.subplots(1, 1, figsize=(18, 8), sharex=True)

# --- Plotting Team Lines, Avoiding Duplicate Labels ---
plotted_teams = set()

for team in teams:
    x_points = []
    y_points = []
    for year, race in race_tuples:
        sub = df_team_points[
            (df_team_points["TeamNameMapped"] == team)
            & (df_team_points["RACEYEAR"] == year)
            & (df_team_points["RACENUMBER"] == race)
        ]
        if not sub.empty:
            x_points.append(race_index_map[(year, race)])
            y_points.append(sub.iloc[0]["CumulativePoints"])
    # Split by year boundaries
    boundaries = [0] + year_boundaries + [len(race_tuples)]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        indices = [j for j, x in enumerate(x_points) if start <= x < end]
        if indices:
            seg_x = [x_points[j] for j in indices]
            seg_y = [y_points[j] for j in indices]
            ax_team.plot(
                seg_x,
                seg_y,
                marker="o",
                label=team if team not in plotted_teams else None,
                color=team_color_map[team],
                linewidth=2,
                markersize=4,
            )
    plotted_teams.add(team)

# --- Axis & Title ---
ax_team.set_ylabel("Cumulative Points (Team)")
ax_team.set_title("Team Points Over Time (Points Reset Each Year)")
ax_team.set_xticks(range(len(race_labels)))
ax_team.set_xticklabels(race_labels, rotation=90)

# --- Draw vertical lines for year boundaries ---
for boundary in year_boundaries:
    ax_team.axvline(
        boundary - 0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7
    )
    ax_team.text(
        boundary,
        ax_team.get_ylim()[1],
        f"{race_tuples[boundary][0]}",
        color="black",
        fontsize=12,
        ha="center",
        va="bottom",
    )

# --- Clean legend (unique and sorted alphabetically) ---
handles, labels = ax_team.get_legend_handles_labels()
by_label = dict(sorted(zip(labels, handles), key=lambda x: (x[0] is None, x[0])))
ax_team.legend(
    by_label.values(),
    by_label.keys(),
    title="Team",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=8,
)

plt.xlabel("Race (Year-Number)")
plt.tight_layout()
plt.savefig("ANALYTICS/TEAM_POINTS_OVER_TIME_BY_YEAR.png")
plt.show()

##################################################################################################################
# DRIVER: AVERAGE GRID POSITION VS FINAL POSITION
##################################################################################################################

df_avg = df[["DriverNameMapped", "GridPosition", "Position"]].copy()

# Remove rows with missing positions
df_avg = df_avg.dropna(subset=["GridPosition", "Position"])

# Group by driver and calculate mean positions
avg_positions = (
    df_avg.groupby(["DriverNameMapped"])
    .agg({"GridPosition": "mean", "Position": "mean"})
    .reset_index()
)

# Sort by average race position (lower is better)
avg_positions = avg_positions.sort_values("GridPosition")

fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(avg_positions))

# Plot grid position and race position as dots
ax.scatter(
    avg_positions["GridPosition"],
    y,
    color="tab:blue",
    label="Avg Grid Position",
    s=100,
    marker="o",
)
ax.scatter(
    avg_positions["Position"],
    y,
    color="tab:orange",
    label="Avg Race Position",
    s=100,
    marker="D",
)

# Draw lines between grid and race position for each driver
for i, row in enumerate(avg_positions.itertuples()):
    ax.plot(
        [row.GridPosition, row.Position],
        [i, i],
        color="gray",
        linestyle="--",
        alpha=0.7,
    )

# Add vertical lines at x=10 and x=3
ax.axvline(x=10, color="blue", linestyle=":", linewidth=2, label="x=10")
ax.axvline(x=3, color="orange", linestyle=":", linewidth=2, label="x=3")

ax.set_yticks(y)
ax.set_yticklabels(avg_positions["DriverNameMapped"])
ax.set_xlabel("Average Position (Lower is Better)")
ax.set_title("Average Grid (Qualifying) vs Final Race Position per Driver")
ax.legend()
ax.invert_xaxis()  # Lower (better) positions to the right
plt.tight_layout()
plt.savefig("ANALYTICS/DRIVER_AVG_GRID_VS_FINAL_POSITION_DOTPLOT.png")
plt.show()

##################################################################################################################
# Average top speed on tyre compounds per race
##################################################################################################################

tyre_compounds = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]
speed_cols = [f"RACE_FastestSpeedOnTyre_{comp}" for comp in tyre_compounds]

# Replace 0s with NaN so they're excluded from mean calculation
df[speed_cols] = df[speed_cols].replace(0, np.nan)

# Group by race and calculate mean only for non-NaN (non-zero) values
grouped = df.groupby("Location")[speed_cols].mean().reset_index()

# Melt to long format for plotting
melted = grouped.melt(
    id_vars=["Location"],
    value_vars=speed_cols,
    var_name="TyreCompound",
    value_name="FastestTopSpeed",
)

# Clean TyreCompound column
melted["TyreCompound"] = melted["TyreCompound"].str.replace(
    "RACE_FastestSpeedOnTyre_", ""
)

# Remove rows where FastestTopSpeed is NaN (no valid data for that compound in that race)
melted = melted.dropna(subset=["FastestTopSpeed"])

# Sort by speed range (largest difference between fastest and slowest compound per location)
race_ranges = (
    melted.groupby("Location")["FastestTopSpeed"]
    .agg(lambda x: x.max() - x.min())
    .sort_values(ascending=False)
)
races = race_ranges.index.tolist()

# Reorder melted data according to the sorted races
melted["Location"] = pd.Categorical(melted["Location"], categories=races, ordered=True)
melted = melted.sort_values("Location")

# Get y positions for the sorted races
y = np.arange(len(races))

# Prepare color and marker maps for tyre compounds
color_map = {
    "HARD": "tab:blue",
    "MEDIUM": "tab:orange",
    "SOFT": "tab:green",
    "INTERMEDIATE": "tab:red",
    "WET": "tab:purple",
}

marker_map = {"HARD": "o", "MEDIUM": "D", "SOFT": "s", "INTERMEDIATE": "^", "WET": "X"}

fig, ax = plt.subplots(figsize=(14, 10))

labeled_compounds = set()

# Plot dots for each tyre compound at y-position corresponding to the race
for i, race in enumerate(races):
    race_data = melted[melted["Location"] == race]
    for _, row in race_data.iterrows():
        compound = row["TyreCompound"]
        # Only add label if this compound hasn't been labeled yet
        label = compound if compound not in labeled_compounds else ""
        if compound not in labeled_compounds:
            labeled_compounds.add(compound)

        ax.scatter(
            row["FastestTopSpeed"],
            i,
            color=color_map.get(compound, "black"),
            marker=marker_map.get(compound, "o"),
            s=100,
            label=label,
        )

    # Optional: draw lines connecting dots for this race
    speeds = race_data["FastestTopSpeed"].values
    ax.plot(speeds, [i] * len(speeds), color="gray", linestyle="--", alpha=0.5)

# Set y-ticks to race names
ax.set_yticks(y)
ax.set_yticklabels(races)

ax.set_xlabel("Fastest Top Speed")
ax.set_title(
    "Fastest Top Speed per Tyre Compound Across Locations (Sorted by Speed Range)"
)

# Handle legend without duplicates
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title="Tyre Compound")

plt.tight_layout()
plt.savefig(
    "ANALYTICS/FASTEST_TOP_SPEED_PER_TYRE_COMPOUND_ACROSS_LOCATIONS.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Optional: Print the speed ranges for analysis
print("\nSpeed Ranges by Location (largest to smallest):")
print("=" * 50)
for location in races:
    location_data = melted[melted["Location"] == location]
    speed_range = (
        location_data["FastestTopSpeed"].max() - location_data["FastestTopSpeed"].min()
    )
    min_speed = location_data["FastestTopSpeed"].min()
    max_speed = location_data["FastestTopSpeed"].max()
    print(
        f"{location:<20} | Range: {speed_range:6.1f} km/h | Min: {min_speed:6.1f} | Max: {max_speed:6.1f}"
    )


##################################################################################################################
# Weather and tyre compound correlation
##################################################################################################################

x_cols = [
    "RACE_AirTemp_mean",
    "RACE_AirTemp_min",
    "RACE_AirTemp_max",
    "RACE_Humidity_mean",
    "RACE_Humidity_min",
    "RACE_Humidity_max",
    "RACE_Pressure_mean",
    "RACE_Pressure_min",
    "RACE_Pressure_max",
    "RACE_TrackTemp_mean",
    "RACE_TrackTemp_min",
    "RACE_TrackTemp_max",
    "RACE_WindDirection_mean",
    "RACE_WindSpeed_mean",
    "RACE_WindSpeed_max",
]

y_cols = [
    "RACE_TotalPitStops",
    "RACE_MaxStint_SOFT",
    "RACE_AvgTyreLife_SOFT",
    "RACE_AvgSpeedOnTyre_SOFT",
    "RACE_FastestSpeedOnTyre_SOFT",
    "RACE_AvgLapTimeOnTyre_SOFT",
    "RACE_FastestLapTimeOnTyre_SOFT",
    "RACE_MaxStint_MEDIUM",
    "RACE_AvgTyreLife_MEDIUM",
    "RACE_AvgSpeedOnTyre_MEDIUM",
    "RACE_FastestSpeedOnTyre_MEDIUM",
    "RACE_AvgLapTimeOnTyre_MEDIUM",
    "RACE_FastestLapTimeOnTyre_MEDIUM",
    "RACE_MaxStint_HARD",
    "RACE_AvgTyreLife_HARD",
    "RACE_AvgSpeedOnTyre_HARD",
    "RACE_FastestSpeedOnTyre_HARD",
    "RACE_AvgLapTimeOnTyre_HARD",
    "RACE_FastestLapTimeOnTyre_HARD",
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
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Selected Weather Features and Tyre Performance")
plt.tight_layout()
plt.savefig("ANALYTICS/WEATHER_AND_TYRE_COMPOUND_CORRELATION.png")
plt.show()

##################################################################################################################
# RACE INCIDENTS PER TRACK
##################################################################################################################

# Step 1: Group by Location and calculate averages
avg_flags_by_location = df.groupby("Location")[
    ["RACE_Red", "RACE_SCDeployed", "RACE_VSCDeployed", "RACE_Yellow"]
].mean()

# Step 2: Count number of races per location
race_counts = (df["Location"].value_counts().rename("RaceCount")) / 20

# Step 3: Merge average flags with race counts
avg_flags_by_location = avg_flags_by_location.merge(
    race_counts, left_index=True, right_index=True
)

# Step 4: Create a 'Total' column for sorting
avg_flags_by_location["Total"] = avg_flags_by_location[
    ["RACE_Red", "RACE_SCDeployed", "RACE_VSCDeployed", "RACE_Yellow"]
].sum(axis=1)

# Step 5: Sort by total incidents
avg_flags_by_location = avg_flags_by_location.sort_values("Total", ascending=False)

# Step 6: Plotting
fig, ax = plt.subplots(figsize=(15, 10))

# Stacked bar components
ax.bar(
    avg_flags_by_location.index,
    avg_flags_by_location["RACE_Red"],
    label="Red Flags",
    color="tab:red",
)
ax.bar(
    avg_flags_by_location.index,
    avg_flags_by_location["RACE_Yellow"],
    bottom=avg_flags_by_location["RACE_Red"],
    label="Yellow Flags",
    color="gold",
)
ax.bar(
    avg_flags_by_location.index,
    avg_flags_by_location["RACE_SCDeployed"],
    bottom=avg_flags_by_location["RACE_Red"] + avg_flags_by_location["RACE_Yellow"],
    label="Safety Car",
    color="gray",
)
ax.bar(
    avg_flags_by_location.index,
    avg_flags_by_location["RACE_VSCDeployed"],
    bottom=avg_flags_by_location["RACE_Red"]
    + avg_flags_by_location["RACE_Yellow"]
    + avg_flags_by_location["RACE_SCDeployed"],
    label="Virtual Safety Car",
    color="tab:blue",
)

# Step 7: Add total race count as annotation
for i, (loc, row) in enumerate(avg_flags_by_location.iterrows()):
    ax.text(
        i,
        row["Total"] + 0.1,
        f"{int(row['RaceCount'])} races",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=90,
    )

# Final touches
ax.set_title("Average Incidents per Race by Location (with Race Counts)", fontsize=16)
ax.set_ylabel("Average Incident Count per Race")
ax.set_xticks(range(len(avg_flags_by_location)))
ax.set_xticklabels(avg_flags_by_location.index, rotation=45, ha="right")
ax.legend(title="Incident Type")
max_y = avg_flags_by_location["Total"].max()
ax.set_ylim(0, max_y + 1)

plt.tight_layout()
plt.savefig("ANALYTICS/RACE_INCIDENTS_PER_TRACK.png")
plt.show()

##################################################################################################################
# TRACK STRUCTURE
##################################################################################################################

# Get distinct track corner data
track_structure = df[["Location", "fast", "medium", "slow"]].drop_duplicates()
track_structure = track_structure.sort_values(by="fast").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15, 10))

# Stacked bar chart
ax.bar(
    range(len(track_structure)),
    track_structure["fast"],
    label="Fast Corners (* < 60)",
    color="#88CCEE",
)  # Muted Blue
ax.bar(
    range(len(track_structure)),
    track_structure["medium"],
    bottom=track_structure["fast"],
    label="Medium Corners (60 < * < 90)",
    color="#DDCC77",
)  # Muted Orange
ax.bar(
    range(len(track_structure)),
    track_structure["slow"],
    bottom=track_structure["fast"] + track_structure["medium"],
    label="Slow Corners (* > 90)",
    color="#CC6677",
)  # Muted Red

# Labels and final formatting
ax.set_title("Corner Type Distribution per Track", fontsize=16)
ax.set_ylabel("Number of Corners")
ax.set_xticks(range(len(track_structure)))
ax.set_xticklabels(track_structure["Location"], rotation=45, ha="right")
ax.legend(title="Corner Type")
ax.set_ylim(0, (track_structure[["fast", "medium", "slow"]].sum(axis=1).max()) + 2)

plt.tight_layout()
plt.savefig("ANALYTICS/TRACK_STRUCTURE.png")
plt.show()

##################################################################################################################
# DRIVER CORRELATION WITH TRACK STRUCTURE
##################################################################################################################

# Step 1: One-hot encode 'Location' column

location_dummies = pd.get_dummies(df["Location"], prefix="Loc")

# Step 2: Normalize numeric features 'slow', 'medium', 'fast'
numeric_features = df[["slow", "medium", "fast"]]
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_features),
    columns=numeric_features.columns,
    index=df.index,
)

# Step 3: Combine normalized numeric features and location dummies with BroadcastName
features = pd.concat(
    [df[["DriverNameMapped"]], numeric_scaled, location_dummies], axis=1
)

# Step 4: Group by 'BroadcastName' and compute mean
driver_features = (
    features.groupby("DriverNameMapped").mean().T
)  # transpose for heatmap layout

# Step 5: Plot heatmap with values shown
plt.figure(figsize=(20, len(driver_features) * 0.5))
sns.heatmap(
    driver_features,
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"label": "Average Value"},
    annot=True,
    fmt=".2f",
)
plt.title("Drivers vs Track Features and Locations")
plt.ylabel("Track Features and Locations")
plt.xlabel("Drivers")
plt.tight_layout()
plt.savefig("ANALYTICS/DRIVER_CORRELATION_WITH_TRACK_STRUCTURE.png")
plt.show()

##################################################################################################################
# TEAM CORRELATION WITH TRACK STRUCTURE
##################################################################################################################

location_dummies = pd.get_dummies(df["Location"], prefix="Loc")

# Step 2: Normalize numeric features 'slow', 'medium', 'fast'
numeric_features = df[["slow", "medium", "fast"]]
scaler = MinMaxScaler()
numeric_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_features),
    columns=numeric_features.columns,
    index=df.index,
)

# Step 3: Combine normalized numeric features and location dummies with BroadcastName
features = pd.concat([df[["TeamNameMapped"]], numeric_scaled, location_dummies], axis=1)

# Step 4: Group by 'BroadcastName' and compute mean
driver_features = (
    features.groupby("TeamNameMapped").mean().T
)  # transpose for heatmap layout

# Step 5: Plot heatmap with values shown
plt.figure(figsize=(10, len(driver_features) * 0.5))
sns.heatmap(
    driver_features,
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"label": "Average Value"},
    annot=True,
    fmt=".2f",
)
plt.title("Teams vs Track Features and Locations")
plt.ylabel("Track Features and Locations")
plt.xlabel("Teams")
plt.tight_layout()
plt.savefig("ANALYTICS/TEAM_CORRELATION_WITH_TRACK_STRUCTURE.png")
plt.show()


##################################################################################################################
# AVERAGE PIT STOP BY TEAM
##################################################################################################################

# --- Filter and transform pit stop data ---
pitstop_df_filtered = df[
    (df["RACE_AvgPitStopDuration_ms"] < 55000) & (df["Status"] == "Finished")
].copy()
pitstop_df_filtered["PitStopDuration_s"] = (
    pitstop_df_filtered["RACE_AvgPitStopDuration_ms"] / 1000
)

# --- Average pit stop by driver ---
avg_pitstop_by_driver = (
    pitstop_df_filtered.groupby(["TeamNameMapped", "Abbreviation"])["PitStopDuration_s"]
    .mean()
    .reset_index()
)

# --- Average by team for sorting ---
avg_pitstop_by_team_for_sort = (
    avg_pitstop_by_driver.groupby("TeamNameMapped")["PitStopDuration_s"]
    .mean()
    .sort_values(ascending=False)
)
sorted_teams = avg_pitstop_by_team_for_sort.index.tolist()

# --- Map team names to positions ---
team_position_map = {team: i for i, team in enumerate(sorted_teams)}
avg_pitstop_by_driver["TeamPosition"] = avg_pitstop_by_driver["TeamNameMapped"].map(
    team_position_map
)
avg_pitstop_by_driver["DriverOffset"] = 0
avg_pitstop_by_driver["PlotPosition"] = avg_pitstop_by_driver["TeamPosition"]

# --- Team color mapping ---
team_base_colors_cmap = plt.cm.get_cmap("tab10", len(sorted_teams))
team_color_map = {team: team_base_colors_cmap(i) for i, team in enumerate(sorted_teams)}

# --- Driver color shades ---
driver_color_map = {}
for team in sorted_teams:
    drivers = sorted(
        avg_pitstop_by_driver[avg_pitstop_by_driver["TeamNameMapped"] == team][
            "Abbreviation"
        ].unique()
    )
    num_drivers = len(drivers)

    base_rgb = to_rgb(team_color_map[team])
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    lightness_values = (
        np.linspace(max(0.1, l - 0.2), min(0.9, l + 0.2), num_drivers)
        if num_drivers > 1
        else [l]
    )

    for i, abbr in enumerate(drivers):
        new_rgb = colorsys.hls_to_rgb(h, lightness_values[i], s)
        driver_color_map[abbr] = new_rgb

# --- Plot setup ---
plt.figure(figsize=(18, 12))
ax = plt.gca()

# --- Reuse the team sort order and averages for background bars ---
team_avg_durations_for_bars = avg_pitstop_by_team_for_sort.reset_index()
team_avg_durations_for_bars.columns = ["TeamNameMapped", "PitStopDuration_s"]
team_avg_durations_for_bars["TeamPosition"] = team_avg_durations_for_bars[
    "TeamNameMapped"
].map(team_position_map)

# --- Background bars ---
for _, row in team_avg_durations_for_bars.iterrows():
    base_rgb = to_rgb(team_color_map[row["TeamNameMapped"]])
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    bar_color = colorsys.hls_to_rgb(h, l + 0.3 * (1 - l), s)

    ax.barh(
        row["TeamPosition"],
        row["PitStopDuration_s"],
        height=0.8,
        color=bar_color,
        alpha=0.3,
        zorder=1,
    )

# --- Plot driver dots and names ---
text_offset_y = 0.1
vertical_text_stack_step = 0.15
text_label_registry = []

for _, row in avg_pitstop_by_driver.iterrows():
    dot_color = driver_color_map.get(row["Abbreviation"], "gray")
    ax.scatter(
        row["PitStopDuration_s"],
        row["PlotPosition"],
        color=dot_color,
        s=200,
        label=row["Abbreviation"],
        zorder=2,
    )

    # Handle text overlap
    text_x = row["PitStopDuration_s"]
    text_y = row["PlotPosition"] + text_offset_y
    text_width = len(row["Abbreviation"]) * 0.02

    overlaps = [
        info
        for info in text_label_registry
        if (abs(info[1] - text_x) < text_width)
        and (abs(info[2] - text_y) < vertical_text_stack_step)
    ]
    text_y += len(overlaps) * vertical_text_stack_step

    ax.text(
        text_x,
        text_y,
        row["Abbreviation"],
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        clip_on=True,
        zorder=3,
    )

    text_label_registry.append((row["Abbreviation"], text_x, text_y))

# --- Y-axis: team names ---
plt.yticks(
    [team_position_map[team] for team in sorted_teams], sorted_teams, fontsize=10
)

# --- Axis labels and grid ---
plt.xlabel("Average Pit Stop Duration (seconds)", fontsize=12)
plt.ylabel("Team Name", fontsize=12)
plt.title(
    "Average Pit Stop Duration by Team and Driver (Pitstops < 55,000 ms)", fontsize=14
)

# --- Dynamic x-axis limits ---
x_min = avg_pitstop_by_driver["PitStopDuration_s"].min() - 0.2
x_max = avg_pitstop_by_driver["PitStopDuration_s"].max() + 0.2
plt.xlim(x_min, x_max)

# --- Driver legend (deduplicated) ---
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(
    unique_labels.values(),
    unique_labels.keys(),
    title="Driver",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10,
)

# --- Save and show ---
plt.tight_layout()
plt.savefig("ANALYTICS/AVG_PITSTOP_DURATION_BY_TEAM_AND_DRIVER_DOT_PLOT.png")
plt.show()

##################################################################################################################
# WEATHER, HYMIDITY AND TYRE TEMPERATURE
##################################################################################################################

# Group by Location and calculate mean for each feature
weather_means = (
    df.groupby("Location")[
        ["RACE_AirTemp_mean", "RACE_Humidity_mean", "RACE_TrackTemp_mean"]
    ]
    .mean()
    .reset_index()
)
weather_means["total"] = (
    weather_means["RACE_AirTemp_mean"]
    + weather_means["RACE_Humidity_mean"]
    + weather_means["RACE_TrackTemp_mean"]
)


# Sort tracks by air temp for better visualization (optional)
weather_means = weather_means.sort_values("total", ascending=False)

fig, ax = plt.subplots(figsize=(16, 10))

bar_width = 0.6
x = np.arange(len(weather_means))

# Plot stacked bars
ax.bar(
    x,
    weather_means["RACE_AirTemp_mean"],
    width=bar_width,
    label="Avg. Air Temp (°C)",
    color="tab:blue",
)
ax.bar(
    x,
    weather_means["RACE_Humidity_mean"],
    width=bar_width,
    bottom=weather_means["RACE_AirTemp_mean"],
    label="Avg. Humidity (%)",
    color="tab:orange",
)
ax.bar(
    x,
    weather_means["RACE_TrackTemp_mean"],
    width=bar_width,
    bottom=weather_means["RACE_AirTemp_mean"] + weather_means["RACE_Humidity_mean"],
    label="Avg. Track Temp (°C)",
    color="tab:green",
)

ax.set_xticks(x)
ax.set_xticklabels(weather_means["Location"], rotation=45, ha="right")
ax.set_ylabel("Mean Value")
ax.set_title("Mean Air Temp, Humidity, and Track Temp per Track (Stacked Bar)")
ax.legend(title="Feature")
plt.tight_layout()
plt.savefig("ANALYTICS/AVG_WEATHER_TRACK_TEMP_HUMIDITY.png")
plt.show()

##################################################################################################################
