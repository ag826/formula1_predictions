# 🏎️ Formula 1 Predictions

This repository aims to analyze and predict Formula 1 race outcomes by leveraging historical data, weather conditions, tyre strategies, and driver performance metrics. The project is currently under active development, and contributions are welcome.

## 📌 Project Overview

This project focuses on:

- **Data Integration**: Combining race results, lap times, weather data, tyre information, and track characteristics.
- **Feature Engineering**: Creating meaningful features such as average sector times, pit stop durations, tyre performance metrics, and track status indicators.
- **Predictive Modeling**: Developing machine learning models to predict race outcomes and driver performances.
- **Visualization**: Generating insightful plots and correlation matrices to understand relationships between variables.

## 📊 Features and Engineering

The project performs extensive feature engineering, including:

- **Race Pace Analysis**: Calculating each driver's pace relative to the race winner.
- **Weather Aggregation**: Summarizing weather conditions like temperature, humidity, and wind speed.
- **Sector Times**: Computing average times for each track sector per driver.
- **Pit Stop Metrics**: Determining average pit stop durations and counts.
- **Tyre Performance**: Assessing tyre life, speed, and lap times across different compounds.
- **Track Status Events**: Counting occurrences of safety cars, virtual safety cars, and yellow/red flags.
- **Track Characteristics**: Analyzing corner types and distributions.

## 📈 Visualizations

The analysis includes generating:

- **Correlation Matrices**: Heatmaps showing correlations between numerical variables.
- **Target Variable Correlations**: Highlighting relationships between the target variable (e.g., finishing position) and other features.

## 📅 Session Schedule Processing

Session times are parsed and categorized into time-of-day segments (morning, afternoon, evening, night) to capture temporal effects on performance.

## 🛠️ Future Work

- **Model Development**: Implement machine learning models for predictive analysis.
- **Data Expansion**: Incorporate additional seasons and data sources.
- **Real-time Predictions**: Develop capabilities for live race predictions.
- **Web Interface**: Create a user-friendly dashboard for visualization and interaction.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

