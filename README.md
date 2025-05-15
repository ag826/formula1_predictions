# 🏎️ Formula 1 Predictions

This repository aims to analyze and predict Formula 1 race outcomes by leveraging historical data, weather conditions, tyre strategies, and driver performance metrics. The project uses machine learning to predict race winners based on practice session data.

## 📌 Project Overview

This project focuses on:

- **Data Integration**: Combining race results, lap times, weather data, tyre information, and track characteristics.
- **Feature Engineering**: Creating meaningful features such as average sector times, pit stop durations, tyre performance metrics, and track status indicators.
- **Machine Learning**: Using XGBoost for binary classification to predict race winners.
- **Model Evaluation**: Comprehensive evaluation using ROC curves, confusion matrices, and feature importance analysis.
- **Visualization**: Generating insightful plots and analysis of model performance.

## 🤖 Machine Learning Model

The project implements a binary classification model to predict race winners:

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **GPU Support**: Automatic detection and utilization of GPU if available
- **Class Balancing**: Implemented using `scale_pos_weight`
- **Hyperparameter Tuning**: RandomizedSearchCV with cross-validation

### Feature Processing
- **Categorical Features**: One-hot encoding with unknown category handling
- **Numerical Features**: Standard scaling
- **Missing Values**: Forward and backward filling strategies

### Model Evaluation
The model evaluation process includes:
- Training, validation, and test set performance metrics
- ROC curves and AUC scores
- Confusion matrices
- Feature importance analysis
- SHAP (SHapley Additive exPlanations) values


## 📊 Features and Engineering

The project performs extensive feature engineering, including:

- **Race Pace Analysis**: Calculating each driver's pace relative to the race winner.
- **Weather Aggregation**: Summarizing weather conditions like temperature, humidity, and wind speed.
- **Sector Times**: Computing average times for each track sector per driver.
- **Pit Stop Metrics**: Determining average pit stop durations and counts.
- **Tyre Performance**: Assessing tyre life, speed, and lap times across different compounds.
- **Track Status Events**: Counting occurrences of safety cars, virtual safety cars, and yellow/red flags.
- **Track Characteristics**: Analyzing corner types and distributions.


## 🛠️ Future Work

- **Model Improvements**: 
  - Experiment with different algorithms
  - Implement ensemble methods
  - Fine-tune hyperparameters further
- **Data Expansion**: Incorporate additional seasons and data sources
- **Real-time Predictions**: Develop capabilities for live race predictions
- **Web Interface**: Create a user-friendly dashboard for visualization and interaction
- **Feature Engineering**:
  - Develop more sophisticated track characteristic features
  - Create compound features from existing predictors
  - Incorporate historical performance trends

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📊 Model Performance

For detailed model performance metrics and visualizations, check the `MODEL_EVALUATION` directory after running the prediction script.

