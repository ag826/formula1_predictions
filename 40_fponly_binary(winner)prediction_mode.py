import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import xgboost
import shap
from scipy.stats import randint, uniform
import pickle
import os
from datetime import datetime

# Import cupy for GPU array handling
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Function to move data to appropriate device
def prepare_data_for_device(X, device='cpu'):
    if device == 'cuda' and HAS_CUDA:
        return cp.array(X)
    return np.array(X)

# Helper functions for evaluation and plotting
def plot_and_save_metrics(y_true, y_pred, y_prob, set_name, output_dir):
    """Unified function to plot and save all evaluation metrics"""
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {set_name} Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    # Save before showing
    plt.savefig(f'{output_dir}/roc_curve_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {set_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # Save before showing
    plt.savefig(f'{output_dir}/confusion_matrix_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Return metrics for reporting
    return {
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred)
    }

# ========== 1. Load and Prepare Data ==========
df = pd.read_csv("PROCESSED_DATA/final_full_data_only_fp_predictors.csv")

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

# Handle missing values
# Replace deprecated fillna method with ffill and bfill
df = df.ffill().bfill()
# Ensure proper data types
df = df.infer_objects()

X = df.drop(columns=['IsWinnerFlag'])
y = df['IsWinnerFlag']

# ========== 2. Train/Val/Test Split ==========
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# ========== 3. Preprocessing Pipeline ==========
print("\nPreprocessing data...")

# Get categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("\nFeature counts:")
print(f"Total features in X: {X.shape[1]}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Create and fit the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('numerical', StandardScaler(), numerical_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# Fit preprocessor and transform data
X_train_enc = preprocessor.fit_transform(X_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# Generate clean feature names
categorical_feature_names = []
if len(categorical_cols) > 0:
    encoder = preprocessor.named_transformers_['categorical']
    for i, col in enumerate(categorical_cols):
        categories = encoder.categories_[i]
        categorical_feature_names.extend([f"{col}_{cat}" for cat in categories])
feature_names = categorical_feature_names + numerical_cols

# Move data to appropriate device
device = 'cuda' if HAS_CUDA else 'cpu'
X_train_enc = prepare_data_for_device(X_train_enc, device)
X_val_enc = prepare_data_for_device(X_val_enc, device)
X_test_enc = prepare_data_for_device(X_test_enc, device)

# ========== 4. Model Training with Hyperparameter Tuning ==========
print("\nCalculating class weights...")
n_negative = len(y_train[y_train == 0])
n_positive = len(y_train[y_train == 1])
scale_pos_weight = n_negative / n_positive

print(f"Class distribution - Negative: {n_negative}, Positive: {n_positive}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Define base model with GPU support if available
try:
    # Try to create a test XGBoost model with GPU
    test_model = XGBClassifier(device='cuda')
    test_model.fit(np.random.rand(10, 2), np.random.randint(2, size=10))
    gpu_available = True and HAS_CUDA
except Exception:
    gpu_available = False

base_model = XGBClassifier(
    device=device,
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# Define hyperparameter search space
param_dist = {
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 1000),
    'min_child_weight': randint(1, 7),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

# Random search with cross validation
print("\nPerforming hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings sampled
    scoring='roc_auc',
    n_jobs=-1,
    cv=5,
    verbose=2,
    random_state=42
)

random_search.fit(X_train_enc, y_train)

print("\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

# Train final model with best parameters
print("\nTraining final model with best parameters...")
best_model = XGBClassifier(
    **random_search.best_params_,
    device=device,
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    eval_metric=['auc', 'logloss'],
    random_state=42
)

# Train the model
print("\nTraining final model...")
best_model.fit(
    X_train_enc, 
    y_train,
    eval_set=[(X_val_enc, y_val)],
    verbose=True
)

# ========== 5. Model Evaluation ==========
print("\nEvaluating model performance...")

# Create MODEL_EVALUATION directory
os.makedirs('MODEL_EVALUATION', exist_ok=True)

# Evaluate all sets and collect metrics
all_metrics = {}
evaluation_sets = [
    (X_train_enc, y_train, "Training"),
    (X_val_enc, y_val, "Validation"),
    (X_test_enc, y_test, "Test")
]

for X, y, name in evaluation_sets:
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    all_metrics[name] = plot_and_save_metrics(y, y_pred, y_prob, name, 'MODEL_EVALUATION')

# ========== 6. Feature Importance Analysis ==========
print("\nAnalyzing feature importance...")

# Verify feature names match
n_features = len(best_model.feature_importances_)
if len(feature_names) != n_features:
    print(f"\nWarning: Feature name count mismatch!")
    print(f"Expected {n_features} features but got {len(feature_names)} names")
    feature_names = [f"feature_{i}" for i in range(n_features)]

# Create and save feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 30 features
n_top_features = min(30, len(feature_importance))
plt.figure(figsize=(15, 10))
plt.barh(range(n_top_features), feature_importance['importance'][:n_top_features])
plt.yticks(range(n_top_features), feature_importance['feature'][:n_top_features])
plt.xlabel('Feature Importance')
plt.title(f'Top {n_top_features} Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
# Save before showing
plt.savefig('MODEL_EVALUATION/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save feature importance data
feature_importance.to_csv('MODEL_EVALUATION/feature_importance.csv', index=False)

# ========== 7. SHAP Values Analysis ==========
print("\nCalculating SHAP values...")

# Convert to CPU if needed for SHAP analysis
X_train_enc_cpu = cp.asnumpy(X_train_enc) if device == 'cuda' else X_train_enc

# Calculate SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train_enc_cpu)

# Plot and save SHAP summary
plt.figure(figsize=(15, 10))
shap.summary_plot(shap_values, X_train_enc_cpu, 
                 feature_names=feature_names,
                 show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
# Save before showing
plt.savefig('MODEL_EVALUATION/shap_importance.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save model and preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save detailed metrics report
metrics_text = ["Model Performance Metrics", "=" * 50]
for dataset_name in ["Training", "Validation", "Test"]:
    metrics = all_metrics[dataset_name]
    metrics_text.extend([
        f"\n{dataset_name} Set Metrics:",
        "-" * 20,
        f"\nClassification Report:\n{metrics['classification_report']}",
        f"ROC AUC Score: {metrics['roc_auc']:.4f}",
        "\nConfusion Matrix:",
        "                 Predicted Negative  Predicted Positive",
        f"Actual Negative       {metrics['confusion_matrix'][0,0]}                {metrics['confusion_matrix'][0,1]}",
        f"Actual Positive       {metrics['confusion_matrix'][1,0]}                {metrics['confusion_matrix'][1,1]}"
    ])

with open('MODEL_EVALUATION/model_metrics.txt', 'w') as f:
    f.write('\n'.join(metrics_text))

print("\nAll results saved in MODEL_EVALUATION directory:")
print("- Feature importance plot and data")
print("- SHAP importance plot")
print("- ROC curves and confusion matrices for all sets")
print("- Detailed metrics report")
print("- Model and preprocessor pickle files")

