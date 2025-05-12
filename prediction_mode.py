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
df = pd.read_csv("/kaggle/input/final-data/final_data.csv")
df=df[["DriverNumber","GridPosition","Status","RACEYEAR","RACENUMBER","IsWinnerFlag","RACE_AirTemp_mean","RACE_AirTemp_min","RACE_AirTemp_max","RACE_Humidity_mean","RACE_Humidity_min","RACE_Humidity_max","RACE_Pressure_mean","RACE_Pressure_min","RACE_Pressure_max","RACE_Rainfall_max","RACE_TrackTemp_mean","RACE_TrackTemp_min","RACE_TrackTemp_max","RACE_WindDirection_mean","RACE_WindSpeed_mean","RACE_WindSpeed_max","AvgPitStopDuration_ms","TotalPitStops","RACE_AvgTyreLife_HARD","RACE_AvgTyreLife_INTERMEDIATE","RACE_AvgTyreLife_MEDIUM","RACE_AvgTyreLife_SOFT","RACE_AvgTyreLife_UNKNOWN","RACE_AvgTyreLife_WET","RACE_AvgSpeedOnTyre_HARD","RACE_AvgSpeedOnTyre_INTERMEDIATE","RACE_AvgSpeedOnTyre_MEDIUM","RACE_AvgSpeedOnTyre_SOFT","RACE_AvgSpeedOnTyre_UNKNOWN","RACE_AvgSpeedOnTyre_WET","RACE_AvgLapTimeOnTyre_HARD","RACE_AvgLapTimeOnTyre_INTERMEDIATE","RACE_AvgLapTimeOnTyre_MEDIUM","RACE_AvgLapTimeOnTyre_SOFT","RACE_AvgLapTimeOnTyre_UNKNOWN","RACE_AvgLapTimeOnTyre_WET","RACE_Red","RACE_SCDeployed","RACE_VSCDeployed","RACE_Yellow","fast","medium","slow","TotalCorners"]]
X = df.drop(columns=['IsWinnerFlag'])
y = df['IsWinnerFlag']

# ========== 2. Train/Val/Test Split ==========
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

# ========== 3. Preprocessing Pipeline ==========
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

preprocessor.fit(X_train)
X_train_enc = preprocessor.transform(X_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# ========== 4. Train XGBoost ==========
scale_pos_weight = 873 / 46  # Class imbalance weight calculation

# Define model with updated parameters
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False,
    random_state=42
)

# Train the model
model.fit(
    X_train_enc, y_train,
    eval_set=[(X_train_enc, y_train), (X_val_enc, y_val)],
    eval_metric='logloss',
    verbose=True
)

# ========== 5. Feature Importance ==========
# Get feature importance
# Get feature importances and names
feature_importance = model.feature_importances_
feature_names = preprocessor.transformers_[0][1].get_feature_names_out(categorical_cols).tolist() + numerical_cols

# Combine and sort
importances_with_names = sorted(zip(feature_importance, feature_names), reverse=True)[:30]
top_importance, top_names = zip(*importances_with_names)

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_importance)), top_importance, align='center')
plt.yticks(range(len(top_names)), top_names)
plt.gca().invert_yaxis()  # highest importance on top
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances from XGBoost Model')
plt.tight_layout()
plt.show()
# ========== 6. Evaluation ==========
y_test_pred = model.predict(X_test_enc)
y_test_proba = model.predict_proba(X_test_enc)[:, 1]

# Classification Report
print("Classification Report on Test Set:\n")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
