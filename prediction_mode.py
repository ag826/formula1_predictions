
#### DROP HIGHLY CORRELATED COLUMNS


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
feature_importance = model.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 20))
plt.barh(range(len(feature_importance)), feature_importance, align='center')
plt.yticks(range(len(feature_importance)), preprocessor.transformers_[0][1].get_feature_names_out(categorical_cols).tolist() + numerical_cols)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from XGBoost Model')
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
