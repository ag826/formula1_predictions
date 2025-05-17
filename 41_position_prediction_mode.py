import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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
def plot_and_save_metrics(y_true, y_pred, y_prob, set_name, output_dir, label_encoder):
    """Unified function to plot and save all evaluation metrics"""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {set_name} Set')
    plt.ylabel('True Position')
    plt.xlabel('Predicted Position')
    plt.tight_layout()
    # Save before showing
    plt.savefig(f'{output_dir}/confusion_matrix_{set_name.lower()}_position_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Calculate per-class ROC curves
    n_classes = len(label_encoder.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Only plot ROC curves for positions 1-5 to avoid overcrowding
        if i < 5:
            plt.plot(fpr[i], tpr[i],
                    label=f'Position {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Top-5 Positions ROC Curves - {set_name} Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    # Save before showing
    plt.savefig(f'{output_dir}/roc_curves_{set_name.lower()}_position_prediction.png', dpi=300, bbox_inches='tight')
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

# Select relevant columns (same as before but replace IsWinnerFlag with Position)
selected_columns = [col for col in df.columns if col != 'Position' and col != 'IsWinnerFlag'] + ['Position']

# Filter DataFrame to only include selected columns
df = df[selected_columns]

# Handle missing values
df = df.ffill().bfill()
df = df.infer_objects()

# Prepare features and target
X = df.drop(columns=['Position'])
y = df['Position']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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
print("\nPreparing multi-class classification model...")

# Define base model with GPU support if available
try:
    test_model = XGBClassifier(device='cuda', objective='multi:softprob')
    test_model.fit(np.random.rand(10, 2), np.random.randint(20, size=10))
    gpu_available = True and HAS_CUDA
except Exception:
    gpu_available = False

base_model = XGBClassifier(
    device=device,
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    use_label_encoder=False,
    random_state=42
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
    n_iter=20,
    scoring='accuracy',  # Changed from 'roc_auc' to 'accuracy' for multi-class
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
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    use_label_encoder=False,
    eval_metric=['mlogloss', 'merror'],
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
    y_prob = best_model.predict_proba(X)
    all_metrics[name] = plot_and_save_metrics(y, y_pred, y_prob, name, 'MODEL_EVALUATION', label_encoder)

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
plt.savefig('MODEL_EVALUATION/feature_importance_position_prediction.png', dpi=300, bbox_inches='tight')
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

# Create directory for SHAP plots
os.makedirs('MODEL_EVALUATION/shap_plots', exist_ok=True)

# Plot and save SHAP summary for each position (top 5)
for position in range(5):  # Plot only top 5 positions to keep it manageable
    plt.figure(figsize=(15, 10))
    shap.summary_plot(
        shap_values[position], 
        X_train_enc_cpu,
        feature_names=feature_names,
        show=False,
        plot_size=(15, 10)
    )
    plt.title(f'SHAP Feature Importance for Position {label_encoder.classes_[position]}')
    plt.tight_layout()
    # Save before showing
    plt.savefig(f'MODEL_EVALUATION/shap_plots/shap_importance_position_{label_encoder.classes_[position]}_prediction.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Plot and save aggregate SHAP summary
plt.figure(figsize=(15, 10))
shap_values_mean = np.abs(np.array(shap_values)).mean(0)  # Average across all positions
shap.summary_plot(
    shap_values_mean,
    X_train_enc_cpu,
    feature_names=feature_names,
    show=False,
    plot_type="bar",
    plot_size=(15, 10)
)
plt.title('Aggregate SHAP Feature Importance Across All Positions')
plt.tight_layout()
# Save before showing
plt.savefig('MODEL_EVALUATION/shap_importance_position_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save model artifacts
os.makedirs('BEST_MODEL', exist_ok=True)

with open('BEST_MODEL/preprocessor_position_prediction.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('BEST_MODEL/best_model_position_prediction.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('BEST_MODEL/label_encoder_position_prediction.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save detailed metrics report
metrics_text = ["Model Performance Metrics", "=" * 50]
for dataset_name in ["Training", "Validation", "Test"]:
    metrics = all_metrics[dataset_name]
    metrics_text.extend([
        f"\n{dataset_name} Set Metrics:",
        "-" * 20,
        f"\nClassification Report:\n{metrics['classification_report']}",
        "\nMean ROC AUC Score (Top 5 positions):",
        f"{np.mean([metrics['roc_auc'][i] for i in range(5)]):.4f}",
        "\nPer-Position ROC AUC Scores (Top 5):"
    ])
    for i in range(5):
        metrics_text.append(f"Position {label_encoder.classes_[i]}: {metrics['roc_auc'][i]:.4f}")

with open('MODEL_EVALUATION/model_metrics.txt', 'w') as f:
    f.write('\n'.join(metrics_text))

print("\nAll results saved in MODEL_EVALUATION directory:")
print("- Feature importance plot and data")
print("- SHAP importance plot")
print("- ROC curves and confusion matrices for all sets")
print("- Detailed metrics report")
print("- Model, preprocessor, and label encoder pickle files") 