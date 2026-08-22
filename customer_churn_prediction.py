"""
Task 3: Customer Churn Prediction
==================================
Predicts customer churn for a subscription-based business using historical
customer data (usage behavior + demographics). Trains and compares three
models: Logistic Regression, Random Forest, and Gradient Boosting.

Dataset: shantanudhakadd/bank-customer-churn-prediction (via kagglehub)

Install dependencies first if needed:
    pip install kagglehub pandas numpy scikit-learn matplotlib seaborn joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/script runs; remove if running in a notebook
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

OUTPUT_DIR = "churn_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Download & load the dataset
# ---------------------------------------------------------------------------
path = kagglehub.dataset_download("shantanudhakadd/bank-customer-churn-prediction")
print("Path to dataset files:", path)

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {path}")
df = pd.read_csv(os.path.join(path, csv_files[0]))

print(f"\nLoaded {csv_files[0]}  ->  shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Identify the target column
# ---------------------------------------------------------------------------
# This dataset's churn label is called "Exited" (1 = customer left, 0 = stayed).
# Fallback search in case the column name differs slightly.
candidates = ["Exited", "Churn", "churn", "is_churn", "Churned"]
target_col = next((c for c in candidates if c in df.columns), None)
if target_col is None:
    raise ValueError(
        f"Couldn't auto-detect the target column. Available columns: {list(df.columns)}"
    )
print(f"\nTarget column: '{target_col}'")

# ---------------------------------------------------------------------------
# 3. Quick EDA
# ---------------------------------------------------------------------------
missing = df.isnull().sum()
missing = missing[missing > 0]
print("\nMissing values per column:")
print(missing if not missing.empty else "None")

churn_rate = df[target_col].value_counts(normalize=True)
print(f"\nChurn distribution:\n{churn_rate}")

plt.figure(figsize=(5, 4))
sns.countplot(x=target_col, data=df)
plt.title("Churn Distribution (0 = Stayed, 1 = Churned)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/churn_distribution.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------------------
# Drop identifier / free-text columns that carry no predictive signal
id_like_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
df = df.drop(columns=id_like_cols)

# One-hot encode categorical features (e.g. Geography, Gender).
# Numeric-dtype check works reliably across pandas versions, unlike relying
# on the "object" dtype label alone.
cat_cols = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features (needed for Logistic Regression; harmless for trees)
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=np.number).columns
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------------------
# 5. Train & evaluate models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

results = []
roc_data = {}
fitted_models = {}
cm_data = {}

for name, model in models.items():
    # Logistic Regression benefits from scaled features; tree models don't need it.
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    fitted_models[name] = model
    cm_data[name] = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc
    })
    roc_data[name] = roc_curve(y_test, y_proba)[:2] + (auc,)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("\n===== Model comparison (sorted by ROC-AUC) =====")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 5b. Model comparison bar chart (all metrics, all models)
# ---------------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
plot_df = results_df.set_index("Model")[metrics]

ax = plot_df.plot(kind="bar", figsize=(9, 5.5), width=0.75, colormap="viridis")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Model Comparison Across Metrics")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(metrics), frameon=False)
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. ROC curve comparison
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 6b. Confusion matrices (one heatmap per model, side by side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(cm_data), figsize=(5 * len(cm_data), 4.5))
if len(cm_data) == 1:
    axes = [axes]
for ax, (name, cm) in zip(axes, cm_data.items()):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"]
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
rf_model = fitted_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(7, 6))
importances.head(10).iloc[::-1].plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 8. Save the best model
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
joblib.dump(best_model, f"{OUTPUT_DIR}/best_model_{best_model_name.replace(' ', '_').lower()}.joblib")
if best_model_name == "Logistic Regression":
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")

print(f"\nBest model: {best_model_name} (ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})")
print(f"All outputs (plots, comparison table, saved model) written to '{OUTPUT_DIR}/'")
