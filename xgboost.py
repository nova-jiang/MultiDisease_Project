import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
import os

# Simulated dataset (will replace once sorted)
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=200, n_features=100, n_informative=20,
    n_classes=4, n_clusters_per_class=1, random_state=42
)

# Nested CV (provisional)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("selector", SelectFromModel(XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), threshold="median")),
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"))
])

param_grid = {
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.1, 0.3],
    "xgb__n_estimators": [50, 100]
}

# Metric containers
all_metrics = []
conf_matrices = []
shap_values_all = []

# Outer CV loop
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    print(f"\n🔁 Fold {fold}")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    metrics = {
        "fold": fold,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr"),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred)
    }
    all_metrics.append(metrics)
    conf_matrices.append(confusion_matrix(y_test, y_pred))

    # SHAP Analysis
    explainer = shap.Explainer(best_model.named_steps["xgb"])
    shap_vals = explainer(best_model.named_steps["scaler"].transform(
        best_model.named_steps["imputer"].transform(X_test)))
    shap_values_all.append(shap_vals)

# Summary
metrics_df = pd.DataFrame(all_metrics)
print("\n📊 Average Metrics Across Folds:")
print(metrics_df.mean())

# Confusion Matrix
for i, cm in enumerate(conf_matrices, 1):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {i}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# SHAP Summary (last fold shown)
print("\nSHAP Summary Plot (last fold)")
shap.summary_plot(shap_values_all[-1], show=True)
