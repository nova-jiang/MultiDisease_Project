"""
Step 3: Random Forest Model Training with Nested Cross-Validation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend that doesn’t require Tcl/Tk
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


class NestedRandomForestClassifier:
    """Random Forest classifier with nested cross-validation."""

    def __init__(self, outer_cv_folds=5, inner_cv_folds=5, random_state=42,
                 results_dir="results/step3_models/random_forest_nested"):
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state
        self.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)
        self.nested_results = {}
        self.final_model = None

    def get_param_grid(self):
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }

    def inner_cv_hyperparameter_optimization(self, X_train, y_train):
        print(f"    Inner CV: Hyperparameter optimization on {X_train.shape[0]} samples...")
        param_grid = self.get_param_grid()
        rf = RandomForestClassifier(random_state=self.random_state)
        inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True,
                                   random_state=self.random_state)
        grid_search = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(
            f"      Best params: {best_params}, F1-macro: {best_score:.4f}"
        )
        return best_model, best_params, best_score

    def nested_cross_validation(self, X, y):
        print("=" * 80)
        print("NESTED CROSS-VALIDATION FOR RANDOM FOREST")
        print("=" * 80)
        outer_cv = StratifiedKFold(
            n_splits=self.outer_cv_folds, shuffle=True, random_state=self.random_state
        )

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\nOuter CV Fold {fold_idx + 1}/{self.outer_cv_folds}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            best_model, best_params, inner_score = self.inner_cv_hyperparameter_optimization(
                X_train, y_train
            )

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            metrics = {
                "fold": fold_idx + 1,
                "best_params": best_params,
                "inner_cv_score": inner_score,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "precision_macro": precision_score(y_test, y_pred, average="macro"),
                "recall_macro": recall_score(y_test, y_pred, average="macro"),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            if hasattr(best_model, "predict_proba"):
                try:
                    y_prob = best_model.predict_proba(X_test)
                    auc = roc_auc_score(
                        pd.get_dummies(y_test), y_prob, multi_class="ovr", average="macro"
                    )
                    metrics["auc_macro"] = auc
                except Exception:
                    metrics["auc_macro"] = None
            else:
                metrics["auc_macro"] = None

            fold_results.append(metrics)
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        overall = {
            "accuracy": np.mean([f["accuracy"] for f in fold_results]),
            "accuracy_std": np.std([f["accuracy"] for f in fold_results]),
            "f1_macro": np.mean([f["f1_macro"] for f in fold_results]),
            "f1_macro_std": np.std([f["f1_macro"] for f in fold_results]),
            "f1_weighted": np.mean([f["f1_weighted"] for f in fold_results]),
            "f1_weighted_std": np.std([f["f1_weighted"] for f in fold_results]),
            "precision_macro": np.mean([f["precision_macro"] for f in fold_results]),
            "precision_macro_std": np.std([f["precision_macro"] for f in fold_results]),
            "recall_macro": np.mean([f["recall_macro"] for f in fold_results]),
            "recall_macro_std": np.std([f["recall_macro"] for f in fold_results]),
        }
        auc_scores = [f["auc_macro"] for f in fold_results if f["auc_macro"] is not None]
        if auc_scores:
            overall["auc_macro"] = np.mean(auc_scores)
            overall["auc_macro_std"] = np.std(auc_scores)

        overall_cm = confusion_matrix(all_y_true, all_y_pred)
        overall_cr = classification_report(all_y_true, all_y_pred, output_dict=True)

        self.nested_results = {
            "fold_results": fold_results,
            "overall_metrics": overall,
            "overall_confusion_matrix": overall_cm.tolist(),
            "overall_classification_report": overall_cr,
            "all_predictions": {"y_true": all_y_true, "y_pred": all_y_pred},
        }
        return self.nested_results

    def train_final_model(self, X, y):
        print("\nTraining final model on complete dataset...")
        best_model, best_params, best_score = self.inner_cv_hyperparameter_optimization(X, y)
        best_model.fit(X, y)
        self.final_model = {
            "model": best_model,
            "best_params": best_params,
            "cv_score": best_score,
            "n_features": X.shape[1],
        }
        print(f"Final model params: {best_params}")
        print(f"CV F1-score: {best_score:.4f}")
        return self.final_model

    def _plot_cross_fold_performance(self):
        folds = [f"Fold {f['fold']}" for f in self.nested_results['fold_results']]
        acc = [f["accuracy"] for f in self.nested_results["fold_results"]]
        f1 = [f["f1_macro"] for f in self.nested_results["fold_results"]]
        plt.figure(figsize=(12, 6))
        x = np.arange(len(folds))
        width = 0.35
        plt.bar(x - width/2, acc, width, label="Accuracy")
        plt.bar(x + width/2, f1, width, label="F1-macro")
        plt.xticks(x, folds, rotation=45)
        plt.ylabel("Score")
        plt.title("Cross-Fold Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/cross_fold_performance.png")
        plt.close()

    def _plot_confusion_matrix(self, y):
        cm = np.array(self.nested_results['overall_confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()),
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Overall Confusion Matrix (Nested CV)")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/overall_confusion_matrix.png")
        plt.close()

    def _plot_feature_importance(self):
        if not self.final_model:
            return
        model = self.final_model["model"]
        if not hasattr(model, "feature_importances_"):
            return
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), indices, rotation=90)
        plt.ylabel("Importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/feature_importance.png")
        plt.close()

    def create_visualizations(self, y):
        self._plot_cross_fold_performance()
        self._plot_confusion_matrix(y)
        self._plot_feature_importance()

    def save_results(self):
        print("Saving results...")
        with open(f"{self.results_dir}/random_forest_nested_cv_results.json", "w") as f:
            json.dump(self.nested_results, f, indent=2)
        if self.final_model:
            info = {
                "best_params": self.final_model["best_params"],
                "cv_score": self.final_model["cv_score"],
                "n_features": self.final_model["n_features"],
            }
            with open(f"{self.results_dir}/final_model_info.json", "w") as f:
                json.dump(info, f, indent=2)
        print(f"Results saved to {self.results_dir}/")

    def run_complete_pipeline(self, X, y):
        self.nested_cross_validation(X, y)
        self.train_final_model(X, y)
        self.create_visualizations(y)
        self.save_results()
        overall = self.nested_results['overall_metrics']
        print("=" * 80)
        print("NESTED CV RESULTS SUMMARY")
        print("=" * 80)
        print(f"Accuracy: {overall['accuracy']:.4f} ± {overall['accuracy_std']:.4f}")
        print(f"F1-Macro: {overall['f1_macro']:.4f} ± {overall['f1_macro_std']:.4f}")
        print(
            f"Precision-Macro: {overall['precision_macro']:.4f} ± {overall['precision_macro_std']:.4f}"
        )
        print(
            f"Recall-Macro: {overall['recall_macro']:.4f} ± {overall['recall_macro_std']:.4f}"
        )
        if 'auc_macro' in overall:
            print(f"AUC-Macro: {overall['auc_macro']:.4f} ± {overall['auc_macro_std']:.4f}")
        print("=" * 80)
        return self.nested_results


def run_random_forest_nested_cv(X, y, results_dir="results/step3_models/random_forest_nested"):
    """Run Random Forest classification with nested cross-validation."""
    classifier = NestedRandomForestClassifier(results_dir=results_dir)
    nested_results = classifier.run_complete_pipeline(X, y)
    return nested_results, classifier