"""
Step 3: XGBoost Model Training with Nested Cross-Validation
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
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, classification_report, confusion_matrix)

class NestedXGBoostClassifier:
    def __init__(self, outer_cv_folds=5, inner_cv_folds=3, random_state=42, results_dir='results/step3_models/xgboost_nested'):
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state
        self.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)
        self.nested_results = {}
        self.final_model = None


    def get_param_grid(self):
        return {
            'learning_rate': [0.01, 0.1, 0.3], # 0.01, 0.1, 0.3
            'max_depth': [3, 5, 7], # 3, 5, 7
            'n_estimators': [50, 100, 200], # 50, 100, 200
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

    def inner_cv_hyperparameter_optimization(self, X_train, y_train):
        print(f"    Inner CV: Hyperparameter optimization on {X_train.shape[0]} samples...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        param_grid = self.get_param_grid()
        inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.random_state)

        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=self.random_state)
        grid = GridSearchCV(model, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1, error_score='raise')
        grid.fit(X_scaled, y_train)

        return grid.best_estimator_, grid.best_score_, grid.best_params_, scaler

    def nested_cross_validation(self, X, y):
        print("="*80)
        print("XGBOOST NESTED CROSS-VALIDATION PIPELINE")
        print("="*80)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.class_names = le.classes_.tolist()

        outer_cv = StratifiedKFold(n_splits=self.outer_cv_folds, shuffle=True, random_state=self.random_state)

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded)):
            print(f"\nOuter CV Fold {fold_idx + 1}/{self.outer_cv_folds}")
            print("-" * 50)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            best_model, best_score, best_params, scaler = self.inner_cv_hyperparameter_optimization(X_train, y_train)
            print(f"    Best inner CV model: XGBoost (F1: {best_score:.4f})")

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)

            try:
                auc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
            except:
                auc = None

            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'auc_macro': auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'best_params': best_params,
                'inner_cv_score': best_score
            })

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        self.nested_results = {
            'fold_results': fold_results,
            'overall_metrics': {
                'accuracy': np.mean([f['accuracy'] for f in fold_results]),
                'f1_macro': np.mean([f['f1_macro'] for f in fold_results]),
                'f1_weighted': np.mean([f['f1_weighted'] for f in fold_results]),
                'precision_macro': np.mean([f['precision_macro'] for f in fold_results]),
                'recall_macro': np.mean([f['recall_macro'] for f in fold_results]),
                'auc_macro': np.mean([f['auc_macro'] for f in fold_results if f['auc_macro'] is not None])
                if any(f['auc_macro'] is not None for f in fold_results) else None
            },
            'overall_confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
            'overall_classification_report': classification_report(all_y_true, all_y_pred, output_dict=True)
        }

        # Save final model info and nested results
        summary_info = {
            "best_params": fold_results[-1]['best_params'],  # from last fold
            "cv_score": self.nested_results['overall_metrics']['f1_macro'],
            "n_features": X.shape[1]
        }
        with open(os.path.join(self.results_dir, "final_model_info.json"), 'w') as f:
            json.dump(summary_info, f, indent=2)

        with open(os.path.join(self.results_dir, "xgboost_nested_cv_results.json"), 'w') as f:
            json.dump(self.nested_results, f, indent=2)

        return self.nested_results

    def create_visualizations(self):
        self._plot_auc_per_fold()
        self._plot_precision_recall_per_fold()
        self._plot_classification_report_heatmap()
        self._plot_cross_fold_performance()
        self._plot_inner_vs_outer()
        self._plot_confusion_matrix()

    def _plot_cross_fold_performance(self):
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'auc_macro']
        folds = [f['fold'] for f in self.nested_results['fold_results']]

        # Assign consistent, distinctive colors per metric
        metric_colors = {
            'accuracy': "#50a2dd",
            'f1_macro': "#eba364",
            'precision_macro': "#72b872",
            'recall_macro': "#e87070",
            'auc_macro': "#9a75bd"
        }

        for metric in metrics:
            scores = [f[metric] for f in self.nested_results['fold_results']]
            mean_score = np.mean(scores)

            plt.figure(figsize=(7, 5))
            bars = plt.bar(
                [f"Fold {i}" for i in folds],
                scores,
                color=metric_colors.get(metric, 'skyblue')
            )

            # Mean line
            plt.axhline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_score:.3f}")
            plt.text(len(folds) - 0.6, mean_score + 0.01, f"Mean: {mean_score:.3f}", color='red', fontsize=10)

            # Formatting
            plt.title(f"{metric.replace('_', ' ').title()} Across CV Folds")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xlabel("Fold")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/{metric}_per_fold.png")
            plt.close()





    def _plot_inner_vs_outer(self):
        folds = [f['fold'] for f in self.nested_results['fold_results']]
        inner_scores = [f['inner_cv_score'] for f in self.nested_results['fold_results']]
        outer_scores = [f['f1_macro'] for f in self.nested_results['fold_results']]

        x = np.arange(len(folds))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, inner_scores, width, label='Inner CV F1')
        plt.bar(x + width/2, outer_scores, width, label='Outer Test F1')
        plt.xticks(x, [f"Fold {i}" for i in folds])
        plt.ylabel('F1 Score')
        plt.title('Inner vs Outer F1 Scores per Fold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/inner_vs_outer_f1.png")
        plt.close()


    def _plot_confusion_matrix(self):
        if 'overall_confusion_matrix' not in self.nested_results:
            return

        cm = np.array(self.nested_results['overall_confusion_matrix'])

        class_names = self.class_names if hasattr(self, 'class_names') else [str(i) for i in range(cm.shape[0])]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title("Overall Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/overall_confusion_matrix.png")
        plt.close()



    def _plot_auc_per_fold(self):
        aucs = [f['auc_macro'] for f in self.nested_results['fold_results'] if f['auc_macro'] is not None]
        folds = [f['fold'] for f in self.nested_results['fold_results'] if f['auc_macro'] is not None]

        if aucs:
            plt.figure(figsize=(8, 5))
            plt.plot(folds, aucs, marker='o')
            plt.axhline(np.mean(aucs), color='red', linestyle='--', label=f"Mean AUC: {np.mean(aucs):.3f}")
            plt.title('ROC AUC per Fold')
            plt.xlabel('Fold')
            plt.ylabel('AUC Score')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/auc_per_fold.png")
            plt.close()

    def _plot_precision_recall_per_fold(self):
        folds = [f['fold'] for f in self.nested_results['fold_results']]
        precision = [f['precision_macro'] for f in self.nested_results['fold_results']]
        recall = [f['recall_macro'] for f in self.nested_results['fold_results']]

        x = np.arange(len(folds))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, precision, width, label='Precision')
        ax.bar(x + width/2, recall, width, label='Recall')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i}" for i in folds])
        ax.set_title('Precision vs Recall per Fold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/precision_recall_per_fold.png")
        plt.close()

    def _plot_classification_report_heatmap(self):
        reports = [f['classification_report'] for f in self.nested_results['fold_results']]
        if not reports:
            return

        # Ensure we have class names
        class_names = getattr(self, 'class_names', [str(i) for i in range(len(reports[0]))])
        idx_to_name = {str(i): name for i, name in enumerate(class_names)}

        averaged = {}
        for idx_str, label_name in idx_to_name.items():
            avg_metrics = {'precision': [], 'recall': [], 'f1-score': []}
            for r in reports:
                if idx_str in r:
                    for k in avg_metrics:
                        avg_metrics[k].append(r[idx_str].get(k, 0))
            averaged[label_name] = {k: np.mean(v) for k, v in avg_metrics.items()}

        df = pd.DataFrame(averaged).T
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Average Classification Report (per class)')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/classification_report_heatmap.png")
        plt.close()


    def print_summary(self):
        print("Saving results...")
        print(f"Results saved to {self.results_dir}/")
        print("="*80)
        print("NESTED CV RESULTS SUMMARY")
        print("="*80)
        for metric, value in self.nested_results['overall_metrics'].items():
            if value is not None:
                std_dev = np.std([f[metric] for f in self.nested_results['fold_results'] if f[metric] is not None])
                print(f"{metric.replace('_', ' ').title()}: {value:.4f} ± {std_dev:.4f}")
        print("="*80)


def run_xgboost_nested_cv(X, y, results_dir='results/step3_models/xgboost_nested'):
    classifier = NestedXGBoostClassifier(results_dir=results_dir)
    nested_results = classifier.nested_cross_validation(X, y)
    classifier.create_visualizations()
    classifier.print_summary()
    return nested_results, classifier
