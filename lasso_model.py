"""
Step 3: Lasso Regression Model Training with Nested Cross-Validation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report)

class NestedLassoClassifier:
    def __init__(self, outer_cv_folds=5, inner_cv_folds=3, random_state=42, results_dir='results/step3_models/lasso_nested'):
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state
        self.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)
        self.nested_results = {}
        self.final_model = None

    def get_param_grid(self):
        return {
            'C': np.logspace(-4, 4, 10),
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'max_iter': [5000]
        }

    def inner_cv_hyperparameter_optimization(self, X_train, y_train):
        print(f"  Inner CV: Hyperparameter optimization on {X_train.shape[0]} samples...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        param_grid = self.get_param_grid()
        inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.random_state)

        grid = GridSearchCV(LogisticRegression(), param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
        grid.fit(X_scaled, y_train)

        return grid.best_estimator_, grid.best_score_, grid.best_params_, scaler

    def nested_cross_validation(self, X, y):
        print("="*80)
        print("NESTED CROSS-VALIDATION FOR LASSO REGRESSION")
        print("="*80)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

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
            print(f"    Best inner CV model: Lasso (F1: {best_score:.4f})")

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)

            try:
                auc = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
            except:
                auc = None

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')

            print("  Fold {} Results:".format(fold_idx + 1))
            print(f"    Test Accuracy: {acc:.4f}")
            print(f"    Test F1-macro: {f1_macro:.4f}")
            print(f"    Test AUC-macro: {auc:.4f}" if auc is not None else "    Test AUC-macro: N/A")
            print(f"    Best Params: {best_params}")

            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': acc,
                'f1_macro': f1_macro,
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

        print("="*80)
        print("NESTED CV RESULTS SUMMARY")
        print("="*80)
        for k, v in self.nested_results['overall_metrics'].items():
            if v is not None:
                values = [f[k] for f in fold_results if f[k] is not None]
                print(f"{k.replace('_', ' ').title()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print("="*80)

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
        data = {metric: [f[metric] for f in self.nested_results['fold_results']] for metric in metrics}

        df = pd.DataFrame(data, index=[f"Fold {i}" for i in folds])
        df.plot(kind='bar', figsize=(12, 6))
        plt.title("Cross-Fold Performance Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/cross_fold_performance.png")
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
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
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

        labels = list(reports[0].keys())
        for skip in ['accuracy', 'macro avg', 'weighted avg']:
            if skip in labels:
                labels.remove(skip)

        averaged = {}
        for label in labels:
            avg_metrics = {'precision': [], 'recall': [], 'f1-score': []}
            for r in reports:
                if label in r:
                    for k in avg_metrics:
                        avg_metrics[k].append(r[label].get(k, 0))
            averaged[label] = {k: np.mean(v) for k, v in avg_metrics.items()}

        df = pd.DataFrame(averaged).T
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Average Classification Report (per class)')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/classification_report_heatmap.png")
        plt.close()

def run_lasso_nested_cv(X, y, results_dir='results/step3_models/lasso_nested'):
    classifier = NestedLassoClassifier(results_dir=results_dir)
    nested_results = classifier.nested_cross_validation(X, y)
    classifier.create_visualizations()
    return nested_results, classifier
