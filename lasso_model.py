import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

class LassoModelTraining:
    def __init__(self, cv_folds=5, test_size=0.2, random_state=42):
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        self.best_params = None
        self.best_model = None

    def load_and_prepare_data(self, X_filtered, y, selected_features):
        X = X_filtered[selected_features].copy()
        X.fillna(X.median(numeric_only=True), inplace=True)

        self.feature_names = selected_features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        return X_scaled, y_encoded

    def hyperparameter_tuning(self, X, y):
        print("Performing Lasso hyperparameter tuning...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0]
        }

        model = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=5000)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_

        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")

        grid_results = pd.DataFrame(grid_search.cv_results_)
        grid_results.to_csv('lasso_grid_search_results.csv', index=False)
        print("  Grid search results saved to 'lasso_grid_search_results.csv'")

        return self.best_model

    def cross_validation_evaluation(self, X, y, model):
        print("Performing cross-validation evaluation...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

        cv_results = {}
        for metric in scoring:
            test_scores = results[f'test_{metric}']
            train_scores = results[f'train_{metric}']
            cv_results[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_scores': test_scores.tolist(),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_scores': train_scores.tolist()
            }

        print("\nCross-validation Results:")
        print("-" * 40)
        for metric, results in cv_results.items():
            print(f"{metric.capitalize()}: {results['test_mean']:.4f} ± {results['test_std']:.4f}")

        return cv_results

    def train_final_model(self, X, y, model):
        print("Training final model and evaluating on test set...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred)

        test_results = {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'test_predictions': y_pred.tolist(),
            'test_true_labels': y_test.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }

        print(f"\nTest Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-score (macro): {f1:.4f}")

        return test_results

    def save_results(self, cv_results, test_results):
        print("Saving results to files...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_filename = f'lasso_model_{timestamp}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'best_params': self.best_params
            }, f)
        print(f"  Model saved to: {model_filename}")

        cv_filename = f'lasso_cv_results_{timestamp}.json'
        with open(cv_filename, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"  CV results saved to: {cv_filename}")

        test_filename = f'lasso_test_results_{timestamp}.json'
        with open(test_filename, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"  Test results saved to: {test_filename}")

        summary = {
            'model_type': 'Lasso (Logistic Regression L1)',
            'timestamp': timestamp,
            'best_parameters': self.best_params,
            'cv_performance': {
                'accuracy': f"{cv_results['accuracy']['test_mean']:.4f} ± {cv_results['accuracy']['test_std']:.4f}",
                'f1_macro': f"{cv_results['f1_macro']['test_mean']:.4f} ± {cv_results['f1_macro']['test_std']:.4f}"
            },
            'test_performance': {
                'accuracy': test_results['accuracy'],
                'f1_macro': test_results['f1_macro']
            },
            'n_features': len(self.feature_names),
            'n_samples': len(test_results['test_true_labels']) / self.test_size
        }

        summary_filename = f'lasso_summary_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary report saved to: {summary_filename}")

        cm = np.array(test_results['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Lasso Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"lasso_confusion_matrix_{timestamp}.png")
        plt.close()

    def run_complete_pipeline(self, X_filtered, y, selected_features):
        print("="*60)
        print("LASSO MODEL TRAINING PIPELINE")
        print("="*60)

        X_processed, y_encoded = self.load_and_prepare_data(X_filtered, y, selected_features)
        best_model = self.hyperparameter_tuning(X_processed, y_encoded)
        cv_results = self.cross_validation_evaluation(X_processed, y_encoded, best_model)
        test_results = self.train_final_model(X_processed, y_encoded, best_model)
        self.save_results(cv_results, test_results)

        print("="*60)
        print("LASSO TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {cv_results['accuracy']['test_mean']:.4f} ± {cv_results['accuracy']['test_std']:.4f}")
        print("="*60)

        return {
            'model': self.best_model,
            'cv_results': cv_results,
            'test_results': test_results,
            'best_params': self.best_params
        }
