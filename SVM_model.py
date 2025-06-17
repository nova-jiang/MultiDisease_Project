import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate, 
                                   GridSearchCV, train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SVMModelTraining:
    """
    Support Vector Machine model training pipeline for microbiome multi-class classification
    """
    
    def __init__(self, cv_folds=5, test_size=0.2, random_state=42):
        """
        Initialize SVM training pipeline
        
        Parameters:
        -----------
        cv_folds : int, default=5
            Number of cross-validation folds
        test_size : float, default=0.2
            Proportion of data to use for final testing
        random_state : int, default=42
            Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model storage
        self.best_model = None
        self.best_params = None
        
        # Results storage
        self.cv_results = {}
        self.test_results = {}
        self.feature_importance = None
        
    def load_and_prepare_data(self, X_filtered, y, selected_features):
        """
        Load and prepare data for SVM training
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered microbiome data from Step 1
        y : pd.Series
            Category labels
        selected_features : list
            Selected feature names from Step 2
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed feature matrix
        y_encoded : np.ndarray
            Encoded labels
        """
        print("Loading and preparing data for SVM training...")
        
        # Select features
        available_features = [f for f in selected_features if f in X_filtered.columns]
        X_selected = X_filtered[available_features].copy()
        
        print(f"Using {len(available_features)} features")
        print(f"Dataset shape: {X_selected.shape}")
        print(f"Category distribution:\n{y.value_counts()}")
        
        # Handle NaN values
        nan_count = X_selected.isnull().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values, filling with median...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_selected = pd.DataFrame(
                imputer.fit_transform(X_selected),
                columns=X_selected.columns,
                index=X_selected.index
            )
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        print("Feature scaling completed")
        
        # Store feature names
        self.feature_names = available_features
        
        return X_scaled, y_encoded
    
    def hyperparameter_tuning(self, X, y):
        """
        Perform hyperparameter tuning for SVM
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        best_svm : SVC
            Best SVM model after hyperparameter tuning
        """
        print("Performing SVM hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4]  # Only used for poly kernel
        }
        
        # Create stratified CV
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        print("  Running grid search (this may take a while)...")
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.best_model = grid_search.best_estimator_
        
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        # Save grid search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        grid_results.to_csv('svm_grid_search_results.csv', index=False)
        print("  Grid search results saved to 'svm_grid_search_results.csv'")
        
        return self.best_model
    
    def cross_validation_evaluation(self, X, y, model):
        """
        Perform detailed cross-validation evaluation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        model : SVC
            Tuned SVM model
            
        Returns:
        --------
        cv_results : dict
            Detailed cross-validation results
        """
        print("Performing cross-validation evaluation...")
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        # Perform cross-validation
        cv_scores = cross_validate(
            model, X, y, 
            cv=cv, 
            scoring=scoring_metrics, 
            return_train_score=True,
            return_estimator=True  # Return fitted estimators for feature importance analysis
        )
        
        # Calculate statistics
        cv_results = {}
        for metric in scoring_metrics:
            test_scores = cv_scores[f'test_{metric}']
            train_scores = cv_scores[f'train_{metric}']
            
            cv_results[metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_scores': test_scores.tolist(),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_scores': train_scores.tolist()
            }
        
        # Print results
        print("\nCross-validation Results:")
        print("-" * 40)
        for metric, results in cv_results.items():
            print(f"{metric.capitalize()}: {results['test_mean']:.4f} ± {results['test_std']:.4f}")
        
        # Store CV estimators for feature importance
        self.cv_estimators = cv_scores['estimator']
        
        return cv_results
    
    def train_final_model(self, X, y, model):
        """
        Train final model and evaluate on test set
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        model : SVC
            Best SVM model
            
        Returns:
        --------
        test_results : dict
            Test set evaluation results
        """
        print("Training final model and evaluating on test set...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Detailed classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
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
            'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-score (macro): {f1:.4f}")
        
        return test_results
    
    def extract_feature_importance(self):
        """
        Extract feature importance from SVM models
        
        Returns:
        --------
        feature_importance : pd.DataFrame or None
            Feature importance if available (linear kernel only)
        """
        print("Extracting feature importance...")
        
        # Check if we can extract feature importance (linear kernel only)
        if hasattr(self.best_model, 'coef_') and self.best_model.coef_ is not None:
            # For multi-class, average absolute coefficients across all binary classifiers
            if len(self.best_model.coef_.shape) > 1:
                importance_scores = np.abs(self.best_model.coef_).mean(axis=0)
            else:
                importance_scores = np.abs(self.best_model.coef_[0])
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:<35} {row['importance']:.6f}")
            
            return feature_importance
        else:
            print("Feature importance not available for non-linear kernels")
            return None
    
    def visualize_results(self, cv_results, test_results, feature_importance):
        """
        Create visualizations for SVM results
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        test_results : dict
            Test results
        feature_importance : pd.DataFrame or None
            Feature importance data
        """
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SVM Model Performance Analysis', fontsize=16)
        
        # 1. Cross-validation scores
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        means = [cv_results[m]['test_mean'] for m in metrics]
        stds = [cv_results[m]['test_std'] for m in metrics]
        
        axes[0, 0].bar(range(len(metrics)), means, yerr=stds, capsize=5, alpha=0.7)
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Cross-Validation Performance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confusion matrix
        cm = np.array(test_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0, 1])
        axes[0, 1].set_title('Test Set Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Feature importance (if available)
        if feature_importance is not None:
            top_features = feature_importance.head(15)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels([f[:30] + '...' if len(f) > 30 else f 
                                      for f in top_features['feature']], fontsize=8)
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].set_title('Top 15 Feature Importance')
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available\n(non-linear kernel)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Feature Importance')
        
        # 4. Per-class performance
        class_report = test_results['classification_report']
        classes = [cls for cls in class_report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [class_report[cls]['f1-score'] for cls in classes]
        
        axes[1, 1].bar(range(len(classes)), f1_scores, alpha=0.7)
        axes[1, 1].set_xticks(range(len(classes)))
        axes[1, 1].set_xticklabels(classes, rotation=45)
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('Per-Class F1-Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('svm_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, cv_results, test_results, feature_importance):
        """
        Save all results to files
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        test_results : dict
            Test results
        feature_importance : pd.DataFrame or None
            Feature importance data
        """
        print("Saving results to files...")
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save trained model
        model_filename = f'svm_model_{timestamp}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'best_params': self.best_params
            }, f)
        print(f"  Model saved to: {model_filename}")
        
        # 2. Save cross-validation results
        cv_filename = f'svm_cv_results_{timestamp}.json'
        with open(cv_filename, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"  CV results saved to: {cv_filename}")
        
        # 3. Save test results
        test_filename = f'svm_test_results_{timestamp}.json'
        with open(test_filename, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"  Test results saved to: {test_filename}")
        
        # 4. Save feature importance (if available)
        if feature_importance is not None:
            importance_filename = f'svm_feature_importance_{timestamp}.csv'
            feature_importance.to_csv(importance_filename, index=False)
            print(f"  Feature importance saved to: {importance_filename}")
        
        # 5. Save summary report
        summary = {
            'model_type': 'SVM',
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
            'n_samples': len(test_results['test_true_labels']) / self.test_size  # Approximate total samples
        }
        
        summary_filename = f'svm_summary_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary report saved to: {summary_filename}")
    
    def run_complete_pipeline(self, X_filtered, y, selected_features):
        """
        Run the complete SVM training pipeline
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered microbiome data from Step 1
        y : pd.Series
            Category labels
        selected_features : list
            Selected feature names from Step 2
            
        Returns:
        --------
        results : dict
            Complete training results
        """
        print("="*60)
        print("SVM MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X_processed, y_encoded = self.load_and_prepare_data(X_filtered, y, selected_features)
        
        # Hyperparameter tuning
        best_model = self.hyperparameter_tuning(X_processed, y_encoded)
        
        # Cross-validation evaluation
        cv_results = self.cross_validation_evaluation(X_processed, y_encoded, best_model)
        
        # Train final model and test
        test_results = self.train_final_model(X_processed, y_encoded, best_model)
        
        # Extract feature importance
        feature_importance = self.extract_feature_importance()
        
        # Visualize results
        self.visualize_results(cv_results, test_results, feature_importance)
        
        # Save all results
        self.save_results(cv_results, test_results, feature_importance)
        
        print("="*60)
        print("SVM TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {cv_results['accuracy']['test_mean']:.4f} ± {cv_results['accuracy']['test_std']:.4f}")
        print("="*60)
        
        return {
            'model': self.best_model,
            'cv_results': cv_results,
            'test_results': test_results,
            'feature_importance': feature_importance,
            'best_params': self.best_params
        }
