"""
Step 3: SVM Model Training with Nested Cross-Validation
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_score, recall_score,
                           roc_auc_score)

import matplotlib.pyplot as plt
import seaborn as sns

class NestedSVMClassifier:
    """
    SVM classifier with nested cross-validation for unbiased performance estimation
    
    Uses pre-selected features from Steps 1&2
    Outer CV: Performance estimation on completely independent test folds
    Inner CV: Hyperparameter optimization only
    """
    
    def __init__(self, 
                 outer_cv_folds=5,
                 inner_cv_folds=3,
                 random_state=42,
                 results_dir='results/step3_models/svm_nested'):
        """
        Initialize nested SVM classifier
        
        Parameters:
        -----------
        outer_cv_folds : int, default=5
            Number of outer cross-validation folds for performance estimation
        inner_cv_folds : int, default=3
            Number of inner cross-validation folds for hyperparameter optimization
        random_state : int, default=42
            Random state for reproducibility
        results_dir : str
            Directory to save results
        """
        self.outer_cv_folds = outer_cv_folds
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Results storage
        self.nested_results = {}
        self.final_model = None
        
    def get_svm_parameter_grids(self):
        """
        Define parameter grids for different SVM variants
        
        Returns:
        --------
        param_grids : dict
            Dictionary of parameter grids for each SVM type
        """
        return {
            'LinearSVC': {
                'estimator': LinearSVC(random_state=self.random_state, max_iter=2000),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'loss': ['hinge', 'squared_hinge'],
                    'dual': [False]  # Better for n_samples > n_features
                }
            },
            'RBF_SVM': {
                'estimator': SVC(kernel='rbf', random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
                    'class_weight': [None, 'balanced']
                }
            },
            'Poly_SVM': {
                'estimator': SVC(kernel='poly', random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'degree': [2, 3],
                    'gamma': ['scale', 'auto'],
                    'class_weight': [None, 'balanced']
                }
            }
        }
    
    def inner_cv_hyperparameter_optimization(self, X_train, y_train):
        """
        Inner CV for hyperparameter optimization
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features (pre-selected from Steps 1&2)
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        best_model : sklearn estimator
            Best model after hyperparameter optimization
        best_model_name : str
            Name of the best model
        best_params : dict
            Best hyperparameters
        best_score : float
            Best cross-validation score
        """
        print(f"    Inner CV: Hyperparameter optimization on {X_train.shape[0]} samples...")
        
        # Standardize features (crucial for SVM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        param_grids = self.get_svm_parameter_grids()
        inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, 
                                 random_state=self.random_state)
        
        best_score = -1
        best_model = None
        best_model_name = None
        best_params = None
        
        for model_name, config in param_grids.items():
            try:
                print(f"      Optimizing {model_name}...")
                
                grid_search = GridSearchCV(
                    estimator=config['estimator'],
                    param_grid=config['params'],
                    cv=inner_cv,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_scaled, y_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    best_params = grid_search.best_params_
                
                print(f"        Best {model_name} F1: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"        Warning: {model_name} optimization failed: {e}")
                continue
        
        if best_model is None:
            # Fallback to simple LinearSVC
            print("      Using fallback LinearSVC")
            best_model = LinearSVC(random_state=self.random_state, max_iter=2000)
            best_model_name = "LinearSVC_fallback"
            best_params = {}
            best_score = 0
        
        print(f"    Best inner CV model: {best_model_name} (F1: {best_score:.4f})")
        return best_model, best_model_name, best_params, best_score, scaler
    
    def nested_cross_validation(self, X, y):
        """
        Perform nested cross-validation for unbiased performance estimation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (pre-selected features from Steps 1&2)
        y : pd.Series
            Target vector
            
        Returns:
        --------
        nested_results : dict
            Results from nested cross-validation
        """
        print("="*80)
        print("NESTED CROSS-VALIDATION FOR SVM")
        print("="*80)
        print(f"Using {X.shape[1]} pre-selected features from Steps 1&2")
        print(f"Dataset: {X.shape[0]} samples")
        print(f"Outer CV: {self.outer_cv_folds} folds")
        print(f"Inner CV: {self.inner_cv_folds} folds")
        
        outer_cv = StratifiedKFold(n_splits=self.outer_cv_folds, shuffle=True, 
                                 random_state=self.random_state)
        
        fold_results = []
        all_y_true = []
        all_y_pred = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\nOuter CV Fold {fold_idx + 1}/{self.outer_cv_folds}")
            print("-" * 50)
            
            # Split data for this fold
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            print(f"  Train: {len(X_train_fold)} samples, Test: {len(X_test_fold)} samples")
            
            # Inner CV for hyperparameter optimization (on training data only)
            best_model, best_model_name, best_params, best_inner_score, scaler = self.inner_cv_hyperparameter_optimization(
                X_train_fold, y_train_fold
            )
            
            # Train final model on full training fold
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            best_model.fit(X_train_scaled, y_train_fold)
            
            # Evaluate on test fold (completely independent)
            y_pred_fold = best_model.predict(X_test_scaled)
            
            # Calculate metrics for this fold
            fold_metrics = {
                'fold': fold_idx + 1,
                'best_model': best_model_name,
                'best_params': best_params,
                'inner_cv_score': best_inner_score,
                'accuracy': accuracy_score(y_test_fold, y_pred_fold),
                'f1_macro': f1_score(y_test_fold, y_pred_fold, average='macro'),
                'f1_weighted': f1_score(y_test_fold, y_pred_fold, average='weighted'),
                'precision_macro': precision_score(y_test_fold, y_pred_fold, average='macro'),
                'recall_macro': recall_score(y_test_fold, y_pred_fold, average='macro'),
                'classification_report': classification_report(y_test_fold, y_pred_fold, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test_fold, y_pred_fold).tolist()
            }
            
            # Calculate AUC if possible
            if hasattr(best_model, 'predict_proba'):
                try:
                    y_prob_fold = best_model.predict_proba(X_test_scaled)
                    lb = LabelBinarizer()
                    y_binary = lb.fit_transform(y_test_fold)
                    
                    if y_binary.shape[1] > 1:
                        auc_score = roc_auc_score(y_binary, y_prob_fold, average='macro', multi_class='ovr')
                    else:
                        auc_score = roc_auc_score(y_binary, y_prob_fold[:, 1])
                    
                    fold_metrics['auc_macro'] = auc_score
                except:
                    fold_metrics['auc_macro'] = None
            else:
                fold_metrics['auc_macro'] = None
            
            fold_results.append(fold_metrics)
            all_y_true.extend(y_test_fold.tolist())
            all_y_pred.extend(y_pred_fold.tolist())
            
            print(f"  Fold {fold_idx + 1} Results:")
            print(f"    Model: {best_model_name}")
            print(f"    Inner CV F1: {best_inner_score:.4f}")
            print(f"    Test Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"    Test F1-macro: {fold_metrics['f1_macro']:.4f}")
            if fold_metrics['auc_macro']:
                print(f"    Test AUC-macro: {fold_metrics['auc_macro']:.4f}")
        
        # Calculate overall statistics
        overall_metrics = {
            'accuracy': np.mean([fold['accuracy'] for fold in fold_results]),
            'accuracy_std': np.std([fold['accuracy'] for fold in fold_results]),
            'f1_macro': np.mean([fold['f1_macro'] for fold in fold_results]),
            'f1_macro_std': np.std([fold['f1_macro'] for fold in fold_results]),
            'f1_weighted': np.mean([fold['f1_weighted'] for fold in fold_results]),
            'f1_weighted_std': np.std([fold['f1_weighted'] for fold in fold_results]),
            'precision_macro': np.mean([fold['precision_macro'] for fold in fold_results]),
            'precision_macro_std': np.std([fold['precision_macro'] for fold in fold_results]),
            'recall_macro': np.mean([fold['recall_macro'] for fold in fold_results]),
            'recall_macro_std': np.std([fold['recall_macro'] for fold in fold_results])
        }
        
        # Add AUC if available
        auc_scores = [fold['auc_macro'] for fold in fold_results if fold['auc_macro'] is not None]
        if auc_scores:
            overall_metrics['auc_macro'] = np.mean(auc_scores)
            overall_metrics['auc_macro_std'] = np.std(auc_scores)
        
        # Overall confusion matrix and classification report
        overall_cm = confusion_matrix(all_y_true, all_y_pred)
        overall_classification_report = classification_report(all_y_true, all_y_pred, output_dict=True)
        
        nested_results = {
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'overall_confusion_matrix': overall_cm.tolist(),
            'overall_classification_report': overall_classification_report,
            'all_predictions': {'y_true': all_y_true, 'y_pred': all_y_pred}
        }
        
        return nested_results
    
    def train_final_model(self, X, y):
        """
        Train final model on complete dataset for deployment
        
        Parameters:
        -----------
        X : pd.DataFrame
            Complete feature matrix (pre-selected features)
        y : pd.Series
            Complete target vector
            
        Returns:
        --------
        final_model : dict
            Final trained model information
        """
        print("\nTraining final model on complete dataset...")
        
        # Hyperparameter optimization on complete dataset
        best_model, best_model_name, best_params, best_score, scaler = self.inner_cv_hyperparameter_optimization(X, y)
        
        # Train on complete dataset
        X_scaled = scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        self.final_model = {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': scaler,
            'best_params': best_params,
            'cv_score': best_score,
            'n_features': X.shape[1]
        }
        
        print(f"Final model: {best_model_name}")
        print(f"Best parameters: {best_params}")
        print(f"CV F1-score: {best_score:.4f}")
        
        return self.final_model
    
    def create_visualizations(self, nested_results, y):
        """
        Create comprehensive visualizations
        """
        print("Creating visualizations...")
        
        # 1. Cross-fold performance
        self._plot_cross_fold_performance(nested_results)
        
        # 2. Overall confusion matrix
        self._plot_overall_confusion_matrix(nested_results, y)
        
        # 3. Model selection frequency
        self._plot_model_selection_frequency(nested_results)
        
        # 4. Performance comparison
        self._plot_performance_comparison(nested_results)
    
    def _plot_cross_fold_performance(self, nested_results):
        """Plot performance across CV folds"""
        fold_results = nested_results['fold_results']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        folds = [f"Fold {r['fold']}" for r in fold_results]
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1_macro'] for r in fold_results]
        
        # Accuracy plot
        bars1 = axes[0].bar(folds, accuracies, alpha=0.7, color='skyblue')
        axes[0].axhline(y=nested_results['overall_metrics']['accuracy'], 
                       color='red', linestyle='--', label=f"Mean: {nested_results['overall_metrics']['accuracy']:.3f}")
        axes[0].set_title('Accuracy Across CV Folds')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add model names as text on bars
        for bar, fold in zip(bars1, fold_results):
            model_name = fold['best_model']
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       model_name, ha='center', va='bottom', fontsize=8, rotation=90)
        
        # F1-macro plot
        bars2 = axes[1].bar(folds, f1_scores, alpha=0.7, color='lightgreen')
        axes[1].axhline(y=nested_results['overall_metrics']['f1_macro'], 
                       color='red', linestyle='--', label=f"Mean: {nested_results['overall_metrics']['f1_macro']:.3f}")
        axes[1].set_title('F1-Macro Across CV Folds')
        axes[1].set_ylabel('F1-Macro Score')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add model names as text on bars
        for bar, fold in zip(bars2, fold_results):
            model_name = fold['best_model']
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       model_name, ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/cross_fold_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_overall_confusion_matrix(self, nested_results, y):
        """Plot overall confusion matrix"""
        cm = np.array(nested_results['overall_confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y.unique()),
                   yticklabels=sorted(y.unique()))
        plt.title('Overall Confusion Matrix (Nested CV)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/overall_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_model_selection_frequency(self, nested_results):
        """Plot frequency of model selection across folds"""
        model_counts = {}
        for fold in nested_results['fold_results']:
            model = fold['best_model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        plt.figure(figsize=(10, 6))
        models = list(model_counts.keys())
        counts = list(model_counts.values())
        
        plt.bar(models, counts, alpha=0.7)
        plt.title('Model Selection Frequency Across CV Folds')
        plt.xlabel('Model Type')
        plt.ylabel('Frequency')
        plt.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(counts)
        for i, (model, count) in enumerate(zip(models, counts)):
            plt.text(i, count + 0.1, f'{count/total*100:.1f}%', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_selection_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_performance_comparison(self, nested_results):
        """Plot comparison of inner CV vs outer CV performance"""
        fold_results = nested_results['fold_results']
        
        inner_scores = [fold['inner_cv_score'] for fold in fold_results]
        outer_scores = [fold['f1_macro'] for fold in fold_results]
        folds = [f"Fold {fold['fold']}" for fold in fold_results]
        
        x = np.arange(len(folds))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, inner_scores, width, label='Inner CV F1-macro', alpha=0.8)
        bars2 = ax.bar(x + width/2, outer_scores, width, label='Outer CV F1-macro', alpha=0.8)
        
        ax.set_xlabel('CV Folds')
        ax.set_ylabel('F1-macro Score')
        ax.set_title('Inner CV vs Outer CV Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, nested_results):
        """Save all results to JSON files"""
        print("Saving results...")
        
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Support Vector Machine (Nested CV)',
            'methodology': 'Nested Cross-Validation with Pre-selected Features',
            'feature_selection': 'Pre-selected from Steps 1&2',
            'n_features': len(nested_results['fold_results'][0]['best_params']) if nested_results['fold_results'] else 0,
            'outer_cv_folds': self.outer_cv_folds,
            'inner_cv_folds': self.inner_cv_folds,
            'parameters': {
                'outer_cv_folds': self.outer_cv_folds,
                'inner_cv_folds': self.inner_cv_folds,
                'random_state': self.random_state
            },
            'nested_cv_results': nested_results
        }
        
        # Save to JSON
        with open(f'{self.results_dir}/svm_nested_cv_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save final model info if available
        if self.final_model:
            final_model_info = {
                'model_name': self.final_model['model_name'],
                'best_params': self.final_model['best_params'],
                'cv_score': self.final_model['cv_score'],
                'n_features': self.final_model['n_features']
            }
            
            with open(f'{self.results_dir}/final_model_info.json', 'w') as f:
                json.dump(final_model_info, f, indent=2)
        
        print(f"Results saved to {self.results_dir}/")
    
    def run_complete_pipeline(self, X, y):
        """
        Run the complete nested CV pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (pre-selected features from Steps 1&2)
        y : pd.Series
            Target vector
            
        Returns:
        --------
        nested_results : dict
            Complete nested CV results
        """
        print("="*80)
        print("SVM NESTED CROSS-VALIDATION PIPELINE")
        print("="*80)
        print(f"Using pre-selected features from Steps 1&2")
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Outer CV: {self.outer_cv_folds} folds")
        print(f"Inner CV: {self.inner_cv_folds} folds")
        
        # Nested cross-validation
        nested_results = self.nested_cross_validation(X, y)
        
        # Train final model
        final_model = self.train_final_model(X, y)
        
        # Create visualizations
        self.create_visualizations(nested_results, y)
        
        # Save results
        self.save_results(nested_results)
        
        # Print summary
        overall = nested_results['overall_metrics']
        print("="*80)
        print("NESTED CV RESULTS SUMMARY")
        print("="*80)
        print(f"Accuracy: {overall['accuracy']:.4f} ± {overall['accuracy_std']:.4f}")
        print(f"F1-Macro: {overall['f1_macro']:.4f} ± {overall['f1_macro_std']:.4f}")
        print(f"F1-Weighted: {overall['f1_weighted']:.4f} ± {overall['f1_weighted_std']:.4f}")
        print(f"Precision-Macro: {overall['precision_macro']:.4f} ± {overall['precision_macro_std']:.4f}")
        print(f"Recall-Macro: {overall['recall_macro']:.4f} ± {overall['recall_macro_std']:.4f}")
        if 'auc_macro' in overall:
            print(f"AUC-Macro: {overall['auc_macro']:.4f} ± {overall['auc_macro_std']:.4f}")
        print("="*80)
        
        return nested_results

def run_svm_nested_cv(X, y, results_dir='results/step3_models/svm_nested'):
    """
    Run SVM classification with nested cross-validation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix (pre-selected features from Steps 1&2)
    y : pd.Series
        Target vector
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    nested_results : dict
        Nested CV results
    classifier : NestedSVMClassifier
        Trained classifier object
    """
    classifier = NestedSVMClassifier(results_dir=results_dir)
    nested_results = classifier.run_complete_pipeline(X, y)
    return nested_results, classifier