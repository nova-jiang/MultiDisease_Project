"""
Step 3: KNN Model Training with Nested Cross-Validation
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, f1_score, precision_score, recall_score,
                           roc_auc_score)

import matplotlib
matplotlib.use('Agg')   # non-interactive backend that doesn’t require Tcl/Tk
import matplotlib.pyplot as plt
import seaborn as sns

class NestedKNNClassifier:
    """
    KNN classifier with nested cross-validation for unbiased performance estimation
    
    Uses pre-selected features from Steps 1&2
    Outer CV: Performance estimation on completely independent test folds
    Inner CV: Hyperparameter optimization only
    """
    
    def __init__(self, 
                 outer_cv_folds=5,
                 inner_cv_folds=3,
                 random_state=42,
                 results_dir='results/step3_models/knn_nested'):
        """
        Initialize nested KNN classifier
        
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
        
    def determine_k_range(self, n_samples, n_classes):
        """
        Determine appropriate K range based on data characteristics
        
        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_classes : int
            Number of classes
            
        Returns:
        --------
        k_range : list
            Appropriate K values to test
        """
        # Rule of thumb: K should be odd, less than sqrt(n_samples)
        max_k = min(int(np.sqrt(n_samples)), 50)
        
        # Ensure K is odd and at least n_classes
        k_range = []
        for k in range(max(3, n_classes), max_k + 1, 2):  # Start from max(3, n_classes), step by 2
            k_range.append(k)
        
        # Add some even numbers for completeness, but limit total range
        for k in range(max(4, n_classes), max_k + 1, 2):
            if k not in k_range:
                k_range.append(k)
        
        # Sort and limit to reasonable number for nested CV efficiency
        k_range = sorted(k_range)[:12]  # Limit to 12 values for efficiency
        
        return k_range
    
    def get_knn_parameter_grids(self, n_samples, n_classes):
        """
        Define parameter grids for different KNN variants
        
        Parameters:
        -----------
        n_samples : int
            Number of training samples
        n_classes : int
            Number of classes
            
        Returns:
        --------
        param_grids : dict
            Dictionary of parameter grids for each distance metric
        """
        k_range = self.determine_k_range(n_samples, n_classes)
        
        return {
            'Euclidean': {
                'estimator': KNeighborsClassifier(metric='euclidean'),
                'params': {
                    'n_neighbors': k_range,
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto']
                }
            },
            'Manhattan': {
                'estimator': KNeighborsClassifier(metric='manhattan'),
                'params': {
                    'n_neighbors': k_range,
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto']
                }
            },
            'Minkowski': {
                'estimator': KNeighborsClassifier(metric='minkowski'),
                'params': {
                    'n_neighbors': k_range[:8],  # Fewer K values for Minkowski to save time
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2, 3],  # p=1: Manhattan, p=2: Euclidean, p=3: Minkowski
                    'algorithm': ['auto']
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
        
        # Standardize features (crucial for KNN - distance-based algorithm)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        n_samples, n_features = X_train.shape
        n_classes = len(y_train.unique())
        
        param_grids = self.get_knn_parameter_grids(n_samples, n_classes)
        inner_cv = StratifiedKFold(n_splits=self.inner_cv_folds, shuffle=True, 
                                 random_state=self.random_state)
        
        best_score = -1
        best_model = None
        best_model_name = None
        best_params = None
        
        for model_name, config in param_grids.items():
            try:
                print(f"      Optimizing {model_name} distance...")
                
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
                
                print(f"        Best {model_name} F1: {grid_search.best_score_:.4f}, K={grid_search.best_params_['n_neighbors']}")
                
            except Exception as e:
                print(f"        Warning: {model_name} optimization failed: {e}")
                continue
        
        if best_model is None:
            # Fallback to simple KNN
            print("      Using fallback KNN")
            best_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            best_model_name = "KNN_fallback"
            best_params = {'n_neighbors': 5, 'metric': 'euclidean', 'weights': 'uniform'}
            best_score = 0
        
        print(f"    Best inner CV model: {best_model_name} (F1: {best_score:.4f}, K={best_params.get('n_neighbors', 'N/A')})")
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
        print("NESTED CROSS-VALIDATION FOR KNN")
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
            y_prob_fold = best_model.predict_proba(X_test_scaled)
            
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
            
            # Calculate AUC
            try:
                lb = LabelBinarizer()
                y_binary = lb.fit_transform(y_test_fold)
                
                if y_binary.shape[1] > 1:
                    auc_score = roc_auc_score(y_binary, y_prob_fold, average='macro', multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_binary, y_prob_fold[:, 1])
                
                fold_metrics['auc_macro'] = auc_score
            except:
                fold_metrics['auc_macro'] = None
            
            fold_results.append(fold_metrics)
            all_y_true.extend(y_test_fold.tolist())
            all_y_pred.extend(y_pred_fold.tolist())
            
            print(f"  Fold {fold_idx + 1} Results:")
            print(f"    Model: {best_model_name}")
            print(f"    Best K: {best_params.get('n_neighbors', 'N/A')}")
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
    
    def analyze_k_selection_patterns(self, nested_results):
        """
        Analyze K value selection patterns across folds
        
        Parameters:
        -----------
        nested_results : dict
            Results from nested cross-validation
            
        Returns:
        --------
        k_analysis : dict
            Analysis of K value selections
        """
        k_values = []
        distance_metrics = []
        weights = []
        
        for fold in nested_results['fold_results']:
            params = fold['best_params']
            k_values.append(params.get('n_neighbors', None))
            weights.append(params.get('weights', None))
            
            # Determine distance metric from model name
            model_name = fold['best_model']
            distance_metrics.append(model_name)
        
        # Filter out None values
        valid_k_values = [k for k in k_values if k is not None]
        
        k_analysis = {
            'k_values_selected': k_values,
            'distance_metrics_selected': distance_metrics,
            'weights_selected': weights,
            'mean_k': np.mean(valid_k_values) if valid_k_values else None,
            'std_k': np.std(valid_k_values) if valid_k_values else None,
            'most_common_k': max(set(valid_k_values), key=valid_k_values.count) if valid_k_values else None,
            'k_range': [min(valid_k_values), max(valid_k_values)] if valid_k_values else None,
            'distance_metric_frequency': {metric: distance_metrics.count(metric) 
                                        for metric in set(distance_metrics)},
            'weights_frequency': {weight: weights.count(weight) 
                                for weight in set(weights) if weight is not None}
        }
        
        return k_analysis
    
    def create_visualizations(self, nested_results, y):
        """
        Create comprehensive visualizations
        """
        print("Creating visualizations...")
        
        # 1. Cross-fold performance
        self._plot_cross_fold_performance(nested_results)
        
        # 2. Overall confusion matrix
        self._plot_overall_confusion_matrix(nested_results, y)
        
        # 3. K value and distance metric analysis
        self._plot_k_and_distance_analysis(nested_results)
        
        # 4. Performance comparison
        self._plot_performance_comparison(nested_results)
    
    def _plot_cross_fold_performance(self, nested_results):
        """Plot performance across CV folds"""
        fold_results = nested_results['fold_results']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        folds = [f"Fold {r['fold']}" for r in fold_results]
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1_macro'] for r in fold_results]
        k_values = [r['best_params'].get('n_neighbors', 0) for r in fold_results]
        
        # Accuracy plot with K values
        bars1 = axes[0].bar(folds, accuracies, alpha=0.7, color='skyblue')
        axes[0].axhline(y=nested_results['overall_metrics']['accuracy'], 
                       color='red', linestyle='--', label=f"Mean: {nested_results['overall_metrics']['accuracy']:.3f}")
        axes[0].set_title('Accuracy Across CV Folds')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add K values as text on bars
        for bar, k in zip(bars1, k_values):
            if k > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'K={k}', ha='center', va='bottom', fontsize=9)
        
        # F1-macro plot with K values
        bars2 = axes[1].bar(folds, f1_scores, alpha=0.7, color='lightgreen')
        axes[1].axhline(y=nested_results['overall_metrics']['f1_macro'], 
                       color='red', linestyle='--', label=f"Mean: {nested_results['overall_metrics']['f1_macro']:.3f}")
        axes[1].set_title('F1-Macro Across CV Folds')
        axes[1].set_ylabel('F1-Macro Score')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add K values as text on bars
        for bar, k in zip(bars2, k_values):
            if k > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'K={k}', ha='center', va='bottom', fontsize=9)
        
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
    
    def _plot_k_and_distance_analysis(self, nested_results):
        """Plot K value and distance metric analysis"""
        k_analysis = self.analyze_k_selection_patterns(nested_results)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # K values distribution
        k_values = [k for k in k_analysis['k_values_selected'] if k is not None]
        if k_values:
            axes[0].hist(k_values, bins=range(min(k_values), max(k_values)+2), 
                        alpha=0.7, edgecolor='black')
            if k_analysis['mean_k']:
                axes[0].axvline(k_analysis['mean_k'], color='red', linestyle='--', 
                               label=f'Mean: {k_analysis["mean_k"]:.1f}')
            axes[0].set_xlabel('K Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('K Value Selection Across Folds')
            axes[0].legend()
        
        # Distance metric frequency
        metrics = list(k_analysis['distance_metric_frequency'].keys())
        frequencies = list(k_analysis['distance_metric_frequency'].values())
        
        bars = axes[1].bar(metrics, frequencies, alpha=0.7)
        axes[1].set_xlabel('Distance Metric')
        axes[1].set_ylabel('Selection Frequency')
        axes[1].set_title('Distance Metric Selection Across Folds')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(frequencies)
        for bar, freq in zip(bars, frequencies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{freq/total*100:.1f}%', ha='center', va='bottom')
        
        # Weights frequency
        if k_analysis['weights_frequency']:
            weights = list(k_analysis['weights_frequency'].keys())
            weight_freqs = list(k_analysis['weights_frequency'].values())
            
            axes[2].bar(weights, weight_freqs, alpha=0.7)
            axes[2].set_xlabel('Weight Scheme')
            axes[2].set_ylabel('Selection Frequency')
            axes[2].set_title('Weight Scheme Selection Across Folds')
            
            # Add percentage labels
            total_weights = sum(weight_freqs)
            for i, (weight, freq) in enumerate(zip(weights, weight_freqs)):
                axes[2].text(i, freq + 0.1, f'{freq/total_weights*100:.1f}%', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/k_and_distance_analysis.png', dpi=300, bbox_inches='tight')
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
        
        # Add K analysis to results
        k_analysis = self.analyze_k_selection_patterns(nested_results)
        
        # Prepare results for JSON serialization
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'K-Nearest Neighbors (Nested CV)',
            'methodology': 'Nested Cross-Validation with Pre-selected Features',
            'feature_selection': 'Pre-selected from Steps 1&2',
            'n_features': nested_results['fold_results'][0]['best_params'] if nested_results['fold_results'] else 0,
            'outer_cv_folds': self.outer_cv_folds,
            'inner_cv_folds': self.inner_cv_folds,
            'parameters': {
                'outer_cv_folds': self.outer_cv_folds,
                'inner_cv_folds': self.inner_cv_folds,
                'random_state': self.random_state
            },
            'nested_cv_results': nested_results,
            'k_analysis': k_analysis
        }
        
        # Save to JSON
        with open(f'{self.results_dir}/knn_nested_cv_results.json', 'w') as f:
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
        print("KNN NESTED CROSS-VALIDATION PIPELINE")
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
        k_analysis = self.analyze_k_selection_patterns(nested_results)
        
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
        if k_analysis['mean_k']:
            print(f"Average K selected: {k_analysis['mean_k']:.1f} ± {k_analysis['std_k']:.1f}")
            print(f"Most common K: {k_analysis['most_common_k']}")
        print(f"Most common distance metric: {max(k_analysis['distance_metric_frequency'], key=k_analysis['distance_metric_frequency'].get)}")
        print("="*80)
        
        return nested_results

def run_knn_nested_cv(X, y, results_dir='results/step3_models/knn_nested'):
    """
    Run KNN classification with nested cross-validation
    
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
    classifier : NestedKNNClassifier
        Trained classifier object
    """
    classifier = NestedKNNClassifier(results_dir=results_dir)
    nested_results = classifier.run_complete_pipeline(X, y)
    return nested_results, classifier