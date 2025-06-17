import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

class KNNModelTraining:
    """
    K-Nearest Neighbors model training pipeline for microbiome multi-class classification
    """
    
    def __init__(self, cv_folds=5, test_size=0.2, random_state=42):
        """
        Initialize KNN training pipeline
        
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
        
    def load_and_prepare_data(self, X_filtered, y, selected_features):
        """
        Load and prepare data for KNN training
        
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
        print("Loading and preparing data for KNN training...")
        
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
        
        # Scale features (very important for KNN!)
        X_scaled = self.scaler.fit_transform(X_selected)
        print("Feature scaling completed (crucial for KNN performance)")
        
        # Store feature names
        self.feature_names = available_features
        
        return X_scaled, y_encoded
    
    def hyperparameter_tuning(self, X, y):
        """
        Perform hyperparameter tuning for KNN
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        best_knn : KNeighborsClassifier
            Best KNN model after hyperparameter tuning
        """
        print("Performing KNN hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25, 31],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'cosine'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]  # For minkowski metric (1=manhattan, 2=euclidean)
        }
        
        # Create stratified CV
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search
        print("  Running grid search (this may take a while)...")
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
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
        grid_results.to_csv('knn_grid_search_results.csv', index=False)
        print("  Grid search results saved to 'knn_grid_search_results.csv'")
        
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
        model : KNeighborsClassifier
            Tuned KNN model
            
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
            return_train_score=True
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
        model : KNeighborsClassifier
            Best KNN model
            
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
        y_pred_proba = model.predict_proba(X_test)
        
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
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        print(f"\nTest Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-score (macro): {f1:.4f}")
        
        return test_results
    
    def analyze_k_values(self, X, y):
        """
        Analyze the effect of different k values on model performance
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        k_analysis : dict
            Analysis results for different k values
        """
        print("Analyzing effect of different k values...")
        
        # Test range of k values
        k_values = range(1, min(51, len(X) // 5))  # Up to 50 or n_samples/5
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        k_scores = []
        k_std = []
        
        for k in k_values:
            if k % 10 == 1:  # Progress indicator
                print(f"  Testing k = {k}...")
                
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.best_params.get('weights', 'uniform'),
                metric=self.best_params.get('metric', 'euclidean')
            )
            
            scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
            k_scores.append(scores.mean())
            k_std.append(scores.std())
        
        k_analysis = {
            'k_values': list(k_values),
            'accuracy_means': k_scores,
            'accuracy_stds': k_std,
            'optimal_k': k_values[np.argmax(k_scores)]
        }
        
        print(f"  Optimal k value: {k_analysis['optimal_k']}")
        print(f"  Best accuracy: {max(k_scores):.4f}")
        
        return k_analysis
    
    def neighbor_analysis(self, X, y):
        """
        Analyze the distribution of neighbors for each class
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        neighbor_stats : dict
            Neighbor analysis statistics
        """
        print("Performing neighbor analysis...")
        
        # Train model on full data for analysis
        self.best_model.fit(X, y)
        
        # Find neighbors for each sample
        distances, indices = self.best_model.kneighbors(X)
        
        # Analyze neighbor class distributions
        neighbor_stats = {
            'class_distribution': {},
            'avg_neighbor_distances': {},
            'neighbor_homogeneity': {}
        }
        
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            class_mask = y == class_idx
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > 0:
                # Get neighbors for this class
                class_neighbors = indices[class_mask]
                class_distances = distances[class_mask]
                
                # Calculate statistics
                neighbor_classes = y[class_neighbors[:, 1:]]  # Exclude self (index 0)
                same_class_ratio = (neighbor_classes == class_idx).mean()
                avg_distance = class_distances[:, 1:].mean()  # Exclude self-distance
                
                neighbor_stats['class_distribution'][class_name] = len(class_indices)
                neighbor_stats['avg_neighbor_distances'][class_name] = avg_distance
                neighbor_stats['neighbor_homogeneity'][class_name] = same_class_ratio
        
        print("  Neighbor homogeneity by class:")
        for class_name, homogeneity in neighbor_stats['neighbor_homogeneity'].items():
            print(f"    {class_name}: {homogeneity:.3f}")
        
        return neighbor_stats
    
    def visualize_results(self, cv_results, test_results, k_analysis, neighbor_stats):
        """
        Create visualizations for KNN results
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        test_results : dict
            Test results
        k_analysis : dict
            K-value analysis results
        neighbor_stats : dict
            Neighbor analysis statistics
        """
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KNN Model Performance Analysis', fontsize=16)
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0, 1])
        axes[0, 1].set_title('Test Set Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. K-value analysis
        axes[0, 2].errorbar(k_analysis['k_values'], k_analysis['accuracy_means'], 
                           yerr=k_analysis['accuracy_stds'], marker='o', capsize=3)
        axes[0, 2].axvline(k_analysis['optimal_k'], color='red', linestyle='--', 
                          label=f'Optimal k = {k_analysis["optimal_k"]}')
        axes[0, 2].axvline(self.best_params['n_neighbors'], color='green', linestyle='--', 
                          label=f'Selected k = {self.best_params["n_neighbors"]}')
        axes[0, 2].set_xlabel('Number of Neighbors (k)')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('K-Value Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Per-class performance
        class_report = test_results['classification_report']
        classes = [cls for cls in class_report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [class_report[cls]['f1-score'] for cls in classes]
        
        axes[1, 0].bar(range(len(classes)), f1_scores, alpha=0.7)
        axes[1, 0].set_xticks(range(len(classes)))
        axes[1, 0].set_xticklabels(classes, rotation=45)
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('Per-Class F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Neighbor homogeneity
        homogeneity_classes = list(neighbor_stats['neighbor_homogeneity'].keys())
        homogeneity_values = list(neighbor_stats['neighbor_homogeneity'].values())
        
        axes[1, 1].bar(range(len(homogeneity_classes)), homogeneity_values, alpha=0.7, color='orange')
        axes[1, 1].set_xticks(range(len(homogeneity_classes)))
        axes[1, 1].set_xticklabels(homogeneity_classes, rotation=45)
        axes[1, 1].set_ylabel('Same-Class Neighbor Ratio')
        axes[1, 1].set_title('Neighbor Class Homogeneity')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Distance distribution by class
        distance_classes = list(neighbor_stats['avg_neighbor_distances'].keys())
        distance_values = list(neighbor_stats['avg_neighbor_distances'].values())
        
        axes[1, 2].bar(range(len(distance_classes)), distance_values, alpha=0.7, color='purple')
        axes[1, 2].set_xticks(range(len(distance_classes)))
        axes[1, 2].set_xticklabels(distance_classes, rotation=45)
        axes[1, 2].set_ylabel('Average Distance to Neighbors')
        axes[1, 2].set_title('Average Neighbor Distances by Class')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('knn_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, cv_results, test_results, k_analysis, neighbor_stats):
        """
        Save all results to files
        
        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        test_results : dict
            Test results
        k_analysis : dict
            K-value analysis results
        neighbor_stats : dict
            Neighbor analysis statistics
        """
        print("Saving results to files...")
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save trained model
        model_filename = f'knn_model_{timestamp}.pkl'
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
        cv_filename = f'knn_cv_results_{timestamp}.json'
        with open(cv_filename, 'w') as f:
            json.dump(cv_results, f, indent=2)
        print(f"  CV results saved to: {cv_filename}")
        
        # 3. Save test results
        test_filename = f'knn_test_results_{timestamp}.json'
        with open(test_filename, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"  Test results saved to: {test_filename}")
        
        # 4. Save K-value analysis
        k_analysis_filename = f'knn_k_analysis_{timestamp}.json'
        with open(k_analysis_filename, 'w') as f:
            json.dump(k_analysis, f, indent=2)
        print(f"  K-value analysis saved to: {k_analysis_filename}")
        
        # 5. Save neighbor analysis
        neighbor_filename = f'knn_neighbor_analysis_{timestamp}.json'
        with open(neighbor_filename, 'w') as f:
            json.dump(neighbor_stats, f, indent=2)
        print(f"  Neighbor analysis saved to: {neighbor_filename}")
        
        # 6. Save summary report
        summary = {
            'model_type': 'KNN',
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
            'optimal_k': k_analysis['optimal_k'],
            'selected_k': self.best_params['n_neighbors'],
            'n_features': len(self.feature_names),
            'n_samples': len(test_results['test_true_labels']) / self.test_size  # Approximate total samples
        }
        
        summary_filename = f'knn_summary_{timestamp}.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary report saved to: {summary_filename}")
    
    def run_complete_pipeline(self, X_filtered, y, selected_features):
        """
        Run the complete KNN training pipeline
        
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
        print("KNN MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X_processed, y_encoded = self.load_and_prepare_data(X_filtered, y, selected_features)
        
        # Hyperparameter tuning
        best_model = self.hyperparameter_tuning(X_processed, y_encoded)
        
        # Cross-validation evaluation
        cv_results = self.cross_validation_evaluation(X_processed, y_encoded, best_model)
        
        # Train final model and test
        test_results = self.train_final_model(X_processed, y_encoded, best_model)
        
        # Analyze K values
        k_analysis = self.analyze_k_values(X_processed, y_encoded)
        
        # Neighbor analysis
        neighbor_stats = self.neighbor_analysis(X_processed, y_encoded)
        
        # Visualize results
        self.visualize_results(cv_results, test_results, k_analysis, neighbor_stats)
        
        # Save all results
        self.save_results(cv_results, test_results, k_analysis, neighbor_stats)
        
        print("="*60)
        print("KNN TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {cv_results['accuracy']['test_mean']:.4f} ± {cv_results['accuracy']['test_std']:.4f}")
        print(f"Optimal k value: {k_analysis['optimal_k']} (selected k: {self.best_params['n_neighbors']})")
        print("="*60)
        
        return {
            'model': self.best_model,
            'cv_results': cv_results,
            'test_results': test_results,
            'k_analysis': k_analysis,
            'neighbor_stats': neighbor_stats,
            'best_params': self.best_params
        }
