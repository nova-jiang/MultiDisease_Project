import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from feature_selection import run_step1
import warnings
warnings.filterwarnings('ignore')

class SVMFeatureSelection:
    """
    SVM-assisted feature selection and ranking for microbiome multi-class classification
    """
    
    def __init__(self, 
                 C=1.0, 
                 kernel='linear', 
                 max_features=50, 
                 cv_folds=5, 
                 random_state=42,
                 feature_selection_method='coef'):
        """
        Initialize SVM feature selection pipeline
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter for SVM
        kernel : str, default='linear'
            Kernel type for SVM ('linear', 'rbf', 'poly')
        max_features : int, default=50
            Maximum number of top features to select
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        feature_selection_method : str, default='coef'
            Method for feature importance ('coef', 'rfe', 'stability')
        """
        self.C = C
        self.kernel = kernel
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.feature_selection_method = feature_selection_method
        
        self.svm_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_rankings = None
        self.selected_features = None
        self.cv_results = None
        
    def load_filtered_data(self, X_filtered, y, feature_list=None):
        """
        Load the filtered data from Step 1
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered microbiome data from biological pre-filtering
        y : pd.Series
            Category labels
        feature_list : list, optional
            Specific list of features to use (if None, use all features in X_filtered)
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed feature matrix
        y_encoded : np.ndarray
            Encoded labels
        """
        print("Loading and processing filtered data...")
        
        # Use specific features if provided, otherwise use all
        if feature_list is not None:
            available_features = [f for f in feature_list if f in X_filtered.columns]
            if len(available_features) != len(feature_list):
                missing = set(feature_list) - set(available_features)
                print(f"Warning: {len(missing)} features not found in data: {list(missing)[:5]}...")
            X_selected = X_filtered[available_features]
        else:
            X_selected = X_filtered.copy()
            
        print(f"Using {X_selected.shape[1]} features and {X_selected.shape[0]} samples")
        print(f"Category distribution: {y.value_counts().to_dict()}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Store feature names for later reference
        self.feature_names = X_selected.columns.tolist()
        
        return X_scaled, y_encoded
    
    def coefficient_based_ranking(self, X, y):
        """
        Rank features based on SVM coefficient importance
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        feature_importance : pd.DataFrame
            Features ranked by importance
        """
        print("Step 2a: Coefficient-based feature ranking...")
        
        # Train linear SVM for multi-class classification
        svm = SVC(kernel='linear', C=self.C, random_state=self.random_state)
        svm.fit(X, y)
        
        # For multi-class SVM, coef_ has shape (n_classes * (n_classes-1) / 2, n_features)
        # We take the mean absolute value across all binary classifiers
        if len(np.unique(y)) > 2:
            coef_importance = np.abs(svm.coef_).mean(axis=0)
        else:
            coef_importance = np.abs(svm.coef_[0])
            
        # Create importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coef_importance
        }).sort_values('importance', ascending=False)
        
        print(f"Top 10 features by SVM coefficient importance:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<40} {row['importance']:.6f}")
            
        return feature_importance
    
    def rfe_based_ranking(self, X, y):
        """
        Rank features using Recursive Feature Elimination with Cross-Validation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
            
        Returns:
        --------
        rfe_results : pd.DataFrame
            Features ranked by RFE importance
        """
        print("Step 2b: RFE-based feature ranking...")
        
        # Use RFECV to find optimal number of features
        svm = SVC(kernel='linear', C=self.C, random_state=self.random_state)
        
        # Use smaller step size for more granular selection
        min_features = max(1, min(10, len(self.feature_names) // 10))
        max_features_rfe = min(self.max_features * 2, len(self.feature_names))
        
        rfecv = RFECV(
            estimator=svm,
            step=1,
            min_features_to_select=min_features,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        rfecv.fit(X, y)
        
        print(f"Optimal number of features: {rfecv.n_features_}")
        print(f"Cross-validation score with optimal features: {rfecv.grid_scores_[rfecv.n_features_-min_features]:.4f}")
        
        # Create RFE results DataFrame
        rfe_results = pd.DataFrame({
            'feature': self.feature_names,
            'selected': rfecv.support_,
            'ranking': rfecv.ranking_
        }).sort_values('ranking')
        
        print(f"Top 10 features by RFE ranking:")
        for i, (_, row) in enumerate(rfe_results.head(10).iterrows()):
            status = "✓" if row['selected'] else "✗"
            print(f"  {i+1:2d}. {row['feature']:<40} Rank: {row['ranking']:2d} {status}")
            
        return rfe_results, rfecv
    
    def stability_based_ranking(self, X, y, n_bootstrap=50):
        """
        Rank features based on stability across bootstrap samples
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        n_bootstrap : int, default=50
            Number of bootstrap iterations
            
        Returns:
        --------
        stability_results : pd.DataFrame
            Features ranked by stability
        """
        print("Step 2c: Stability-based feature ranking...")
        
        feature_selection_counts = defaultdict(int)
        importance_scores = defaultdict(list)
        
        for i in range(n_bootstrap):
            if (i + 1) % 10 == 0:
                print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
                
            # Bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train SVM
            svm = SVC(kernel='linear', C=self.C, random_state=self.random_state + i)
            svm.fit(X_boot, y_boot)
            
            # Get feature importance
            if len(np.unique(y_boot)) > 2:
                coef_importance = np.abs(svm.coef_).mean(axis=0)
            else:
                coef_importance = np.abs(svm.coef_[0])
                
            # Select top features
            top_indices = np.argsort(coef_importance)[-self.max_features:]
            
            for idx in top_indices:
                feature_selection_counts[self.feature_names[idx]] += 1
                importance_scores[self.feature_names[idx]].append(coef_importance[idx])
        
        # Calculate stability metrics
        stability_results = []
        for feature in self.feature_names:
            selection_frequency = feature_selection_counts[feature] / n_bootstrap
            mean_importance = np.mean(importance_scores[feature]) if importance_scores[feature] else 0
            std_importance = np.std(importance_scores[feature]) if len(importance_scores[feature]) > 1 else 0
            
            stability_results.append({
                'feature': feature,
                'selection_frequency': selection_frequency,
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'stability_score': selection_frequency * mean_importance  # Combined metric
            })
        
        stability_df = pd.DataFrame(stability_results).sort_values('stability_score', ascending=False)
        
        print(f"Top 10 features by stability ranking:")
        for i, (_, row) in enumerate(stability_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<40} "
                  f"Freq: {row['selection_frequency']:.3f} "
                  f"Imp: {row['mean_importance']:.6f}")
            
        return stability_df
    
    def combine_rankings(self, coef_ranking, rfe_ranking=None, stability_ranking=None):
        """
        Combine different ranking methods to get final feature selection
        
        Parameters:
        -----------
        coef_ranking : pd.DataFrame
            Coefficient-based ranking
        rfe_ranking : pd.DataFrame, optional
            RFE-based ranking
        stability_ranking : pd.DataFrame, optional
            Stability-based ranking
            
        Returns:
        --------
        final_ranking : pd.DataFrame
            Combined feature ranking
        """
        print("Step 2d: Combining feature rankings...")
        
        # Start with coefficient ranking
        final_df = coef_ranking[['feature', 'importance']].copy()
        final_df['coef_rank'] = range(1, len(final_df) + 1)
        
        # Add RFE ranking if available
        if rfe_ranking is not None:
            rfe_dict = dict(zip(rfe_ranking['feature'], rfe_ranking['ranking']))
            final_df['rfe_rank'] = final_df['feature'].map(rfe_dict)
        
        # Add stability ranking if available
        if stability_ranking is not None:
            stability_dict = dict(zip(stability_ranking['feature'], range(1, len(stability_ranking) + 1)))
            final_df['stability_rank'] = final_df['feature'].map(stability_dict)
        
        # Calculate combined score (lower is better for ranks)
        rank_cols = [col for col in final_df.columns if col.endswith('_rank')]
        if len(rank_cols) > 1:
            # Normalize ranks to 0-1 scale
            for col in rank_cols:
                final_df[f'{col}_norm'] = 1 - (final_df[col] - 1) / (final_df[col].max() - 1)
            
            # Combined score as average of normalized ranks
            norm_cols = [col for col in final_df.columns if col.endswith('_norm')]
            final_df['combined_score'] = final_df[norm_cols].mean(axis=1)
            final_df = final_df.sort_values('combined_score', ascending=False)
        else:
            final_df['combined_score'] = 1 - (final_df['coef_rank'] - 1) / (final_df['coef_rank'].max() - 1)
        
        print(f"Final top 10 features (combined ranking):")
        for i, (_, row) in enumerate(final_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<40} Score: {row['combined_score']:.4f}")
        
        return final_df
    
    def evaluate_feature_subsets(self, X, y, final_ranking):
        """
        Evaluate different feature subset sizes using cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Encoded labels
        final_ranking : pd.DataFrame
            Final feature ranking
            
        Returns:
        --------
        cv_results : pd.DataFrame
            Cross-validation results for different feature subset sizes
        """
        print("Step 2e: Evaluating feature subset sizes...")
        
        feature_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
        feature_sizes = [size for size in feature_sizes if size <= len(final_ranking)]
        
        cv_results = []
        
        for n_features in feature_sizes:
            print(f"  Evaluating top {n_features} features...")
            
            # Select top n features
            top_features = final_ranking.head(n_features)['feature'].tolist()
            feature_indices = [self.feature_names.index(f) for f in top_features if f in self.feature_names]
            X_subset = X[:, feature_indices]
            
            # Cross-validation
            svm = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
            cv_scores = cross_val_score(
                svm, X_subset, y,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='accuracy'
            )
            
            cv_results.append({
                'n_features': n_features,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'min_accuracy': cv_scores.min(),
                'max_accuracy': cv_scores.max()
            })
        
        cv_df = pd.DataFrame(cv_results)
        
        # Find optimal number of features
        optimal_idx = cv_df['mean_accuracy'].idxmax()
        optimal_n_features = cv_df.loc[optimal_idx, 'n_features']
        optimal_accuracy = cv_df.loc[optimal_idx, 'mean_accuracy']
        
        print(f"\nOptimal number of features: {optimal_n_features}")
        print(f"Optimal CV accuracy: {optimal_accuracy:.4f} ± {cv_df.loc[optimal_idx, 'std_accuracy']:.4f}")
        
        return cv_df, optimal_n_features
    
    def visualize_results(self, final_ranking, cv_results, optimal_n_features):
        """
        Create visualizations for SVM feature selection results
        
        Parameters:
        -----------
        final_ranking : pd.DataFrame
            Final feature ranking
        cv_results : pd.DataFrame
            Cross-validation results
        optimal_n_features : int
            Optimal number of features
        """
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SVM Feature Selection Results', fontsize=16)
        
        # 1. Top features importance
        top_features = final_ranking.head(20)
        axes[0, 0].barh(range(len(top_features)), top_features['combined_score'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels([f[:30] + '...' if len(f) > 30 else f 
                                   for f in top_features['feature']], fontsize=8)
        axes[0, 0].set_xlabel('Combined Importance Score')
        axes[0, 0].set_title('Top 20 Features by Combined Ranking')
        axes[0, 0].invert_yaxis()
        
        # 2. Feature importance distribution
        axes[0, 1].hist(final_ranking['combined_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(final_ranking.head(optimal_n_features)['combined_score'].min(), 
                          color='red', linestyle='--', 
                          label=f'Top {optimal_n_features} threshold')
        axes[0, 1].set_xlabel('Combined Importance Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Feature Importance Distribution')
        axes[0, 1].legend()
        
        # 3. Cross-validation performance vs number of features
        axes[1, 0].errorbar(cv_results['n_features'], cv_results['mean_accuracy'], 
                           yerr=cv_results['std_accuracy'], marker='o', capsize=5)
        axes[1, 0].axvline(optimal_n_features, color='red', linestyle='--', 
                          label=f'Optimal: {optimal_n_features} features')
        axes[1, 0].set_xlabel('Number of Features')
        axes[1, 0].set_ylabel('Cross-Validation Accuracy')
        axes[1, 0].set_title('Model Performance vs Feature Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature ranking comparison (if multiple methods used)
        rank_cols = [col for col in final_ranking.columns if col.endswith('_rank')]
        if len(rank_cols) > 1:
            for i, col in enumerate(rank_cols):
                axes[1, 1].scatter(final_ranking[rank_cols[0]], final_ranking[col], 
                                  alpha=0.6, label=col.replace('_rank', '').upper())
            axes[1, 1].set_xlabel(rank_cols[0].replace('_rank', '').upper() + ' Rank')
            axes[1, 1].set_ylabel('Other Method Ranks')
            axes[1, 1].set_title('Feature Ranking Methods Comparison')
            axes[1, 1].legend()
        else:
            # Show top features in a different way
            top_20 = final_ranking.head(20)
            colors = ['red' if i < optimal_n_features else 'gray' for i in range(len(top_20))]
            axes[1, 1].bar(range(len(top_20)), top_20['importance'], color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Feature Rank')
            axes[1, 1].set_ylabel('SVM Coefficient Importance')
            axes[1, 1].set_title(f'Top 20 Features (Red = Selected Top {optimal_n_features})')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, X_filtered, y, feature_list=None):
        """
        Run the complete SVM feature selection pipeline
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered microbiome data from Step 1
        y : pd.Series
            Category labels
        feature_list : list, optional
            Specific features to use (if None, use all from X_filtered)
            
        Returns:
        --------
        selected_features : list
            Final selected feature names
        final_ranking : pd.DataFrame
            Complete feature ranking
        optimal_n_features : int
            Optimal number of features
        """
        print("="*60)
        print("SVM FEATURE SELECTION PIPELINE")
        print("="*60)
        
        # Load and process data
        X_processed, y_encoded = self.load_filtered_data(X_filtered, y, feature_list)
        
        # Method 1: Coefficient-based ranking
        coef_ranking = self.coefficient_based_ranking(X_processed, y_encoded)
        
        # Method 2: RFE-based ranking (optional, can be time-consuming)
        rfe_ranking = None
        if self.feature_selection_method in ['rfe', 'combined']:
            rfe_ranking, rfecv_model = self.rfe_based_ranking(X_processed, y_encoded)
        
        # Method 3: Stability-based ranking (optional)
        stability_ranking = None
        if self.feature_selection_method in ['stability', 'combined']:
            stability_ranking = self.stability_based_ranking(X_processed, y_encoded)
        
        # Combine rankings
        final_ranking = self.combine_rankings(coef_ranking, rfe_ranking, stability_ranking)
        
        # Evaluate different feature subset sizes
        cv_results, optimal_n_features = self.evaluate_feature_subsets(X_processed, y_encoded, final_ranking)
        
        # Select final features
        selected_features = final_ranking.head(optimal_n_features)['feature'].tolist()
        
        # Store results
        self.feature_rankings = final_ranking
        self.selected_features = selected_features
        self.cv_results = cv_results
        
        # Visualize results
        self.visualize_results(final_ranking, cv_results, optimal_n_features)
        
        print("="*60)
        print("SVM FEATURE SELECTION COMPLETED!")
        print(f"Selected {len(selected_features)} optimal features")
        print(f"Expected CV accuracy: {cv_results[cv_results['n_features']==optimal_n_features]['mean_accuracy'].iloc[0]:.4f}")
        print("="*60)
        
        return selected_features, final_ranking, optimal_n_features

if __name__ == "__main__":
    # Load results from Step 1 
    X_filtered, y, selected_features_step1, results_df = bio_filter.run_complete_pipeline('gmrepo_cleaned_dataset.csv')
    
    # Initialize SVM feature selection
    svm_selector = SVMFeatureSelection(
        C=1.0,                           # SVM regularization parameter
        kernel='linear',                 # Use linear kernel for interpretability
        max_features=50,                 # Maximum features to consider
        cv_folds=5,                      # 5-fold cross-validation
        feature_selection_method='coef'  # Use coefficient-based selection (fastest)
        # Options: 'coef', 'rfe', 'stability', 'combined'
    )
    
   # Run SVM feature selection pipeline
    selected_features_final, feature_rankings, optimal_n = svm_selector.run_complete_pipeline(
        X_filtered, y
    )
    
   # Save results
    feature_rankings.to_csv('svm_feature_rankings.csv', index=False)
    
    print(f"\nFinal selected features ({len(selected_features_final)}):")
    for i, feature in enumerate(selected_features_final, 1):
        print(f"{i:2d}. {feature}")