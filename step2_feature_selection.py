import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter1d
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelInformedFeatureSelector:
    """
    Model-informed feature selection pipeline
    Phase 1: XGBoost initial feature ranking + importance analysis 
    Phase 2: Recursive Feature Elimination with Cross-Validation (RFECV)
    """
    
    def __init__(self, 
                 xgb_top_k=164,
                 cv_folds=5,
                 random_state=42,
                 results_dir='results/step2_feature_selection'):
        """
        Initialize the feature selection pipeline
        
        Parameters:
        -----------
        xgb_top_k : int, 164
            Number of top features to select from XGBoost
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        results_dir : str
            Directory to save results
        """
        self.xgb_top_k = xgb_top_k
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Results storage
        self.phase1_results = {}
        self.phase2_results = {}
        self.phase3_results = {}
        self.final_selected_features = None
        
    def phase1_xgboost_ranking(self, X, y):
        """
        Phase 1: XGBoost feature importance ranking
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data from Step 1
        y : pd.Series
            Category labels
            
        Returns:
        --------
        top_features : list
            Top-K features selected by XGBoost
        """
        print("="*60)
        print("PHASE 1: XGBoost Feature Importance Ranking")
        print("="*60)
        
        # To ensure data is numeric and handle any potential issues
        X_clean = X.copy()
        
        # remove problematic characters
        print("Cleaning feature names for XGBoost compatibility...")
        original_feature_names = X_clean.columns.tolist()
        
        # Create a mapping of clean names to original names
        clean_feature_names = []
        feature_name_mapping = {}
        
        for i, original_name in enumerate(original_feature_names):
            # clean problematic data
            clean_name = str(original_name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            clean_name = clean_name.replace(' ', '_').replace('(', '_').replace(')', '_')
            clean_name = clean_name.replace(',', '_').replace(':', '_').replace(';', '_')
            clean_name = f"feature_{i}_{clean_name}" if not clean_name[0].isalpha() else clean_name
            
            clean_feature_names.append(clean_name)
            feature_name_mapping[clean_name] = original_name
        
        # Rename columns
        X_clean.columns = clean_feature_names
        
        if not X_clean.select_dtypes(include=[np.number]).shape[1] == X_clean.shape[1]:
            print("Converting non-numeric data to numeric...")
            X_clean = X_clean.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN with median for each column
        if X_clean.isnull().any().any():
            print("Filling NaN values with median...")
            for col in X_clean.columns:
                if X_clean[col].isnull().any():
                    median_val = X_clean[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_clean[col] = X_clean[col].fillna(median_val)
        
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        if X_clean.isnull().any().any():
            print("Handling remaining infinite values...")
            for col in X_clean.columns:
                if X_clean[col].isnull().any():
                    median_val = X_clean[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_clean[col] = X_clean[col].fillna(median_val)
        
        assert not X_clean.isnull().any().any(), "Still have NaN values after cleaning!"
        assert not np.isinf(X_clean.values).any(), "Still have infinite values after cleaning!"
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean), 
            columns=X_clean.columns, 
            index=X_clean.index
        )
        
        # Prepare data for XGBoost
        # Convert string labels to numeric
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"Training XGBoost on {X.shape[1]} features...")
        print(f"Categories: {list(le.classes_)}")
        print(f"Data shape after cleaning: {X_scaled.shape}")
        print(f"Data range: [{X_scaled.min().min():.3f}, {X_scaled.max().max():.3f}]")
        print(f"Feature name mapping created: {len(feature_name_mapping)} features")
        
           
        # XGBoost grid search with simple parameters
        xgb_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
        
        # Initialize XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=self.random_state,
            n_jobs=1,  # Use single thread for stability
            verbosity=0  # Reduce verbosity
        )
        
        # Perform grid search
        cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        print("Performing grid search for optimal XGBoost parameters...")
    
        grid_search = GridSearchCV(
            xgb_clf, 
            xgb_param_grid, 
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=1,
            verbose=1,
            error_score='raise'  # This will help debug errors
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        # Get best model
        best_xgb = grid_search.best_estimator_
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        
        # Get feature importances
        feature_importance = best_xgb.feature_importances_
        
        # Store results
        self.phase1_results = {
            'model_used': 'XGBoost',
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'feature_importance': None,  # Will be filled below
            'top_features': None,  # Will be filled below
            'scaler': scaler,
            'label_encoder': le,
            'feature_name_mapping': feature_name_mapping
        }
            
        
        # Create feature importance DataFrame with original names
        importance_df = pd.DataFrame({
            'feature': [feature_name_mapping[clean_name] for clean_name in X_clean.columns],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Select top-K features
        top_features = importance_df.head(self.xgb_top_k)['feature'].tolist()
        
        print(f"Selected top {len(top_features)} features based on XGBoost importance")
        print(f"Top 10 features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Update results with feature importance
        self.phase1_results['feature_importance'] = importance_df
        self.phase1_results['top_features'] = top_features
        
        # Perform cumulative importance analysis for informational purposes only
        print("\n" + "="*50)
        print("CUMULATIVE IMPORTANCE ANALYSIS (INFORMATIONAL)")
        print("="*50)
        
        optimal_k_suggested, recommendations = self._analyze_cumulative_importance(importance_df, feature_name_mapping)
        
        # Store the analysis results but DON'T change the user's selection
        self.phase1_results['cumulative_analysis'] = {
            'suggested_optimal_k': optimal_k_suggested,
            'recommendations': recommendations,
            'user_specified_k': self.xgb_top_k,
            'actually_used_k': len(top_features)
        }
        
        print(f"\n" + "="*50)
        print("FEATURE SELECTION DECISION")
        print("="*50)
        print(f"Specified xgb_top_k: {self.xgb_top_k}")
        print(f"Actually selecting: {len(top_features)} features")
        
        # Visualize feature importance (traditional plot)
        self._plot_traditional_importance(importance_df)
        
        return top_features
    
    def _plot_traditional_importance(self, importance_df):
        """Plot traditional XGBoost feature importance"""
        plt.figure(figsize=(12, 8))
        
        # Plot top 50 features
        top_50 = importance_df.head(50)
        
        plt.barh(range(len(top_50)), top_50['importance'])
        plt.yticks(range(len(top_50)), top_50['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 50 Features by XGBoost Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/phase1_traditional_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def phase2_rfecv(self, X, y, candidate_features):
        """
        Phase 2: Recursive Feature Elimination with Cross-Validation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Category labels
        candidate_features : list
            Candidate features from Phase 1
            
        Returns:
        --------
        optimal_features : list
            Optimal features selected by RFECV
        """
        print("="*60)
        print("PHASE 2: Recursive Feature Elimination with CV")
        print("="*60)
        
        # Use only candidate features
        X_candidates = X[candidate_features].copy()
        
        print("Cleaning data for RFECV...")
        
        for col in X_candidates.columns:
            X_candidates[col] = pd.to_numeric(X_candidates[col], errors='coerce')
        
        # Handle NaN values
        if X_candidates.isnull().any().any():
            print("Handling NaN values...")
            for col in X_candidates.columns:
                if X_candidates[col].isnull().any():
                    median_val = X_candidates[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_candidates[col] = X_candidates[col].fillna(median_val)
        
        # Handle infinite values
        X_candidates = X_candidates.replace([np.inf, -np.inf], np.nan)
        if X_candidates.isnull().any().any():
            print("Handling infinite values...")
            for col in X_candidates.columns:
                if X_candidates[col].isnull().any():
                    median_val = X_candidates[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_candidates[col] = X_candidates[col].fillna(median_val)

        assert not X_candidates.isnull().any().any(), "Still have NaN values after cleaning!"
        assert not np.isinf(X_candidates.values).any(), "Still have infinite values after cleaning!"
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_candidates), 
            columns=X_candidates.columns, 
            index=X_candidates.index
        )
        assert not X_scaled.isnull().any().any(), "NaN values after scaling!"
        assert not np.isinf(X_scaled.values).any(), "Infinite values after scaling!"
        
        print(f"Running RFECV on {len(candidate_features)} candidate features...")
        print(f"Data shape: {X_scaled.shape}")
        print(f"Data range: [{X_scaled.min().min():.3f}, {X_scaled.max().max():.3f}]")
        
        # Try different estimators for RFECV
        estimators = {
            'LogisticRegression': LogisticRegression(
                penalty='l2', 
                max_iter=1000,  # Increase max_iter
                random_state=self.random_state,
                multi_class='ovr',
                solver='liblinear'  # More stable solver
            ),
            'LinearSVC': LinearSVC(
                penalty='l2', 
                max_iter=2000,  # Increase max_iter
                random_state=self.random_state,
                multi_class='ovr',
                dual=False  # Recommended when n_samples > n_features
            )
        }
        
        cv_strategy = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        rfecv_results = {}
        
        for name, estimator in estimators.items():
            print(f"\nTesting RFECV with {name}...")
            
            try:
                # Test the estimator first
                print(f"  Testing basic fit with {name}...")
                test_estimator = estimator
                test_estimator.fit(X_scaled, y)
                print(f"  Basic fit successful!")
                
                # Perform RFECV
                print(f"  Running RFECV...")
                rfecv = RFECV(
                    estimator=estimator,
                    step=1,
                    cv=cv_strategy,
                    scoring='f1_macro',
                    n_jobs=1,  # Use single thread for stability
                    verbose=1
                )
                
                rfecv.fit(X_scaled, y)
                
                # Get results
                optimal_features = X_candidates.columns[rfecv.support_].tolist()
                cv_scores = rfecv.cv_results_['mean_test_score']
                
                rfecv_results[name] = {
                    'rfecv_object': rfecv,
                    'optimal_features': optimal_features,
                    'n_features': rfecv.n_features_,
                    'cv_scores': cv_scores,
                    'best_score': np.max(cv_scores)
                }
                
                print(f"  Optimal number of features: {rfecv.n_features_}")
                print(f"  Best CV F1-score: {np.max(cv_scores):.4f}")
                print(f"  Selected features: {len(optimal_features)}")
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                continue
        
        # Select best estimator based on CV score
        if rfecv_results:
            best_estimator_name = max(rfecv_results.keys(), 
                                    key=lambda x: rfecv_results[x]['best_score'])
            best_result = rfecv_results[best_estimator_name]
            optimal_features = best_result['optimal_features']
            
            print(f"\nBest estimator: {best_estimator_name}")
            print(f"Best CV F1-score: {best_result['best_score']:.4f}")
            print(f"Final selected features: {len(optimal_features)}")
        else:
            print("ERROR: No RFECV results obtained!")
            # More conservative fallback
            optimal_features = candidate_features[:min(20, len(candidate_features))]
            print(f"Using fallback: top {len(optimal_features)} features from Phase 1")
        
        # Store results
        self.phase2_results = {
            'rfecv_results': rfecv_results,
            'best_estimator': best_estimator_name if rfecv_results else None,
            'optimal_features': optimal_features,
            'scaler': scaler
        }
        
        # Visualize RFECV results
        self._plot_rfecv_results(rfecv_results)
        
        return optimal_features
        
        # Visualize RFECV results
        self._plot_rfecv_results(rfecv_results)
        
        return optimal_features
    
    def _analyze_cumulative_importance(self, importance_df, feature_name_mapping):
        """
        Analyze cumulative feature importance to determine optimal number of features
        
        Parameters:
        -----------
        importance_df : pd.DataFrame
            Feature importance DataFrame with 'feature' and 'importance' columns
        feature_name_mapping : dict
            Mapping from clean names to original names
            
        Returns:
        --------
        optimal_k : int
            Suggested optimal number of features
        """
        print("\n" + "="*60)
        print("CUMULATIVE FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Calculate cumulative importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['cumulative_percentage'] = (importance_df['cumulative_importance'] / importance_df['importance'].sum()) * 100
        
        # Calculate marginal contribution (how much each additional feature adds)
        importance_df['marginal_contribution'] = importance_df['importance'] / importance_df['importance'].sum() * 100
        
        print(f"Total importance sum: {importance_df['importance'].sum():.4f}")
        print(f"Analysis of top features:")
        
        # Key thresholds to analyze
        thresholds = [50, 70, 80, 85, 90, 95, 99]
        threshold_features = {}
        
        for threshold in thresholds:
            features_needed = (importance_df['cumulative_percentage'] >= threshold).idxmax() + 1
            threshold_features[threshold] = features_needed
            print(f"  {threshold}% of importance captured by top {features_needed} features")
        
        # Find the "elbow point" - where marginal contribution drops significantly
        # Use multiple methods for more robust elbow detection
        elbow_points = {}
        
        # Get marginal contribution data
        marginal_contrib = importance_df['marginal_contribution'].values
        
        # Apply smoothing for better analysis
        smoothed_contrib = gaussian_filter1d(marginal_contrib, sigma=1)
        
        # Method 1: Traditional second derivative (but with better preprocessing)
        if len(smoothed_contrib) > 20:
            # Apply stronger smoothing for second derivative
            heavily_smoothed = gaussian_filter1d(marginal_contrib, sigma=3)
            second_deriv = np.diff(heavily_smoothed, n=2)
            # Skip the first 10 features to avoid initial steep drop
            second_deriv_subset = second_deriv[10:]
            elbow_method1 = np.argmax(second_deriv_subset) + 12  # +10 for offset +2 for double diff
            elbow_points['second_derivative'] = min(elbow_method1, len(importance_df)-1)
        
        # Method 2: Percentage change in slope
        changes = np.diff(marginal_contrib)
        pct_changes = np.abs(changes[1:] - changes[:-1]) / (np.abs(changes[:-1]) + 1e-8)
        # Find where percentage change drops below threshold, but skip first 15 features
        start_idx = 15
        try:
            elbow_method2 = start_idx + np.where(pct_changes[start_idx:] < 0.1)[0][0]
            elbow_points['percentage_change'] = min(elbow_method2, len(importance_df)-1)
        except:
            elbow_points['percentage_change'] = min(50, len(importance_df)-1)
        
        # Method 3: Kneedle algorithm (if available) - more robust elbow detection
        try:
            from kneed import KneeLocator
            x_vals = np.arange(1, len(marginal_contrib) + 1)
            y_vals = marginal_contrib
            
            kl = KneeLocator(x_vals, y_vals, curve="convex", direction="decreasing", 
                           interp_method="interp1d", online=True)
            if kl.elbow is not None and kl.elbow > 10:  # Ensure reasonable minimum
                elbow_points['kneedle'] = kl.elbow
        except ImportError:
            print("  Note: kneed library not available, using alternative methods")
        except:
            pass
        
        # Method 4: Diminishing returns threshold
        # Find where marginal contribution drops below X% of the maximum
        max_contrib = marginal_contrib[0]  # First feature has highest contribution
        threshold_5pct = max_contrib * 0.05  # 5% of max contribution
        threshold_10pct = max_contrib * 0.10  # 10% of max contribution
        
        try:
            elbow_5pct = np.where(marginal_contrib < threshold_5pct)[0][0]
            elbow_points['diminishing_5pct'] = elbow_5pct
        except:
            elbow_points['diminishing_5pct'] = len(importance_df)
            
        try:
            elbow_10pct = np.where(marginal_contrib < threshold_10pct)[0][0]
            elbow_points['diminishing_10pct'] = elbow_10pct
        except:
            elbow_points['diminishing_10pct'] = len(importance_df)
        
        # Method 5: Cumulative importance plateau detection
        # Find where the rate of cumulative increase drops significantly
        cum_pct = importance_df['cumulative_percentage'].values
        cum_diff = np.diff(cum_pct)
        # Find where cumulative increase per feature drops below threshold
        try:
            elbow_plateau = np.where(cum_diff < 0.5)[0][0] + 1  # +1 for diff offset
            elbow_points['plateau_detection'] = min(elbow_plateau, len(importance_df)-1)
        except:
            elbow_points['plateau_detection'] = min(100, len(importance_df)-1)
        
        print(f"\nElbow detection methods:")
        for method, point in elbow_points.items():
            print(f"  {method}: {point} features")
        
        # Smart ensemble of methods - remove outliers and take median
        valid_points = [p for p in elbow_points.values() if 10 <= p <= 200]  # Reasonable range
        
        if len(valid_points) >= 2:
            # Remove extreme outliers (beyond 1.5 IQR)
            q25, q75 = np.percentile(valid_points, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            filtered_points = [p for p in valid_points if lower_bound <= p <= upper_bound]
            
            if len(filtered_points) >= 2:
                elbow_point = int(np.median(filtered_points))
            else:
                elbow_point = int(np.median(valid_points))
        else:
            # Fallback to heuristic
            elbow_point = min(60, len(importance_df)//3)
        
        print(f"\nRobust elbow point (median of valid methods): {elbow_point} features")
        
        # Additional heuristics
        print(f"\nAdditional:")
        
        # Rule 1: Features contributing > 1% individually
        significant_features = (importance_df['marginal_contribution'] >= 1.0).sum()
        print(f"  Features with >1% individual contribution: {significant_features}")
        
        # Rule 2: Features contributing > 0.5% individually
        meaningful_features = (importance_df['marginal_contribution'] >= 0.5).sum()
        print(f"  Features with >0.5% individual contribution: {meaningful_features}")
        
        # Rule 3: Features with importance > mean importance
        mean_importance = importance_df['importance'].mean()
        above_mean_features = (importance_df['importance'] > mean_importance).sum()
        print(f"  Features above mean importance: {above_mean_features}")
        
        # Create comprehensive visualization
        self._plot_cumulative_importance_analysis(importance_df, threshold_features, elbow_point, 
                                                significant_features, meaningful_features)
        
        # Make final recommendation
        recommendations = {
            'elbow_point': elbow_point,
            'significant_1pct': significant_features,
            'meaningful_0.5pct': meaningful_features,
            'above_mean': above_mean_features,
            '90_percent_threshold': threshold_features[90]
        }
        
        print(f"\n Importance Summary:")
        print(f"  elbow point: {elbow_point} features")
        print(f"  90% importance: {threshold_features[90]} features")
        
        optimal_k = elbow_point
 
        return optimal_k, recommendations
    
    def _plot_cumulative_importance_analysis(self, importance_df, threshold_features, elbow_point, 
                                           significant_features, meaningful_features):
        """
        Create comprehensive visualization of cumulative importance analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cumulative Feature Importance Analysis', fontsize=16)
        
        # Plot 1: Cumulative importance curve
        x = range(1, len(importance_df) + 1)
        axes[0, 0].plot(x, importance_df['cumulative_percentage'], 'b-', linewidth=2, label='Cumulative Importance')
        
        # Add threshold lines
        for threshold in [80, 85, 90, 95]:
            features_needed = threshold_features[threshold]
            axes[0, 0].axhline(y=threshold, color='gray', linestyle='--', alpha=0.6)
            axes[0, 0].axvline(x=features_needed, color='gray', linestyle='--', alpha=0.6)
            axes[0, 0].plot(features_needed, threshold, 'ro', markersize=6)
            axes[0, 0].annotate(f'{threshold}%: {features_needed} features', 
                              xy=(features_needed, threshold), xytext=(10, 10), 
                              textcoords='offset points', fontsize=8)
        
        # Mark elbow point
        elbow_y = importance_df.iloc[elbow_point-1]['cumulative_percentage']
        axes[0, 0].axvline(x=elbow_point, color='red', linestyle='-', linewidth=2, label=f'Elbow Point ({elbow_point})')
        axes[0, 0].plot(elbow_point, elbow_y, 'ro', markersize=8)
        
        axes[0, 0].set_xlabel('Number of Features')
        axes[0, 0].set_ylabel('Cumulative Importance (%)')
        axes[0, 0].set_title('Cumulative Importance Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, min(200, len(importance_df)))
        
        # Plot 2: Marginal contribution (individual feature importance)
        top_50 = importance_df.head(50)
        axes[0, 1].bar(range(len(top_50)), top_50['marginal_contribution'], alpha=0.7)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='1% threshold')
        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', label='0.5% threshold')
        axes[0, 1].axvline(x=elbow_point-1, color='red', linestyle='-', linewidth=2, label='Elbow Point')
        axes[0, 1].set_xlabel('Feature Rank')
        axes[0, 1].set_ylabel('Individual Contribution (%)')
        axes[0, 1].set_title('Individual Feature Contributions (Top 50)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Log-scale importance decay
        axes[1, 0].semilogy(x, importance_df['importance'], 'g-', linewidth=2)
        axes[1, 0].axvline(x=elbow_point, color='red', linestyle='-', linewidth=2, label='Elbow Point')
        axes[1, 0].axvline(x=significant_features, color='blue', linestyle='--', label=f'>1% features ({significant_features})')
        axes[1, 0].axvline(x=meaningful_features, color='orange', linestyle='--', label=f'>0.5% features ({meaningful_features})')
        axes[1, 0].set_xlabel('Feature Rank')
        axes[1, 0].set_ylabel('Feature Importance (log scale)')
        axes[1, 0].set_title('Feature Importance Decay')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, min(200, len(importance_df)))
        
        # Plot 4: Recommendations comparison
        recommendations = {
            'Elbow Point': elbow_point,
            '85% Threshold': threshold_features[85],
            '90% Threshold': threshold_features[90],
            '>1% Contrib': significant_features,
            '>0.5% Contrib': meaningful_features
        }
        
        methods = list(recommendations.keys())
        values = list(recommendations.values())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        bars = axes[1, 1].bar(methods, values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Feature Selection Recommendations')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/cumulative_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return recommendations
    
    def _plot_rfecv_results(self, rfecv_results):
        """Plot RFECV results"""
        if not rfecv_results:
            return
            
        fig, axes = plt.subplots(1, len(rfecv_results), figsize=(6*len(rfecv_results), 5))
        if len(rfecv_results) == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(rfecv_results.items()):
            cv_scores = result['cv_scores']
            n_features_range = range(1, len(cv_scores) + 1)
            
            axes[i].plot(n_features_range, cv_scores, 'b-', marker='o', markersize=3)
            axes[i].axvline(result['n_features'], color='red', linestyle='--', 
                           label=f'Optimal: {result["n_features"]} features')
            axes[i].set_xlabel('Number of Features')
            axes[i].set_ylabel('CV F1-Score')
            axes[i].set_title(f'RFECV Results - {name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/phase2_rfecv_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results to JSON files"""
        print("Saving results...")
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Prepare data for JSON serialization
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'xgb_top_k': int(self.xgb_top_k),
                'cv_folds': int(self.cv_folds),
                'random_state': int(self.random_state)
            },
            'phase1': {
                'model_used': self.phase1_results.get('model_used', 'unknown'),
                'best_params': convert_numpy_types(self.phase1_results.get('best_params', {})),
                'best_cv_score': float(self.phase1_results.get('best_cv_score', 0)),
                'top_features': self.phase1_results.get('top_features', []),
                'user_specified_k': int(self.xgb_top_k),
                'actually_used_k': len(self.phase1_results.get('top_features', [])),
                'feature_importance': convert_numpy_types(
                    self.phase1_results.get('feature_importance', pd.DataFrame()).to_dict('records') 
                    if isinstance(self.phase1_results.get('feature_importance'), pd.DataFrame) else []
                ),
                'cumulative_analysis': convert_numpy_types(self.phase1_results.get('cumulative_analysis', {}))
            },
            'phase2': {
                'best_estimator': self.phase2_results.get('best_estimator', ''),
                'optimal_features': self.phase2_results.get('optimal_features', []),
                'rfecv_summary': convert_numpy_types({
                    name: {
                        'n_features': int(result['n_features']),
                        'best_score': float(result['best_score'])
                    } for name, result in self.phase2_results.get('rfecv_results', {}).items()
                })
            },
            'final_selected_features': self.final_selected_features or []
        }
        
        # Convert the entire results_summary to handle any remaining numpy types
        results_summary = convert_numpy_types(results_summary)
        
        # Save to JSON
        with open(f'{self.results_dir}/step2_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save feature lists separately
        if self.final_selected_features:
            with open(f'{self.results_dir}/final_selected_features.txt', 'w') as f:
                for feature in self.final_selected_features:
                    f.write(f"{feature}\n")
        
        print(f"Results saved to {self.results_dir}/")
    
    def run_complete_pipeline(self, X, y):
        """
        Run the complete model-informed feature selection pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data from Step 1
        y : pd.Series
            Category labels
            
        Returns:
        --------
        final_features : list
            Final selected features
        """
        print("="*80)
        print("MODEL-INFORMED FEATURE SELECTION PIPELINE")
        print("="*80)
        
        print(f"Input data shape: {X.shape}")
        print(f"Categories: {y.value_counts().to_dict()}")
        
        # Phase 1: XGBoost ranking with cumulative importance analysis
        selected_features_phase1 = self.phase1_xgboost_ranking(X, y)
        
        # Phase 2: RFECV
        final_features = self.phase2_rfecv(X, y, selected_features_phase1)
        
        # Final selection
        self.final_selected_features = final_features
        
        # Save results
        self.save_results()
        
        print("="*80)
        print("FEATURE SELECTION PIPELINE COMPLETED!")
        print(f"Original features: {X.shape[1]}")
        print(f"After Phase 1 (XGBoost + Cumulative Analysis): {len(selected_features_phase1)}")
        print(f"Final selected features (After RFECV): {len(final_features)}")
        print("="*80)
        
        return final_features

def run_step2(X, y, results_dir='results/step2_feature_selection'):
    """
    Run Step 2: Model-informed feature selection with cumulative importance analysis
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data from Step 1
    y : pd.Series
        Category labels
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    selected_features : list
        Final selected features
    """
    # adjusted by analysis
    selector = ModelInformedFeatureSelector(
        xgb_top_k=164,  
        cv_folds=5,
        random_state=42,
        results_dir=results_dir
    )
    
    # Run pipeline
    selected_features = selector.run_complete_pipeline(X, y)
    
    return selected_features, selector