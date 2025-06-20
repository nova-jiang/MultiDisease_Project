import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class ThreeParallelFeatureSelection:
    """
    Comprehensive feature selection comparison using three parallel methods:
    1. SVM Coefficient Ranking
    2. RFE + Cross Validation  
    3. Permutation Importance
    """
    
    def __init__(self, 
                 feature_numbers=[20, 30, 50, 70, 100, 140],
                 cv_folds=5,
                 test_size=0.2,
                 random_state=42,
                 n_permutations=30):
        """
        Initialize the three parallel methods comparison
        
        Parameters:
        -----------
        feature_numbers : list
            List of feature subset sizes to test
        cv_folds : int
            Number of cross-validation folds
        test_size : float
            Test set proportion
        random_state : int
            Random state for reproducibility
        n_permutations : int
            Number of permutations for permutation importance
        """
        self.feature_numbers = feature_numbers
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.n_permutations = n_permutations
        
        # Data preprocessing tools
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Results storage
        self.results = {}
        self.selected_features = {}
        self.feature_rankings = {}
        self.computation_times = {}
        
    def prepare_data(self, X_filtered, y, selected_features_step1):
        """
        Prepare data for feature selection comparison
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered data from Step 1
        y : pd.Series
            Target labels
        selected_features_step1 : list
            Features from biological pre-filtering
            
        Returns:
        --------
        X_processed : np.ndarray
            Processed feature matrix
        y_encoded : np.ndarray
            Encoded labels
        """
        print("Preparing data for three parallel methods comparison...")
        
        # Select available features from Step 1
        available_features = [f for f in selected_features_step1 if f in X_filtered.columns]
        X_selected = X_filtered[available_features].copy()
        
        print(f"Using {len(available_features)} features from biological pre-filtering")
        print(f"Dataset shape: {X_selected.shape}")
        
        # Handle missing values
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
        
        # Encode labels and scale features
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Store feature names for later reference
        self.feature_names = available_features
        self.classes = self.label_encoder.classes_
        
        print(f"Data preparation completed")
        print(f"Classes: {list(self.classes)}")
        
        return X_scaled, y_encoded
    
    def method1_svm_coefficient_ranking(self, X, y):
        """
        Method 1: SVM Coefficient-based Feature Ranking
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        feature_ranking : pd.DataFrame
            Features ranked by SVM coefficient importance
        """
        print("\n🔥 Method 1: SVM Coefficient Ranking")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train linear SVM for coefficient extraction
        svm = LinearSVC(C=1.0, random_state=self.random_state, max_iter=10000)
        svm.fit(X, y)
        
        # Calculate feature importance (mean absolute coefficient across classes)
        if len(np.unique(y)) > 2:
            coef_importance = np.abs(svm.coef_).mean(axis=0)
        else:
            coef_importance = np.abs(svm.coef_[0])
        
        # Create ranking DataFrame
        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'importance_score': coef_importance,
            'method': 'SVM_Coefficient'
        }).sort_values('importance_score', ascending=False).reset_index(drop=True)
        
        feature_ranking['rank'] = range(1, len(feature_ranking) + 1)
        
        computation_time = time.time() - start_time
        self.computation_times['SVM_Coefficient'] = computation_time
        
        print(f"✅ Completed in {computation_time:.2f} seconds")
        print("Top 10 features by SVM coefficient:")
        for i, row in feature_ranking.head(10).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<35} {row['importance_score']:.6f}")
        
        return feature_ranking
    
    def method2_rfe_cross_validation(self, X, y):
        """
        Method 2: Recursive Feature Elimination with Cross Validation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        feature_ranking : pd.DataFrame
            Features ranked by RFE importance
        """
        print("\n🎯 Method 2: RFE + Cross Validation")
        print("-" * 50)
        
        start_time = time.time()
        
        # Use LinearSVC as base estimator for RFE
        base_estimator = LinearSVC(C=1.0, random_state=self.random_state, max_iter=10000)
        
        # Determine minimum features and step size for efficiency
        min_features = max(5, min(self.feature_numbers))
        max_features = min(len(self.feature_names), max(self.feature_numbers))
        step_size = max(1, (len(self.feature_names) - min_features) // 20)  # Adaptive step size
        
        print(f"  Running RFECV with {min_features}-{max_features} features, step={step_size}")
        
        # Perform RFECV
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        rfecv = RFECV(
            estimator=base_estimator,
            step=step_size,
            min_features_to_select=min_features,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1  # Use parallel processing
        )
        
        rfecv.fit(X, y)
        
        # Create ranking DataFrame
        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'rfe_ranking': rfecv.ranking_,
            'selected': rfecv.support_,
            'method': 'RFE_CV'
        }).sort_values('rfe_ranking').reset_index(drop=True)
        
        # Convert RFE ranking to importance score (lower rank = higher importance)
        max_rank = feature_ranking['rfe_ranking'].max()
        feature_ranking['importance_score'] = (max_rank - feature_ranking['rfe_ranking'] + 1) / max_rank
        feature_ranking['rank'] = range(1, len(feature_ranking) + 1)
        
        computation_time = time.time() - start_time
        self.computation_times['RFE_CV'] = computation_time
        
        print(f"✅ Completed in {computation_time:.2f} seconds")
        print(f"  Optimal number of features found: {rfecv.n_features_}")
        
        # Get the best CV score - use cv_results_ instead of grid_scores_
        if hasattr(rfecv, 'cv_results_'):
            best_score = max(rfecv.cv_results_['mean_test_score'])
            print(f"  Best CV score: {best_score:.4f}")
        else:
            # Fallback: manually calculate score for optimal features
            optimal_features = np.where(rfecv.support_)[0]
            X_optimal = X[:, optimal_features]
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(base_estimator, X_optimal, y, cv=cv, scoring='accuracy')
            print(f"  Best CV score: {scores.mean():.4f}")
        
        print("Top 10 features by RFE ranking:")
        for i, row in feature_ranking.head(10).iterrows():
            status = "✓" if row['selected'] else "✗"
            print(f"  {row['rank']:2d}. {row['feature']:<35} Rank: {row['rfe_ranking']:2d} {status}")
        
        return feature_ranking
    
    def method3_permutation_importance(self, X, y):
        """
        Method 3: Permutation Importance
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        feature_ranking : pd.DataFrame
            Features ranked by permutation importance
        """
        print("\n🧪 Method 3: Permutation Importance")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train a final model for permutation importance
        # Use the same model type for fair comparison
        final_model = SVC(kernel='rbf', C=1.0, random_state=self.random_state)
        
        # Split data to get realistic importance scores
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        final_model.fit(X_train, y_train)
        
        print(f"  Computing permutation importance with {self.n_permutations} permutations...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            final_model, X_test, y_test,
            n_repeats=self.n_permutations,
            random_state=self.random_state,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Create ranking DataFrame
        feature_ranking = pd.DataFrame({
            'feature': self.feature_names,
            'importance_score': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'method': 'Permutation'
        }).sort_values('importance_score', ascending=False).reset_index(drop=True)
        
        feature_ranking['rank'] = range(1, len(feature_ranking) + 1)
        
        computation_time = time.time() - start_time
        self.computation_times['Permutation'] = computation_time
        
        print(f"✅ Completed in {computation_time:.2f} seconds")
        print("Top 10 features by permutation importance:")
        for i, row in feature_ranking.head(10).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<35} "
                  f"{row['importance_score']:.6f} ± {row['importance_std']:.6f}")
        
        return feature_ranking
    
    def evaluate_feature_subsets(self, X, y, rankings):
        """
        Evaluate different feature subset sizes for each method
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        rankings : dict
            Feature rankings from all three methods
            
        Returns:
        --------
        evaluation_results : dict
            Performance results for each method and feature number
        """
        print("\n📊 Evaluating Feature Subset Performance")
        print("=" * 60)
        
        evaluation_results = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for method_name, ranking_df in rankings.items():
            print(f"\n🔍 Evaluating {method_name}...")
            method_results = {}
            
            for n_features in self.feature_numbers:
                if n_features > len(ranking_df):
                    print(f"  Skipping {n_features} features (exceeds available features)")
                    continue
                    
                print(f"  Testing with {n_features} features...")
                
                # Select top n features
                top_features = ranking_df.head(n_features)['feature'].tolist()
                feature_indices = [self.feature_names.index(f) for f in top_features 
                                 if f in self.feature_names]
                X_subset = X[:, feature_indices]
                
                # Cross-validation evaluation
                model = SVC(kernel='rbf', C=1.0, random_state=self.random_state)
                cv_scores = cross_val_score(model, X_subset, y, cv=cv, scoring='accuracy')
                
                # Train-test evaluation for additional metrics
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset, y, test_size=self.test_size, 
                    random_state=self.random_state, stratify=y
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='macro')
                
                method_results[n_features] = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'selected_features': top_features
                }
                
                print(f"    CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"    Test Accuracy: {test_accuracy:.4f}")
            
            evaluation_results[method_name] = method_results
        
        return evaluation_results
    
    def analyze_feature_overlap(self, rankings):
        """
        Analyze overlap between different methods' feature selections
        
        Parameters:
        -----------
        rankings : dict
            Feature rankings from all three methods
            
        Returns:
        --------
        overlap_analysis : dict
            Analysis of feature overlap between methods
        """
        print("\n🔗 Analyzing Feature Overlap Between Methods")
        print("-" * 50)
        
        overlap_analysis = {}
        methods = list(rankings.keys())
        
        # Analyze overlap at different feature numbers
        for n_features in [20, 30, 50]:
            if n_features <= min(len(rankings[m]) for m in methods):
                print(f"\nTop {n_features} features overlap:")
                
                feature_sets = {}
                for method in methods:
                    top_features = set(rankings[method].head(n_features)['feature'])
                    feature_sets[method] = top_features
                
                # Calculate pairwise overlaps
                pairwise_overlaps = {}
                for i, method1 in enumerate(methods):
                    for method2 in methods[i+1:]:
                        overlap = len(feature_sets[method1] & feature_sets[method2])
                        overlap_pct = overlap / n_features * 100
                        pairwise_overlaps[f"{method1}_vs_{method2}"] = {
                            'overlap_count': overlap,
                            'overlap_percentage': overlap_pct
                        }
                        print(f"  {method1} vs {method2}: {overlap}/{n_features} ({overlap_pct:.1f}%)")
                
                # Find consensus features (selected by all methods)
                consensus = feature_sets[methods[0]]
                for method in methods[1:]:
                    consensus = consensus & feature_sets[method]
                
                print(f"  Consensus features (all methods): {len(consensus)}")
                if len(consensus) > 0:
                    print(f"    {list(consensus)[:5]}{'...' if len(consensus) > 5 else ''}")
                
                overlap_analysis[n_features] = {
                    'pairwise_overlaps': pairwise_overlaps,
                    'consensus_features': list(consensus),
                    'consensus_count': len(consensus)
                }
        
        return overlap_analysis
    
    def visualize_results(self, evaluation_results, rankings, overlap_analysis):
        """
        Create comprehensive visualizations of the comparison results
        
        Parameters:
        -----------
        evaluation_results : dict
            Performance evaluation results
        rankings : dict
            Feature rankings from all methods
        overlap_analysis : dict
            Feature overlap analysis
        """
        print("\n📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Three Parallel Feature Selection Methods Comparison', fontsize=16)
        
        # 1. Performance comparison across feature numbers
        ax1 = axes[0, 0]
        for method_name, results in evaluation_results.items():
            feature_nums = list(results.keys())
            cv_means = [results[n]['cv_accuracy_mean'] for n in feature_nums]
            cv_stds = [results[n]['cv_accuracy_std'] for n in feature_nums]
            
            ax1.errorbar(feature_nums, cv_means, yerr=cv_stds, marker='o', 
                        label=method_name, capsize=5, linewidth=2)
        
        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('CV Accuracy')
        ax1.set_title('Performance vs Feature Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Test accuracy comparison
        ax2 = axes[0, 1]
        methods = list(evaluation_results.keys())
        colors = ['blue', 'red', 'green']
        
        for i, method in enumerate(methods):
            results = evaluation_results[method]
            feature_nums = list(results.keys())
            test_accs = [results[n]['test_accuracy'] for n in feature_nums]
            
            ax2.plot(feature_nums, test_accs, marker='s', color=colors[i], 
                    label=method, linewidth=2)
        
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Test Set Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Computation time comparison
        ax3 = axes[0, 2]
        comp_times = [self.computation_times.get(m.replace('_', ' '), 0) for m in methods]
        bars = ax3.bar(methods, comp_times, color=['lightblue', 'lightcoral', 'lightgreen'])
        ax3.set_ylabel('Computation Time (seconds)')
        ax3.set_title('Computation Time Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, comp_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 4. Feature overlap heatmap
        ax4 = axes[1, 0]
        if 30 in overlap_analysis:
            overlap_data = overlap_analysis[30]['pairwise_overlaps']
            overlap_matrix = np.zeros((len(methods), len(methods)))
            
            # Fill diagonal with 100% (self-overlap)
            np.fill_diagonal(overlap_matrix, 100)
            
            # Fill off-diagonal with pairwise overlaps
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        key = f"{method1}_vs_{method2}"
                        if key in overlap_data:
                            overlap_pct = overlap_data[key]['overlap_percentage']
                            overlap_matrix[i, j] = overlap_pct
                            overlap_matrix[j, i] = overlap_pct
            
            sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=methods, yticklabels=methods, ax=ax4)
            ax4.set_title('Feature Overlap (Top 30 Features)')
        
        # 5. Top features comparison
        ax5 = axes[1, 1]
        # Show top 10 features from each method
        top_n = 10
        all_top_features = set()
        for ranking in rankings.values():
            all_top_features.update(ranking.head(top_n)['feature'])
        
        # Create a matrix showing which method selected which feature
        feature_matrix = []
        feature_labels = []
        
        for feature in list(all_top_features)[:15]:  # Limit to 15 for readability
            row = []
            for method in methods:
                rank = rankings[method][rankings[method]['feature'] == feature].index
                if len(rank) > 0 and rank[0] < top_n:
                    row.append(top_n - rank[0])  # Higher value = higher importance
                else:
                    row.append(0)
            feature_matrix.append(row)
            feature_labels.append(feature[:25] + '...' if len(feature) > 25 else feature)
        
        if feature_matrix:
            sns.heatmap(feature_matrix, annot=False, cmap='Reds',
                       xticklabels=methods, yticklabels=feature_labels, ax=ax5)
            ax5.set_title('Top Features by Method')
        
        # 6. Best method summary
        ax6 = axes[1, 2]
        best_performances = {}
        for method_name, results in evaluation_results.items():
            best_acc = max(results[n]['cv_accuracy_mean'] for n in results.keys())
            best_n = max(results.keys(), key=lambda n: results[n]['cv_accuracy_mean'])
            best_performances[method_name] = {
                'accuracy': best_acc,
                'n_features': best_n
            }
        
        methods_short = [m.replace('_', '\n') for m in methods]
        accuracies = [best_performances[m]['accuracy'] for m in methods]
        feature_counts = [best_performances[m]['n_features'] for m in methods]
        
        bars = ax6.bar(methods_short, accuracies, color=['lightblue', 'lightcoral', 'lightgreen'])
        ax6.set_ylabel('Best CV Accuracy')
        ax6.set_title('Best Performance by Method')
        
        # Add labels showing optimal feature count
        for bar, acc, n_feat in zip(bars, accuracies, feature_counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}\n({n_feat} features)', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('three_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, rankings, evaluation_results, overlap_analysis):
        """
        Save all results to files for later analysis
        
        Parameters:
        -----------
        rankings : dict
            Feature rankings from all methods
        evaluation_results : dict
            Performance evaluation results
        overlap_analysis : dict
            Feature overlap analysis
        """
        print("\n💾 Saving results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save feature rankings
        for method, ranking_df in rankings.items():
            filename = f"feature_ranking_{method}_{timestamp}.csv"
            ranking_df.to_csv(filename, index=False)
            print(f"  {method} ranking saved to: {filename}")
        
        # Save evaluation results
        eval_filename = f"evaluation_results_{timestamp}.json"
        with open(eval_filename, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for method, results in evaluation_results.items():
                json_results[method] = {}
                for n_features, metrics in results.items():
                    json_results[method][str(n_features)] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    }
            json.dump(json_results, f, indent=2)
        print(f"  Evaluation results saved to: {eval_filename}")
        
        # Save overlap analysis
        overlap_filename = f"overlap_analysis_{timestamp}.json"
        with open(overlap_filename, 'w') as f:
            json.dump(overlap_analysis, f, indent=2)
        print(f"  Overlap analysis saved to: {overlap_filename}")
        
        # Save computation times
        time_filename = f"computation_times_{timestamp}.json"
        with open(time_filename, 'w') as f:
            json.dump(self.computation_times, f, indent=2)
        print(f"  Computation times saved to: {time_filename}")
        
    def run_complete_comparison(self, X_filtered, y, selected_features_step1):
        """
        Run the complete three parallel methods comparison
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Filtered data from Step 1
        y : pd.Series
            Target labels  
        selected_features_step1 : list
            Features from biological pre-filtering
            
        Returns:
        --------
        complete_results : dict
            All results from the comparison
        """
        print("="*80)
        print("THREE PARALLEL FEATURE SELECTION METHODS COMPARISON")
        print("="*80)
        
        # Prepare data
        X_processed, y_encoded = self.prepare_data(X_filtered, y, selected_features_step1)
        
        # Run three methods
        rankings = {}
        
        # Method 1: SVM Coefficient
        rankings['SVM_Coefficient'] = self.method1_svm_coefficient_ranking(X_processed, y_encoded)
        
        # Method 2: RFE + CV
        rankings['RFE_CV'] = self.method2_rfe_cross_validation(X_processed, y_encoded)
        
        # Method 3: Permutation Importance
        rankings['Permutation'] = self.method3_permutation_importance(X_processed, y_encoded)
        
        # Store rankings
        self.feature_rankings = rankings
        
        # Evaluate feature subsets
        evaluation_results = self.evaluate_feature_subsets(X_processed, y_encoded, rankings)
        
        # Analyze feature overlap
        overlap_analysis = self.analyze_feature_overlap(rankings)
        
        # Visualize results
        self.visualize_results(evaluation_results, rankings, overlap_analysis)
        
        # Save results
        self.save_results(rankings, evaluation_results, overlap_analysis)
        
        # Generate summary
        self.generate_summary(rankings, evaluation_results, overlap_analysis)
        
        print("\n🎉 THREE PARALLEL METHODS COMPARISON COMPLETED!")
        print("="*80)
        
        return {
            'rankings': rankings,
            'evaluation_results': evaluation_results,
            'overlap_analysis': overlap_analysis,
            'computation_times': self.computation_times
        }
    
    def generate_summary(self, rankings, evaluation_results, overlap_analysis):
        """
        Generate a comprehensive summary of the comparison
        """
        print("\n📋 COMPARISON SUMMARY")
        print("="*60)
        
        # Find best performing method
        best_method = None
        best_accuracy = 0
        best_n_features = 0
        
        for method_name, results in evaluation_results.items():
            for n_features, metrics in results.items():
                if metrics['cv_accuracy_mean'] > best_accuracy:
                    best_accuracy = metrics['cv_accuracy_mean']
                    best_method = method_name
                    best_n_features = n_features
        
        print(f"🏆 BEST PERFORMING METHOD: {best_method}")
        print(f"   Best CV Accuracy: {best_accuracy:.4f}")
        print(f"   Optimal Features: {best_n_features}")
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        for method_name, results in evaluation_results.items():
            max_acc = max(results[n]['cv_accuracy_mean'] for n in results.keys())
            optimal_n = max(results.keys(), key=lambda n: results[n]['cv_accuracy_mean'])
            comp_time = self.computation_times.get(method_name, 0)
            
            print(f"   {method_name}:")
            print(f"     Best CV Accuracy: {max_acc:.4f}")
            print(f"     Optimal Features: {optimal_n}")
            print(f"     Computation Time: {comp_time:.2f}s")
        
        # Feature overlap insights
        if 30 in overlap_analysis:
            consensus_count = overlap_analysis[30]['consensus_count']
            print(f"\n🔗 FEATURE OVERLAP (Top 30):")
            print(f"   Consensus features: {consensus_count}")
            
            pairwise = overlap_analysis[30]['pairwise_overlaps']
            for pair, data in pairwise.items():
                print(f"   {pair}: {data['overlap_percentage']:.1f}%")

