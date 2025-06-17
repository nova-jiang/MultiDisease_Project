import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BiologicalPrefiltering:
    """
    Biological statistical pre-filtering for microbiome data
    Includes: data preprocessing, sparse filtering, Kruskal-Wallis test, and effect size calculation
    """
    
    def __init__(self, min_prevalence=0.1, fdr_threshold=0.05, effect_size_threshold=0.01):
        """
        Initialize the biological pre-filtering pipeline
        
        Parameters:
        -----------
        min_prevalence : float, default=0.1
            Minimum prevalence threshold for sparse filtering (remove features present in <10% samples)
        fdr_threshold : float, default=0.05
            False discovery rate threshold for multiple testing correction
        effect_size_threshold : float, default=0.01
            Minimum effect size (eta-squared) threshold for biological significance
        """
        self.min_prevalence = min_prevalence
        self.fdr_threshold = fdr_threshold
        self.effect_size_threshold = effect_size_threshold
        self.selected_features = None
        self.statistical_results = None
        
    def load_and_prepare_data(self, filepath):
        """
        Load the gmrepo_cleaned_dataset and prepare for analysis
        
        Parameters:
        -----------
        filepath : str
            Path to the gmrepo_cleaned_dataset.csv file
            
        Returns:
        --------
        X : pd.DataFrame
            Microbiome abundance data (samples x features)
        y : pd.Series
            Category labels
        metadata : pd.DataFrame
            Sample metadata
        """
        print("Loading and preparing data...")
        
        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Extract metadata columns (first 20 columns in the final data file)
        metadata_cols = df.columns[:20].tolist()
        metadata = df[metadata_cols].copy()
        
        # Extract microbiome abundance data (from column 21 onwards)
        microbe_cols = df.columns[20:].tolist()
        X = df[microbe_cols].copy()
        
        # Extract category labels
        y = df['category'].copy()
        
        print(f"Metadata columns: {len(metadata_cols)}")
        print(f"Microbiome features: {len(microbe_cols)}")
        print(f"Category distribution:\n{y.value_counts()}")
        
        return X, y, metadata
    
    def data_preprocessing(self, X):
        """
        Preprocess microbiome count data
        1. Convert to relative abundance
        2. Add pseudocount
        3. CLR transformation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Raw microbiome count data
            
        Returns:
        --------
        X_processed : pd.DataFrame
            Preprocessed microbiome data
        """
        print("Step 1a: Data preprocessing...")
        
        # Check if data looks like counts
        print(f"Data range: {X.min().min():.6f} to {X.max().max():.6f}")
        row_sums = X.sum(axis=1)
        print(f"Row sums range: {row_sums.min():.1f} to {row_sums.max():.1f}")
        
        # Convert to relative abundance
        X_rel = X.div(X.sum(axis=1), axis=0)
        print("Converted to relative abundance")
        
        # Add pseudocount to avoid log(0) in CLR transformation
        pseudocount = 1e-6
        X_pseudo = X_rel + pseudocount
        
        # CLR (Centered Log-Ratio) transformation
        # CLR(x) = log(x) - mean(log(x))
        X_log = np.log(X_pseudo)
        X_clr = X_log.sub(X_log.mean(axis=1), axis=0)
        
        print("Applied CLR transformation")
        print(f"CLR data range: {X_clr.min().min():.6f} to {X_clr.max().max():.6f}")
        
        return X_clr
    
    def sparse_filtering(self, X_original, X_processed):
        """
        Remove features with low prevalence across samples
        
        Parameters:
        -----------
        X_original : pd.DataFrame
            Original microbiome count data (before preprocessing)
        X_processed : pd.DataFrame
            Preprocessed microbiome data (after CLR transformation)
            
        Returns:
        --------
        X_filtered : pd.DataFrame
            Data after sparse filtering
        """
        print("Step 1b: Sparse filtering...")
        
        # Calculate prevalence using original count data
        # Prevalence = proportion of samples where feature has non-zero counts
        prevalence = (X_original > 0).mean(axis=0)
        
        # Show detailed prevalence statistics
        print(f"Prevalence distribution:")
        print(f"  Mean: {prevalence.mean():.3f}")
        print(f"  Median: {prevalence.median():.3f}")
        print(f"  Min: {prevalence.min():.3f}")
        print(f"  Max: {prevalence.max():.3f}")
        print(f"  25th percentile: {prevalence.quantile(0.25):.3f}")
        print(f"  75th percentile: {prevalence.quantile(0.75):.3f}")
        
        # Filter features based on minimum prevalence
        features_to_keep = prevalence >= self.min_prevalence
        
        print(f"Features before filtering: {len(X_processed.columns)}")
        print(f"Features after prevalence filtering (>={self.min_prevalence*100:.1f}%): {features_to_keep.sum()}")
        print(f"Removed {len(X_processed.columns) - features_to_keep.sum()} features")
        
        # Apply filtering to processed data
        X_filtered = X_processed.loc[:, features_to_keep]
        
        # Show some examples of removed vs kept features
        removed_features = prevalence[~features_to_keep].sort_values()
        kept_features = prevalence[features_to_keep].sort_values()
        
        if len(removed_features) > 0:
            print(f"\nExamples of removed features (lowest prevalence):")
            for i, (feature, prev) in enumerate(removed_features.head(5).items()):
                print(f"  {feature}: {prev:.3f}")
                
        print(f"\nExamples of kept features (lowest prevalence among kept):")
        for i, (feature, prev) in enumerate(kept_features.head(5).items()):
            print(f"  {feature}: {prev:.3f}")
        
        return X_filtered
    
    def kruskal_wallis_test(self, X, y):
        """
        Perform Kruskal-Wallis test for each feature across all categories
        
        Parameters:
        -----------
        X : pd.DataFrame
            Filtered microbiome data
        y : pd.Series
            Category labels
            
        Returns:
        --------
        results_df : pd.DataFrame
            Statistical test results for each feature
        """
        print("Step 1c: Kruskal-Wallis test...")
        
        # Get unique categories
        categories = y.unique()
        print(f"Testing differences across {len(categories)} categories: {categories}")
        
        # Prepare results storage
        results = {
            'feature': [],
            'kruskal_stat': [],
            'p_value': [],
            'eta_squared': []
        }
        
        # Perform Kruskal-Wallis test for each feature
        n_features = len(X.columns)
        processed_features = 0
        
        for i, feature in enumerate(X.columns):
            if (i + 1) % 200 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{n_features} features...")
                
            # Split data by category
            groups = []
            for cat in categories:
                group_data = X.loc[y == cat, feature].values
                # Remove any NaN or infinite values
                group_data = group_data[np.isfinite(group_data)]
                if len(group_data) > 0:
                    groups.append(group_data)
            
            # Skip if we don't have enough groups or data
            if len(groups) < 2:
                continue
                
            # Check if all groups have at least some variation
            group_sizes = [len(group) for group in groups]
            if min(group_sizes) < 3:  # Need at least 3 samples per group
                continue
                
            # Perform Kruskal-Wallis test
            try:
                kruskal_stat, p_value = kruskal(*groups)
                
                # Check for valid results
                if not np.isfinite(kruskal_stat) or not np.isfinite(p_value):
                    continue
                    
                # Calculate effect size (eta-squared)
                # eta-squared = (H - k + 1) / (n - k)
                # where H is Kruskal-Wallis statistic, k is number of groups, n is total sample size
                n = sum(len(group) for group in groups)
                k = len(groups)
                
                if n > k and kruskal_stat >= k - 1:
                    eta_squared = (kruskal_stat - k + 1) / (n - k)
                    eta_squared = max(0, eta_squared)  # Ensure non-negative
                else:
                    eta_squared = 0
                
                results['feature'].append(feature)
                results['kruskal_stat'].append(kruskal_stat)
                results['p_value'].append(p_value)
                results['eta_squared'].append(eta_squared)
                processed_features += 1
                
            except ValueError as e:
                # This can happen if all values in a group are identical
                print(f"  Warning: Skipping feature {feature} due to insufficient variation")
                continue
            except Exception as e:
                print(f"  Error testing feature {feature}: {e}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        print(f"Completed Kruskal-Wallis test for {len(results_df)} features")
        print(f"Skipped {n_features - processed_features} features due to insufficient data/variation")
        
        if len(results_df) == 0:
            print("WARNING: No features passed the statistical test!")
            print("This might indicate:")
            print("  - Too stringent filtering parameters")
            print("  - Insufficient variation in the data")
            print("  - Data preprocessing issues")
            
        return results_df
    
    def multiple_testing_correction(self, results_df):
        """
        Apply Benjamini-Hochberg FDR correction
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Statistical test results
            
        Returns:
        --------
        results_df : pd.DataFrame
            Results with FDR correction
        """
        print("Step 1d: Multiple testing correction...")
        
        # Apply Benjamini-Hochberg FDR correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            results_df['p_value'], 
            alpha=self.fdr_threshold, 
            method='fdr_bh'
        )
        
        # Add corrected p-values and significance flags
        results_df['p_corrected'] = p_corrected
        results_df['significant'] = rejected
        
        print(f"Features with significant differences (FDR < {self.fdr_threshold}):")
        print(f"  Before correction: {(results_df['p_value'] < self.fdr_threshold).sum()}")
        print(f"  After FDR correction: {rejected.sum()}")
        
        return results_df
    
    def effect_size_filtering(self, results_df):
        """
        Filter features based on effect size threshold
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Statistical test results with FDR correction
            
        Returns:
        --------
        final_features : list
            List of selected feature names
        """
        print("Step 1e: Effect size filtering...")
        
        # Filter by significance and effect size
        significant_features = results_df[
            (results_df['significant']) & 
            (results_df['eta_squared'] >= self.effect_size_threshold)
        ]
        
        print(f"Features passing all filters:")
        print(f"  Significant (FDR < {self.fdr_threshold}): {results_df['significant'].sum()}")
        print(f"  Effect size >= {self.effect_size_threshold}: {(results_df['eta_squared'] >= self.effect_size_threshold).sum()}")
        print(f"  Final selected features: {len(significant_features)}")
        
        # Sort by effect size (descending)
        significant_features = significant_features.sort_values('eta_squared', ascending=False)
        
        return significant_features['feature'].tolist()
    
    def visualize_results(self, results_df, X_filtered, y):
        """
        Create visualizations for the filtering results
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Statistical test results
        X_filtered : pd.DataFrame
            Filtered microbiome data
        y : pd.Series
            Category labels
        """
        print("Creating visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Biological Pre-filtering Results', fontsize=16)
        
        # Check if we have results to plot
        if len(results_df) == 0:
            # If no results, just show category distribution
            axes[0, 0].text(0.5, 0.5, 'No statistical results\nto display', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('P-value Distribution (No Data)')
            
            axes[0, 1].text(0.5, 0.5, 'No statistical results\nto display', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Effect Size Distribution (No Data)')
            
            axes[1, 0].text(0.5, 0.5, 'No statistical results\nto display', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Volcano Plot (No Data)')
        else:
            # 1. P-value distribution
            valid_pvals = results_df['p_value'].dropna()
            if len(valid_pvals) > 0:
                axes[0, 0].hist(valid_pvals, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(self.fdr_threshold, color='red', linestyle='--', 
                                  label=f'FDR threshold ({self.fdr_threshold})')
                axes[0, 0].set_xlabel('P-value')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('P-value Distribution')
                axes[0, 0].legend()
            
            # 2. Effect size distribution
            valid_eta = results_df['eta_squared'].dropna()
            if len(valid_eta) > 0:
                axes[0, 1].hist(valid_eta, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(self.effect_size_threshold, color='red', linestyle='--', 
                                  label=f'Effect size threshold ({self.effect_size_threshold})')
                axes[0, 1].set_xlabel('Effect Size (eta-squared)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Effect Size Distribution')
                axes[0, 1].legend()
            
            # 3. Volcano plot (Effect size vs -log10(p-value))
            if 'p_corrected' in results_df.columns:
                valid_data = results_df.dropna(subset=['eta_squared', 'p_corrected'])
                if len(valid_data) > 0:
                    x = valid_data['eta_squared']
                    y = -np.log10(valid_data['p_corrected'].replace(0, 1e-300))  # Avoid log(0)
                    
                    # Color points based on significance and effect size
                    colors = ['red' if (sig and eta >= self.effect_size_threshold) else 'gray' 
                             for sig, eta in zip(valid_data['significant'], valid_data['eta_squared'])]
                    
                    axes[1, 0].scatter(x, y, c=colors, alpha=0.6)
                    axes[1, 0].axhline(-np.log10(self.fdr_threshold), color='red', linestyle='--', alpha=0.7)
                    axes[1, 0].axvline(self.effect_size_threshold, color='red', linestyle='--', alpha=0.7)
                    axes[1, 0].set_xlabel('Effect Size (eta-squared)')
                    axes[1, 0].set_ylabel('-log10(FDR corrected p-value)')
                    axes[1, 0].set_title('Volcano Plot')
        
        # 4. Category distribution (always show this)
        category_counts = y.value_counts()
        axes[1, 1].bar(category_counts.index, category_counts.values)
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Sample Distribution by Category')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, filepath):
        """
        Run the complete biological pre-filtering pipeline
        
        Parameters:
        -----------
        filepath : str
            Path to the gmrepo_cleaned_dataset.csv file
            
        Returns:
        --------
        X_final : pd.DataFrame
            Final filtered microbiome data
        y : pd.Series
            Category labels
        selected_features : list
            List of selected feature names
        results_df : pd.DataFrame
            Complete statistical results
        """
        print("="*60)
        print("BIOLOGICAL PRE-FILTERING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        X_original, y, metadata = self.load_and_prepare_data(filepath)
        
        # Data preprocessing
        X_processed = self.data_preprocessing(X_original)
        
        # Sparse filtering (using original data for prevalence calculation)
        X_filtered = self.sparse_filtering(X_original, X_processed)
        
        # Statistical testing
        results_df = self.kruskal_wallis_test(X_filtered, y)
        
        # Only proceed with correction and filtering if we have results
        if len(results_df) > 0:
            # Multiple testing correction
            results_df = self.multiple_testing_correction(results_df)
            
            # Effect size filtering
            selected_features = self.effect_size_filtering(results_df)
            
            # Create final filtered dataset
            if len(selected_features) > 0:
                X_final = X_filtered[selected_features]
            else:
                print("WARNING: No features passed all filters!")
                print("Returning top 50 features by effect size as backup...")
                top_features = results_df.nlargest(50, 'eta_squared')['feature'].tolist()
                X_final = X_filtered[top_features]
                selected_features = top_features
        else:
            print("WARNING: No statistical results generated!")
            print("Returning original filtered data...")
            X_final = X_filtered
            selected_features = X_filtered.columns.tolist()
            
        # Store results
        self.selected_features = selected_features
        self.statistical_results = results_df
        
        # Visualize results
        self.visualize_results(results_df, X_filtered, y)
        
        print("="*60)
        print("PIPELINE COMPLETED!")
        print(f"Final dataset shape: {X_final.shape}")
        print(f"Selected features: {len(selected_features)}")
        print("="*60)
        
        return X_final, y, selected_features, results_df

# # To print out what we have in this step
# if __name__ == "__main__":
#     # Initialize the biological pre-filtering pipeline
#     bio_filter = BiologicalPrefiltering(
#         min_prevalence=0.1,      # Keep features present in at least 10% of samples
#         fdr_threshold=0.05,      # FDR < 0.05
#         effect_size_threshold=0.01  # Minimum effect size
#     )
    
#     # Run the complete pipeline
#     X_final, y, selected_features, results_df = bio_filter.run_complete_pipeline(
#         'gmrepo_cleaned_dataset.csv'
#     )
    
#     # Save results
#     X_final.to_csv('filtered_features.csv', index=True)
#     results_df.to_csv('statistical_results.csv', index=False)
    
#     print(f"\nTop 10 most important features by effect size:")
#     print(results_df[results_df['significant']].nlargest(10, 'eta_squared')[['feature', 'eta_squared', 'p_corrected']])

def run_step1(filepath):
    """Run biological pre-filtering and return results"""
    bio_filter = BiologicalPrefiltering(
        min_prevalence=0.1,
        fdr_threshold=0.05,
        effect_size_threshold=0.01
    )
    
    return bio_filter.run_complete_pipeline(filepath)