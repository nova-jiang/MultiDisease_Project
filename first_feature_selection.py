import pandas as pd
import numpy as np
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BiologicalPrefiltering:
    """
    Biological statistical pre-filtering for microbiome data
    Follows proper microbiome data processing pipeline:
    1. Handle missing values and data quality
    2. Sparse filtering (prevalence)
    3. Relative abundance conversion
    4. Pseudocount addition
    5. CLR transformation
    6. Statistical testing
    7. Multiple testing correction
    8. Effect size filtering
    """
    
    def __init__(self, min_prevalence=0.03,  fdr_threshold=0.1, effect_size_threshold=0.003):
        """
        Initialize the biological pre-filtering pipeline
        
        Parameters:
        -----------
        min_prevalence : float, default=0.03
            Minimum prevalence threshold for sparse filtering (remove features present in <5% samples)
        fdr_threshold : float, default=0.1
            False discovery rate threshold for multiple testing correction
        effect_size_threshold : float, default=0.003
            Minimum effect size (eta-squared) threshold for biological significance
        """
        self.min_prevalence = min_prevalence
        self.fdr_threshold = fdr_threshold
        self.effect_size_threshold = effect_size_threshold
        self.selected_features = None
        self.statistical_results = None
        self.processing_log = {}  # Track filtering at each step
        
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
        
        # Initialize processing log
        self.processing_log['original_features'] = len(microbe_cols)
        self.processing_log['original_samples'] = len(X)
        
        return X, y, metadata
    
    def step1_handle_missing_values(self, X):
        """
        Step 1: Handle missing values and data quality issues
        
        Parameters:
        -----------
        X : pd.DataFrame
            Raw microbiome count data
            
        Returns:
        --------
        X_clean : pd.DataFrame
            Data with missing values handled
        """
        print("\n" + "="*50)
        print("STEP 1: HANDLE MISSING VALUES AND DATA QUALITY")
        print("="*50)
        
        X_clean = X.copy()
        
        # Check data types
        print(f"Data types: {X_clean.dtypes.value_counts().to_dict()}")
        
        # Check for missing values
        missing_counts = X_clean.isnull().sum()
        features_with_missing = (missing_counts > 0).sum()
        total_missing = missing_counts.sum()
        
        print(f"Features with missing values: {features_with_missing}/{len(X_clean.columns)}")
        print(f"Total missing values: {total_missing}")
        
        if total_missing > 0:
            print("Handling missing values...")
            # For microbiome data, missing usually means absent (0)
            X_clean = X_clean.fillna(0)
            print("Filled missing values with 0 (assumes missing = absent)")
        
        # Convert to numeric (handle any string/object columns)
        non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting {len(non_numeric_cols)} non-numeric columns to numeric...")
            for col in non_numeric_cols:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            # Fill any new NaNs created by conversion
            X_clean = X_clean.fillna(0)
        
        # Check for negative values (shouldn't exist in count data)
        negative_counts = (X_clean < 0).sum().sum()
        if negative_counts > 0:
            print(f"WARNING: Found {negative_counts} negative values. Setting to 0.")
            X_clean = X_clean.clip(lower=0)
        
        # Check data range
        print(f"Data range after cleaning: {X_clean.min().min():.6f} to {X_clean.max().max():.6f}")
        
        # Check row sums (total abundance per sample)
        row_sums = X_clean.sum(axis=1)
        print(f"Sample abundance sums - Min: {row_sums.min():.1f}, Max: {row_sums.max():.1f}, Mean: {row_sums.mean():.1f}")
        
        # Identify samples with very low total abundance
        low_abundance_samples = (row_sums < row_sums.quantile(0.01)).sum()
        if low_abundance_samples > 0:
            print(f"WARNING: {low_abundance_samples} samples have very low total abundance (bottom 1%)")
        
        self.processing_log['after_missing_handling'] = {
            'features': len(X_clean.columns),
            'samples': len(X_clean),
            'missing_values_filled': total_missing,
            'negative_values_fixed': negative_counts
        }
        
        return X_clean
    
    def step2_sparse_filtering(self, X_clean):
        """
        Step 2: Remove features with low prevalence (sparse filtering)
        
        Parameters:
        -----------
        X_clean : pd.DataFrame
            Clean microbiome count data
            
        Returns:
        --------
        X_filtered : pd.DataFrame
            Data after sparse filtering
        """
        print("\n" + "="*50)
        print("STEP 2: SPARSE FILTERING (PREVALENCE-BASED)")
        print("="*50)
        
        # Calculate prevalence (proportion of samples where feature > 0)
        prevalence = (X_clean > 0).mean(axis=0)
        
        # Show prevalence distribution
        print(f"Prevalence statistics:")
        print(f"  Mean: {prevalence.mean():.3f}")
        print(f"  Median: {prevalence.median():.3f}")
        print(f"  Min: {prevalence.min():.3f}")
        print(f"  Max: {prevalence.max():.3f}")
        print(f"  25th percentile: {prevalence.quantile(0.25):.3f}")
        print(f"  75th percentile: {prevalence.quantile(0.75):.3f}")
        
        # Apply prevalence filter
        features_to_keep = prevalence >= self.min_prevalence
        X_filtered = X_clean.loc[:, features_to_keep]
        
        print(f"\nFiltering results:")
        print(f"  Features before filtering: {len(X_clean.columns)}")
        print(f"  Features after prevalence filtering (>={self.min_prevalence*100:.1f}%): {features_to_keep.sum()}")
        print(f"  Features removed: {len(X_clean.columns) - features_to_keep.sum()}")
        print(f"  Percentage kept: {(features_to_keep.sum() / len(X_clean.columns)) * 100:.1f}%")
        
        # Show examples of removed and kept features
        removed_features = prevalence[~features_to_keep].sort_values()
        kept_features = prevalence[features_to_keep].sort_values()
        
        if len(removed_features) > 0:
            print(f"\nExamples of removed features (lowest prevalence):")
            for i, (feature, prev) in enumerate(removed_features.head(5).items()):
                print(f"  {feature}: {prev:.3f}")
                
        print(f"\nExamples of kept features (lowest prevalence among kept):")
        for i, (feature, prev) in enumerate(kept_features.head(5).items()):
            print(f"  {feature}: {prev:.3f}")
        
        self.processing_log['after_sparse_filtering'] = {
            'features': len(X_filtered.columns),
            'samples': len(X_filtered),
            'features_removed': len(X_clean.columns) - features_to_keep.sum(),
            'min_prevalence_used': self.min_prevalence
        }
        
        return X_filtered
    
    def step3_relative_abundance_and_clr(self, X_filtered):
        """
        Step 3: Convert to relative abundance, add pseudocount, and apply CLR transformation
        
        Parameters:
        -----------
        X_filtered : pd.DataFrame
            Data after sparse filtering
            
        Returns:
        --------
        X_clr : pd.DataFrame
            CLR-transformed data
        """
        print("\n" + "="*50)
        print("STEP 3: RELATIVE ABUNDANCE + CLR TRANSFORMATION")
        print("="*50)
        
        # Check if data looks like counts
        print(f"Data range before transformation: {X_filtered.min().min():.6f} to {X_filtered.max().max():.6f}")
        row_sums = X_filtered.sum(axis=1)
        print(f"Row sums range: {row_sums.min():.1f} to {row_sums.max():.1f}")
        
        # Step 3a: Convert to relative abundance
        print("Converting to relative abundance...")
        X_rel = X_filtered.div(X_filtered.sum(axis=1), axis=0)
        
        # Check for samples with zero total abundance
        zero_sum_samples = (X_filtered.sum(axis=1) == 0).sum()
        if zero_sum_samples > 0:
            print(f"WARNING: {zero_sum_samples} samples have zero total abundance!")
            # Remove these samples
            valid_samples = X_filtered.sum(axis=1) > 0
            X_rel = X_rel.loc[valid_samples]
            print(f"Removed {zero_sum_samples} samples with zero abundance")
        
        print(f"Relative abundance range: {X_rel.min().min():.6f} to {X_rel.max().max():.6f}")
        
        # Step 3b: Add pseudocount for CLR transformation
        pseudocount = 1e-6
        print(f"Adding pseudocount: {pseudocount}")
        X_pseudo = X_rel + pseudocount
        
        # Step 3c: CLR (Centered Log-Ratio) transformation
        print("Applying CLR transformation...")
        X_log = np.log(X_pseudo)
        X_clr = X_log.sub(X_log.mean(axis=1), axis=0)
        
        print(f"CLR data range: {X_clr.min().min():.6f} to {X_clr.max().max():.6f}")
        
        # Check for any problematic values
        nan_count = X_clr.isnull().sum().sum()
        inf_count = np.isinf(X_clr.values).sum()
        
        if nan_count > 0:
            print(f"WARNING: {nan_count} NaN values after CLR transformation!")
            X_clr = X_clr.fillna(0)  # Replace NaN with 0
            
        if inf_count > 0:
            print(f"WARNING: {inf_count} infinite values after CLR transformation!")
            X_clr = X_clr.replace([np.inf, -np.inf], 0)  # Replace inf with 0
        
        print("CLR transformation completed successfully")
        
        self.processing_log['after_clr_transformation'] = {
            'features': len(X_clr.columns),
            'samples': len(X_clr),
            'zero_sum_samples_removed': zero_sum_samples,
            'nan_values_fixed': nan_count,
            'inf_values_fixed': inf_count
        }
        
        return X_clr
    
    def step4_statistical_testing(self, X_clr, y):
        """
        Step 4: Perform Kruskal-Wallis test for each feature
        
        Parameters:
        -----------
        X_clr : pd.DataFrame
            CLR-transformed data
        y : pd.Series
            Category labels (aligned with X_clr after any sample removal)
            
        Returns:
        --------
        results_df : pd.DataFrame
            Statistical test results
        """
        print("\n" + "="*50)
        print("STEP 4: STATISTICAL TESTING (KRUSKAL-WALLIS)")
        print("="*50)
        
        # Ensure y is aligned with X_clr (in case samples were removed)
        y_aligned = y.loc[X_clr.index]
        
        # Get unique categories
        categories = y_aligned.unique()
        print(f"Testing differences across {len(categories)} categories: {categories}")
        print(f"Sample sizes per category:")
        for cat in categories:
            print(f"  {cat}: {(y_aligned == cat).sum()}")
        
        # Prepare results storage
        results = {
            'feature': [],
            'kruskal_stat': [],
            'p_value': [],
            'eta_squared': []
        }
        
        # Perform Kruskal-Wallis test for each feature
        n_features = len(X_clr.columns)
        processed_features = 0
        skipped_features = 0
        
        print(f"Testing {n_features} features...")
        
        for i, feature in enumerate(X_clr.columns):
            if (i + 1) % 100 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{n_features} features...")
                
            # Split data by category
            groups = []
            for cat in categories:
                group_data = X_clr.loc[y_aligned == cat, feature].values
                # Remove any NaN or infinite values
                group_data = group_data[np.isfinite(group_data)]
                if len(group_data) > 0:
                    groups.append(group_data)
            
            # Skip if we don't have enough groups or data
            if len(groups) < 2:
                skipped_features += 1
                continue
                
            # Check if all groups have at least some variation
            group_sizes = [len(group) for group in groups]
            if min(group_sizes) < 3:  # Need at least 3 samples per group
                skipped_features += 1
                continue
                
            # Perform Kruskal-Wallis test
            try:
                kruskal_stat, p_value = kruskal(*groups)
                
                # Check for valid results
                if not np.isfinite(kruskal_stat) or not np.isfinite(p_value):
                    skipped_features += 1
                    continue
                    
                # Calculate effect size (eta-squared)
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
                skipped_features += 1
                continue
            except Exception as e:
                print(f"  Error testing feature {feature}: {e}")
                skipped_features += 1
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        print(f"\nStatistical testing completed:")
        print(f"  Features tested successfully: {len(results_df)}")
        print(f"  Features skipped: {skipped_features}")
        print(f"  Success rate: {(len(results_df) / n_features) * 100:.1f}%")
        
        if len(results_df) == 0:
            print("ERROR: No features passed the statistical test!")
            
        self.processing_log['after_statistical_testing'] = {
            'features_tested': len(results_df),
            'features_skipped': skipped_features,
            'input_features': n_features
        }
        
        return results_df
    
    def step5_multiple_testing_correction(self, results_df):
        """
        Step 5: Apply Benjamini-Hochberg FDR correction
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Statistical test results
            
        Returns:
        --------
        results_df : pd.DataFrame
            Results with FDR correction
        """
        print("\n" + "="*50)
        print("STEP 5: MULTIPLE TESTING CORRECTION")
        print("="*50)
        
        if len(results_df) == 0:
            print("No results to correct!")
            return results_df
        
        # Apply Benjamini-Hochberg FDR correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            results_df['p_value'], 
            alpha=self.fdr_threshold, 
            method='fdr_bh'
        )
        
        # Add corrected p-values and significance flags
        results_df['p_corrected'] = p_corrected
        results_df['significant'] = rejected
        
        print(f"Multiple testing correction results:")
        print(f"  Features before correction: {len(results_df)}")
        print(f"  Significant (raw p < {self.fdr_threshold}): {(results_df['p_value'] < self.fdr_threshold).sum()}")
        print(f"  Significant (FDR < {self.fdr_threshold}): {rejected.sum()}")
        print(f"  Correction rate: {(rejected.sum() / len(results_df)) * 100:.1f}%")
        
        self.processing_log['after_fdr_correction'] = {
            'significant_features': rejected.sum(),
            'total_tested': len(results_df),
            'fdr_threshold': self.fdr_threshold
        }
        
        return results_df
    
    def step6_effect_size_filtering(self, results_df):
        """
        Step 6: Filter features based on effect size threshold
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Statistical test results with FDR correction
            
        Returns:
        --------
        final_features : list
            List of selected feature names
        """
        print("\n" + "="*50)
        print("STEP 6: EFFECT SIZE FILTERING")
        print("="*50)
        
        if len(results_df) == 0:
            print("No results to filter!")
            return []
        
        # Show effect size distribution
        print(f"Effect size (eta-squared) distribution:")
        print(f"  Mean: {results_df['eta_squared'].mean():.4f}")
        print(f"  Median: {results_df['eta_squared'].median():.4f}")
        print(f"  Min: {results_df['eta_squared'].min():.4f}")
        print(f"  Max: {results_df['eta_squared'].max():.4f}")
        print(f"  95th percentile: {results_df['eta_squared'].quantile(0.95):.4f}")
        
        # Filter by significance and effect size
        significant_features = results_df[
            (results_df['significant']) & 
            (results_df['eta_squared'] >= self.effect_size_threshold)
        ]
        
        print(f"\nFinal filtering results:")
        print(f"  Significant (FDR < {self.fdr_threshold}): {results_df['significant'].sum()}")
        print(f"  Effect size >= {self.effect_size_threshold}: {(results_df['eta_squared'] >= self.effect_size_threshold).sum()}")
        print(f"  Both criteria: {len(significant_features)}")
        print(f"  Final selection rate: {(len(significant_features) / len(results_df)) * 100:.1f}%")
        
        if len(significant_features) == 0:
            print("WARNING: No features passed all filters!")
            print("Consider relaxing the thresholds:")
            print(f"  - Current effect size threshold: {self.effect_size_threshold}")
            print(f"  - Current FDR threshold: {self.fdr_threshold}")
            
            # Show what would happen with relaxed thresholds
            relaxed_effect = results_df['eta_squared'].quantile(0.8)  # Top 20%
            relaxed_features = results_df[
                (results_df['significant']) & 
                (results_df['eta_squared'] >= relaxed_effect)
            ]
            print(f"  - With effect size >= {relaxed_effect:.4f} (80th percentile): {len(relaxed_features)} features")
            
            # Return top features by effect size as backup
            backup_features = results_df.nlargest(min(50, len(results_df)), 'eta_squared')['feature'].tolist()
            print(f"Returning top {len(backup_features)} features by effect size as backup")
            
            self.processing_log['final_selection'] = {
                'method': 'backup_top_effect_size',
                'features_selected': len(backup_features),
                'effect_size_threshold_used': self.effect_size_threshold
            }
            
            return backup_features
        
        # Sort by effect size (descending)
        significant_features = significant_features.sort_values('eta_squared', ascending=False)
        final_features = significant_features['feature'].tolist()
        
        print(f"\nTop 10 selected features by effect size:")
        for i, (_, row) in enumerate(significant_features.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: eta² = {row['eta_squared']:.4f}, p = {row['p_corrected']:.2e}")
        
        self.processing_log['final_selection'] = {
            'method': 'full_criteria',
            'features_selected': len(final_features),
            'effect_size_threshold_used': self.effect_size_threshold,
            'fdr_threshold_used': self.fdr_threshold
        }
        
        return final_features
    
    def print_processing_summary(self):
        """Print a summary of all processing steps"""
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        
        for step, info in self.processing_log.items():
            if isinstance(info, dict):
                if 'features' in info:
                    print(f"{step.replace('_', ' ').title()}: {info['features']} features")
                elif 'features_selected' in info:
                    print(f"{step.replace('_', ' ').title()}: {info['features_selected']} features")
                elif 'features_tested' in info:
                    print(f"{step.replace('_', ' ').title()}: {info['features_tested']} features tested")
                elif 'significant_features' in info:
                    print(f"{step.replace('_', ' ').title()}: {info['significant_features']} significant features")
            else:
                print(f"{step.replace('_', ' ').title()}: {info}")
        
        # Calculate overall reduction
        original = self.processing_log.get('original_features', 0)
        final = self.processing_log.get('final_selection', {}).get('features_selected', 0)
        if original > 0:
            reduction = ((original - final) / original) * 100
            print(f"\nOverall feature reduction: {original} → {final} ({reduction:.1f}% reduction)")
    
    def visualize_results(self, results_df, X_filtered, y):
        """Create visualizations for the filtering results"""
        print("\nCreating visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Biological Pre-filtering Results', fontsize=16)
        
        # Check if we have results to plot
        if len(results_df) == 0:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No statistical results\nto display', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('No Data')
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
            
            # 3. Volcano plot
            if 'p_corrected' in results_df.columns:
                valid_data = results_df.dropna(subset=['eta_squared', 'p_corrected'])
                if len(valid_data) > 0:
                    x = valid_data['eta_squared']
                    y = -np.log10(valid_data['p_corrected'].replace(0, 1e-300))
                    
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
        Run the complete biological pre-filtering pipeline with proper microbiome data processing
        """
        print("="*80)
        print("BIOLOGICAL PRE-FILTERING PIPELINE")
        print("PROPER MICROBIOME DATA PROCESSING WORKFLOW")
        print("="*80)
        
        # Load data
        X_original, y, metadata = self.load_and_prepare_data(filepath)
        
        # Step 1: Handle missing values and data quality
        X_clean = self.step1_handle_missing_values(X_original)
        
        # Step 2: Sparse filtering
        X_filtered = self.step2_sparse_filtering(X_clean)
        
        # Step 3: Relative abundance + CLR transformation
        X_clr = self.step3_relative_abundance_and_clr(X_filtered)
        
        # Align y with X_clr (in case samples were removed)
        y_aligned = y.loc[X_clr.index]
        
        # Step 4: Statistical testing
        results_df = self.step4_statistical_testing(X_clr, y_aligned)
        
        # Step 5: Multiple testing correction
        if len(results_df) > 0:
            results_df = self.step5_multiple_testing_correction(results_df)
            
            # Step 6: Effect size filtering
            selected_features = self.step6_effect_size_filtering(results_df)
        else:
            print("No statistical results - using top features by variance")
            # Fallback: select features with highest variance
            feature_vars = X_clr.var(axis=0).sort_values(ascending=False)
            selected_features = feature_vars.head(50).index.tolist()
            
        # Create final dataset
        if len(selected_features) > 0:
            X_final = X_clr[selected_features]
        else:
            print("ERROR: No features selected! Using all features from CLR transformation.")
            X_final = X_clr
            selected_features = X_clr.columns.tolist()
        
        # Store results
        self.selected_features = selected_features
        self.statistical_results = results_df
        
        # Print processing summary
        self.print_processing_summary()
        
        # Visualize results
        self.visualize_results(results_df, X_filtered, y_aligned)
        
        print("="*80)
        print("PIPELINE COMPLETED!")
        print(f"Final dataset shape: {X_final.shape}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Data is ready for machine learning!")
        print("="*80)
        
        return X_final, y_aligned, selected_features, results_df

def run_step1(filepath):
    """Run biological pre-filtering with optimized parameters for more features"""
    bio_filter = BiologicalPrefiltering(
        min_prevalence=0.03,        # 3% of samples (relaxed from 8%)
        fdr_threshold=0.1,          # FDR < 0.1
        effect_size_threshold=0.003  # eta-squared >= 0.003 (relaxed from 0.005)
    )
    
    return bio_filter.run_complete_pipeline(filepath)