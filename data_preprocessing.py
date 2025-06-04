import pandas as pd
import numpy as np
import os
import requests
import json
import time
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import ranksums
import warnings
warnings.filterwarnings('ignore')

class EnhancedMicrobiomeFeatureSelection:
    def __init__(self, data_path, api_delay= 1):
        self.data_path = data_path
        self.api_delay = api_delay
        self.base_url = 'https://gmrepo.humangut.info/api/'
        
        # Define disease phenotype mapping
        self.phenotypes = {
            'D015212': 'Inflammatory_Bowel_Diseases',
            'D009765': 'Obesity', 
            'D006262': 'Health',
            'D003924': 'Diabetes Mellitus, Type 2',
            'D015179': 'Colorectal_Neoplasms'
        }
        
        # Disease codes excluding health for comparisons
        self.disease_codes = [code for code in self.phenotypes.keys() if code != 'D006262']
        
        self.all_species_data = {}
        self.abundance_matrix = None
        self.sample_metadata = None
        self.wilcoxon_results = {}
        self.shared_features = []
        self.disease_specific_features = {}
        
    def load_species_data(self):
        """
        Load all species association TSV files
        """
        print("=== Loading Species Association Data ===")
        
        filenames = [
            "species_associated_with_D015212",
            "species_associated_with_D009765", 
            "species_associated_with_D006262",
            "species_associated_with_D003924",
            "species_associated_with_D015179"
        ]
        
        for filename in filenames:
            file_path = os.path.join(self.data_path, filename + ".tsv")
            try:
                df = pd.read_csv(file_path, sep="\t")
                
                # Extract phenotype ID from filename
                phenotype_id = filename.split("_")[-1]
                df['phenotype_id'] = phenotype_id
                
                self.all_species_data[phenotype_id] = df
                print(f"Loaded {filename}: {df.shape[0]} species")
                
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        
        print(f"Total datasets loaded: {len(self.all_species_data)}")
        return self.all_species_data
    
    def create_initial_candidate_pool(self):
        """
        Create initial candidate pool with basic filtering
        """
        print("\n=== Creating Initial Candidate Pool ===")
        
        all_candidates = []
        
        for phenotype_id, df in self.all_species_data.items():
            print(f"Processing {phenotype_id} ({self.phenotypes.get(phenotype_id, 'Unknown')}):")
            
            # Basic filtering
            filtered_df = df.copy()
            
            # Frequency filtering: >= 10 runs (more lenient for initial pool)
            if 'Nr. runs in which the species/genus can be found' in filtered_df.columns:
                frequency_threshold = 10
                filtered_df = filtered_df[
                    filtered_df['Nr. runs in which the species/genus can be found'] >= frequency_threshold
                ]
                print(f"  After frequency filtering (≥{frequency_threshold} runs): {len(filtered_df)}")
            
            # Abundance filtering: >= 0.0005 (more lenient)
            if 'mean relative abundance' in filtered_df.columns:
                abundance_threshold = 0.0005
                filtered_df = filtered_df[
                    filtered_df['mean relative abundance'] >= abundance_threshold
                ]
                print(f"  After abundance filtering (≥{abundance_threshold}): {len(filtered_df)}")
            
            # Remove missing taxon IDs
            if 'NCBI taxon id' in filtered_df.columns:
                filtered_df = filtered_df.dropna(subset=['NCBI taxon id'])
                print(f"  After removing missing taxon IDs: {len(filtered_df)}")
            
            # Take top candidates by abundance for initial pool
            if len(filtered_df) > 0:
                # Take more candidates for statistical testing
                n_candidates = min(100, len(filtered_df))
                if 'mean relative abundance' in filtered_df.columns:
                    top_candidates = filtered_df.nlargest(n_candidates, 'mean relative abundance')
                else:
                    top_candidates = filtered_df.head(n_candidates)
                
                top_candidates['source_phenotype'] = phenotype_id
                all_candidates.append(top_candidates)
                print(f"  Selected {len(top_candidates)} candidates for statistical testing")
        
        # Combine all candidates
        if all_candidates:
            combined_candidates = pd.concat(all_candidates, ignore_index=True)
            print(f"\nTotal candidates before deduplication: {len(combined_candidates)}")
            
            # Deduplicate by NCBI taxon ID, keeping highest abundance
            if 'mean relative abundance' in combined_candidates.columns:
                unique_candidates = combined_candidates.loc[
                    combined_candidates.groupby('NCBI taxon id')['mean relative abundance'].idxmax()
                ]
            else:
                unique_candidates = combined_candidates.drop_duplicates(subset=['NCBI taxon id'], keep='first')
            
            print(f"Unique candidates for statistical testing: {len(unique_candidates)}")
            
            # Save candidate pool
            unique_candidates.to_csv('candidate_species_pool.csv', index=False)
            
            return unique_candidates['NCBI taxon id'].tolist()
        else:
            print("No candidates found!")
            return []
    
    """ simulated data I used to test feature selection method
    def simulate_abundance_matrix(self, candidate_taxon_ids):
        #Simulate abundance matrix for statistical testing
        
        print("\n=== Simulating Abundance Matrix ===")
        print("Note: This is a simulation. In real implementation, use API data.")
        
        # Simulate sample sizes based on earlier results
        sample_sizes = {
            'D015212': 300,  # IBD
            'D009765': 200,  # Obesity
            'D006262': 500,  # Health (largest)
            'D003924': 250,  # Diabetes
            'D015179': 180   # Colorectal
        }
        
        # Create sample IDs and metadata
        all_samples = []
        all_phenotypes = []
        
        for phenotype, n_samples in sample_sizes.items():
            samples = [f"{phenotype}_sample_{i:04d}" for i in range(n_samples)]
            all_samples.extend(samples)
            all_phenotypes.extend([phenotype] * n_samples)
        
        # Create abundance matrix (samples x taxa)
        n_samples = len(all_samples)
        n_taxa = len(candidate_taxon_ids)
        
        # Simulate realistic abundance data
        np.random.seed(42)  # For reproducibility
        
        # Base abundance matrix with realistic microbiome patterns
        abundance_matrix = np.random.exponential(scale=0.01, size=(n_samples, n_taxa))
        
        # Add disease-specific patterns
        phenotype_array = np.array(all_phenotypes)
        for i, taxon_id in enumerate(candidate_taxon_ids):
            # Some taxa are health-associated (decreased in disease)
            if i % 4 == 0:
                disease_mask = phenotype_array != 'D006262'
                abundance_matrix[disease_mask, i] *= 0.3
            
            # Some taxa are disease-associated (increased in specific diseases)
            elif i % 4 == 1:
                if i % 12 < 4:  # IBD-specific
                    ibd_mask = phenotype_array == 'D015212'
                    abundance_matrix[ibd_mask, i] *= 3
                elif i % 12 < 8:  # Diabetes-specific
                    diabetes_mask = phenotype_array == 'D003924'
                    abundance_matrix[diabetes_mask, i] *= 2.5
            
            # Some taxa are shared across diseases
            elif i % 4 == 2:
                multi_disease_mask = np.isin(phenotype_array, ['D015212', 'D003924', 'D009765'])
                abundance_matrix[multi_disease_mask, i] *= 2
        
        # Normalize to relative abundances (rows sum to 1)
        row_sums = abundance_matrix.sum(axis=1, keepdims=True)
        abundance_matrix = abundance_matrix / row_sums
        
        # Create DataFrames
        self.abundance_matrix = pd.DataFrame(
            abundance_matrix,
            index=all_samples,
            columns=candidate_taxon_ids
        )
        
        self.sample_metadata = pd.DataFrame({
            'run_id': all_samples,
            'phenotype': all_phenotypes
        }).set_index('run_id')
        
        print(f"Simulated abundance matrix: {self.abundance_matrix.shape}")
        print(f"Sample distribution:")
        print(self.sample_metadata['phenotype'].value_counts())
        
        # Save simulated data
        self.abundance_matrix.to_csv('simulated_abundance_matrix.csv')
        self.sample_metadata.to_csv('simulated_sample_metadata.csv')
        
        return self.abundance_matrix, self.sample_metadata
    """

    def perform_wilcoxon_analysis(self):
        """
        Perform Wilcoxon rank-sum tests for each disease vs Health
        """
        print("\n=== Performing Wilcoxon Rank-Sum Analysis ===")
        
        health_samples = self.sample_metadata[self.sample_metadata['phenotype'] == 'D006262'].index
        health_data = self.abundance_matrix.loc[health_samples]
        
        results = {}
        
        for disease_code in self.disease_codes:
            print(f"Analyzing {disease_code} vs Health...")
            
            disease_samples = self.sample_metadata[self.sample_metadata['phenotype'] == disease_code].index
            if len(disease_samples) == 0:
                print(f"  No samples found for {disease_code}")
                continue
                
            disease_data = self.abundance_matrix.loc[disease_samples]
            
            disease_results = []
            
            for taxon_id in self.abundance_matrix.columns:
                health_values = health_data[taxon_id].values
                disease_values = disease_data[taxon_id].values
                
                # Wilcoxon rank-sum test
                try:
                    statistic, p_value = ranksums(disease_values, health_values)
                    
                    # Calculate AUC as effect size
                    try:
                        # Combine labels: 0 for health, 1 for disease
                        y_true = np.concatenate([np.zeros(len(health_values)), np.ones(len(disease_values))])
                        y_scores = np.concatenate([health_values, disease_values])
                        auc = roc_auc_score(y_true, y_scores)
                        auc_effect = abs(auc - 0.5)  # Distance from no-discrimination
                    except:
                        auc = 0.5
                        auc_effect = 0
                    
                    # Calculate effect direction
                    mean_health = np.mean(health_values)
                    mean_disease = np.mean(disease_values)
                    effect_direction = 'increased' if mean_disease > mean_health else 'decreased'
                    
                    disease_results.append({
                        'taxon_id': taxon_id,
                        'disease': disease_code,
                        'p_value': p_value,
                        'auc': auc,
                        'auc_effect': auc_effect,
                        'mean_health': mean_health,
                        'mean_disease': mean_disease,
                        'effect_direction': effect_direction,
                        'n_health': len(health_values),
                        'n_disease': len(disease_values)
                    })
                    
                except Exception as e:
                    print(f"    Error with taxon {taxon_id}: {e}")
                    continue
            
            results[disease_code] = pd.DataFrame(disease_results)
            print(f"  Completed analysis for {len(disease_results)} taxa")
        
        self.wilcoxon_results = results
        
        # Save results
        for disease_code, df in results.items():
            df.to_csv(f'wilcoxon_results_{disease_code}.csv', index=False)
        
        return results
    
    def identify_shared_and_specific_features(self, p_threshold=0.01, min_diseases_for_shared=3):
        """
        Identify shared vs disease-specific features based on Wilcoxon results
        """
        print(f"\n=== Identifying Shared vs Disease-Specific Features ===")
        print(f"Significance threshold: p < {p_threshold}")
        print(f"Shared feature criterion: significant in ≥{min_diseases_for_shared} diseases")
        
        # Collect significance status for each taxon across diseases
        taxon_significance = defaultdict(dict)
        taxon_scores = defaultdict(list)
        
        for disease_code, results_df in self.wilcoxon_results.items():
            significant_taxa = results_df[results_df['p_value'] < p_threshold]
            
            for _, row in results_df.iterrows():
                taxon_id = row['taxon_id']
                is_significant = row['p_value'] < p_threshold
                
                taxon_significance[taxon_id][disease_code] = {
                    'significant': is_significant,
                    'p_value': row['p_value'],
                    'auc_effect': row['auc_effect'],
                    'effect_direction': row['effect_direction']
                }
                
                if is_significant:
                    taxon_scores[taxon_id].append(row['auc_effect'])
        
        # Classify features
        shared_features = []
        disease_specific_features = {disease: [] for disease in self.disease_codes}
        
        for taxon_id, disease_data in taxon_significance.items():
            significant_diseases = [
                disease for disease, data in disease_data.items() 
                if data['significant']
            ]
            
            n_significant = len(significant_diseases)
            
            if n_significant >= min_diseases_for_shared:
                # Shared feature
                avg_auc_effect = np.mean(taxon_scores[taxon_id])
                max_auc_effect = np.max(taxon_scores[taxon_id]) if taxon_scores[taxon_id] else 0
                
                shared_features.append({
                    'taxon_id': taxon_id,
                    'n_significant_diseases': n_significant,
                    'significant_diseases': significant_diseases,
                    'avg_auc_effect': avg_auc_effect,
                    'max_auc_effect': max_auc_effect,
                    'score': avg_auc_effect * n_significant  # Composite score
                })
                
            elif n_significant == 1:
                # Disease-specific feature
                specific_disease = significant_diseases[0]
                disease_data_specific = disease_data[specific_disease]
                
                disease_specific_features[specific_disease].append({
                    'taxon_id': taxon_id,
                    'disease': specific_disease,
                    'p_value': disease_data_specific['p_value'],
                    'auc_effect': disease_data_specific['auc_effect'],
                    'effect_direction': disease_data_specific['effect_direction']
                })
        
        # Sort shared features by score
        shared_features.sort(key=lambda x: x['score'], reverse=True)
        
        # Sort disease-specific features by AUC effect
        for disease in disease_specific_features:
            disease_specific_features[disease].sort(key=lambda x: x['auc_effect'], reverse=True)
        
        print(f"\nFeature Classification Results:")
        print(f"Shared features (≥{min_diseases_for_shared} diseases): {len(shared_features)}")
        for disease in self.disease_codes:
            n_specific = len(disease_specific_features[disease])
            print(f"Disease-specific for {disease}: {n_specific}")
        
        self.shared_features = shared_features
        self.disease_specific_features = disease_specific_features
        
        # Save classification results
        pd.DataFrame(shared_features).to_csv('shared_features_analysis.csv', index=False)
        
        for disease, features in disease_specific_features.items():
            if features:
                pd.DataFrame(features).to_csv(f'disease_specific_features_{disease}.csv', index=False)
        
        return shared_features, disease_specific_features
    
    def select_final_feature_set(self, target_shared=45, target_specific_per_disease=12):
        """
        Select final balanced feature set: shared + disease-specific
        """
        print(f"\n=== Selecting Final Feature Set ===")
        print(f"Target: {target_shared} shared + {target_specific_per_disease}×{len(self.disease_codes)} specific")
        
        final_features = []
        feature_details = []
        
        # Select shared features
        n_shared_available = len(self.shared_features)
        n_shared_select = min(target_shared, n_shared_available)
        
        selected_shared = self.shared_features[:n_shared_select]
        
        for feature in selected_shared:
            final_features.append(feature['taxon_id'])
            feature_details.append({
                'taxon_id': feature['taxon_id'],
                'feature_type': 'shared',
                'n_significant_diseases': feature['n_significant_diseases'],
                'significant_diseases': ','.join(map(str, feature['significant_diseases'])),
                'score': feature['score']
            })
        
        print(f"Selected shared features: {len(selected_shared)}")
        
        # Select disease-specific features
        total_specific_selected = 0
        
        for disease in self.disease_codes:
            disease_features = self.disease_specific_features[disease]
            n_available = len(disease_features)
            n_select = min(target_specific_per_disease, n_available)
            
            selected_disease_specific = disease_features[:n_select]
            
            for feature in selected_disease_specific:
                if feature['taxon_id'] not in final_features:  # Avoid duplicates
                    final_features.append(feature['taxon_id'])
                    feature_details.append({
                        'taxon_id': feature['taxon_id'],
                        'feature_type': f'specific_{disease}',
                        'disease': disease,
                        'auc_effect': feature['auc_effect'],
                        'effect_direction': feature['effect_direction'],
                        'p_value': feature['p_value']
                    })
                    total_specific_selected += 1
            
            print(f"Selected {disease} specific: {n_select} (available: {n_available})")
        
        print(f"Total specific features selected: {total_specific_selected}")
        print(f"Final feature set size: {len(final_features)}")
        
        # Save final feature set
        final_features_df = pd.DataFrame(feature_details)
        final_features_df.to_csv('final_selected_features.csv', index=False)
        
        # Create summary
        summary = {
            'total_features': len(final_features),
            'shared_features': len(selected_shared),
            'specific_features': total_specific_selected,
            'feature_breakdown': final_features_df['feature_type'].value_counts().to_dict()
        }
        
        print(f"\nFinal Feature Set Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return final_features, final_features_df, summary
    
    def get_abundance_data_via_api(self, mesh_id, ncbi_taxon_id):
        """
        Get abundance data for specific disease-microbe combination via API
        """
        url = f"{self.base_url}getMicrobeAbundancesByPhenotypeMeshIDAndNCBITaxonID"
        query = {'mesh_id': mesh_id, 'ncbi_taxon_id': str(int(ncbi_taxon_id))}
        
        try:
            response = requests.post(url, data=json.dumps(query))
            response.raise_for_status()
            data = response.json()
            
            # Extract abundance and metadata
            abundance_meta = data.get('abundance_and_meta_data', [])
            
            abundance_records = []
            for record in abundance_meta:
                abundance_records.append({
                    'run_id': record.get('run_id'),
                    'relative_abundance': record.get('relative_abundance', 0),
                    'phenotype': mesh_id,
                    'ncbi_taxon_id': int(ncbi_taxon_id)
                })
            
            return abundance_records
            
        except Exception as e:
            print(f"Error getting abundance for {mesh_id}-{ncbi_taxon_id}: {e}")
            return []
    
    def collect_real_abundance_matrix(self, selected_taxon_ids, target_samples_per_disease=200):
        """
        Collect real abundance matrix using API for selected features
        """
        print("\n=== Collecting Real Abundance Matrix via API ===")
        print(f"Target: {len(selected_taxon_ids)} features from {len(self.phenotypes)} diseases")
        print(f"Expected samples per disease: ~{target_samples_per_disease}")
        
        all_abundance_data = []
        total_calls = len(self.phenotypes) * len(selected_taxon_ids)
        current_call = 0
        failed_calls = []
        
        for disease_code in self.phenotypes.keys():
            print(f"\nProcessing {disease_code} ({self.phenotypes[disease_code]})...")
            
            for taxon_id in selected_taxon_ids:
                current_call += 1
                print(f"Progress: {current_call}/{total_calls} - {disease_code}-{taxon_id}")
                
                abundance_records = self.get_abundance_data_via_api(disease_code, taxon_id)
                
                if abundance_records:
                    all_abundance_data.extend(abundance_records)
                    print(f"  → Got {len(abundance_records)} samples")
                else:
                    failed_calls.append((disease_code, taxon_id))
                    print(f"  → No data returned")
                
                # Save intermediate results every 50 calls
                if current_call % 50 == 0:
                    df_temp = pd.DataFrame(all_abundance_data)
                    df_temp.to_csv(f'abundance_data_temp_{current_call}.csv', index=False)
                    print(f"  Saved intermediate results at call {current_call}")
                
                time.sleep(self.api_delay)
        
        print(f"\nAPI Collection Summary:")
        print(f"  Total API calls: {total_calls}")
        print(f"  Failed calls: {len(failed_calls)}")
        print(f"  Total abundance records: {len(all_abundance_data)}")
        
        if failed_calls:
            print("Failed calls (disease-taxon combinations):")
            for disease, taxon in failed_calls[:10]:  # Show first 10
                print(f"  {disease}-{taxon}")
            if len(failed_calls) > 10:
                print(f"  ... and {len(failed_calls) - 10} more")
        
        # Save raw abundance data
        df_abundance = pd.DataFrame(all_abundance_data)
        df_abundance.to_csv('real_abundance_data_raw.csv', index=False)
        print("Saved raw abundance data to 'real_abundance_data_raw.csv'")
        
        return df_abundance
    
    def process_abundance_matrix(self, df_abundance, selected_taxon_ids):
        """
        Process abundance data into ML-ready format with proper normalization
        """
        print("\n=== Processing Abundance Matrix ===")
        
        # Check data quality
        print(f"Raw abundance records: {len(df_abundance)}")
        print(f"Unique samples: {df_abundance['run_id'].nunique()}")
        print(f"Unique taxa: {df_abundance['ncbi_taxon_id'].nunique()}")
        print(f"Samples per disease:")
        print(df_abundance.groupby('phenotype')['run_id'].nunique())
        
        # Create abundance matrix: samples × features
        print("\nCreating abundance matrix...")
        abundance_matrix = df_abundance.pivot_table(
            index='run_id',
            columns='ncbi_taxon_id',
            values='relative_abundance',
            fill_value=0  # Step 1: Fill missing with 0
        )
        
        # Ensure all selected features are present as columns
        missing_features = set(selected_taxon_ids) - set(abundance_matrix.columns)
        if missing_features:
            print(f"Adding {len(missing_features)} missing features as zero columns")
            for taxon_id in missing_features:
                abundance_matrix[taxon_id] = 0
        
        # Reorder columns to match selected features
        abundance_matrix = abundance_matrix[selected_taxon_ids]
        
        print(f"Abundance matrix shape: {abundance_matrix.shape}")
        
        # Create sample metadata from abundance data
        sample_metadata = df_abundance[['run_id', 'phenotype']].drop_duplicates()
        sample_metadata.set_index('run_id', inplace=True)
        
        # Align matrices (keep only samples present in both)
        common_samples = abundance_matrix.index.intersection(sample_metadata.index)
        abundance_matrix = abundance_matrix.loc[common_samples]
        sample_metadata = sample_metadata.loc[common_samples]
        
        print(f"Final aligned data:")
        print(f"  Matrix shape: {abundance_matrix.shape}")
        print(f"  Samples per disease:")
        print(sample_metadata['phenotype'].value_counts())
        
        # Step 2: Row normalization (relative abundances sum to 1)
        print("\nApplying row normalization...")
        abundance_matrix_norm = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
        abundance_matrix_norm = abundance_matrix_norm.fillna(0)  # Handle division by zero
        
        print(f"Row sums after normalization (should be ~1.0):")
        row_sums = abundance_matrix_norm.sum(axis=1)
        print(f"  Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}, Mean: {row_sums.mean():.6f}")
        
        # Step 3: CLR transformation (Centered Log-Ratio)
        print("\nApplying CLR transformation...")
        epsilon = 1e-6  # Small constant to avoid log(0)
        
        # Add epsilon and take log
        log_matrix = np.log(abundance_matrix_norm + epsilon)
        
        # Subtract row means (centering)
        row_means = log_matrix.mean(axis=1, keepdims=True)
        abundance_matrix_clr = log_matrix - row_means
        
        # Convert back to DataFrame
        abundance_matrix_clr = pd.DataFrame(
            abundance_matrix_clr,
            index=abundance_matrix_norm.index,
            columns=abundance_matrix_norm.columns
        )
        
        print(f"CLR matrix statistics:")
        print(f"  Shape: {abundance_matrix_clr.shape}")
        print(f"  Mean: {abundance_matrix_clr.values.mean():.6f}")
        print(f"  Std: {abundance_matrix_clr.values.std():.6f}")
        print(f"  Row means (should be ~0): {abundance_matrix_clr.mean(axis=1).abs().max():.6f}")
        
        # Save all matrix versions
        abundance_matrix.to_csv('abundance_matrix_raw.csv')
        abundance_matrix_norm.to_csv('abundance_matrix_normalized.csv')
        abundance_matrix_clr.to_csv('abundance_matrix_clr.csv')
        sample_metadata.to_csv('sample_metadata_final.csv')
        
        print("\nSaved matrices:")
        print("- abundance_matrix_raw.csv: Raw abundance values")
        print("- abundance_matrix_normalized.csv: Row-normalized (sum=1)")
        print("- abundance_matrix_clr.csv: CLR-transformed (ML-ready)")
        print("- sample_metadata_final.csv: Sample labels")
        
        # Store for further analysis
        self.abundance_matrix = abundance_matrix_clr  # Use CLR for ML
        self.sample_metadata = sample_metadata
        
        return abundance_matrix_clr, sample_metadata
    
    def validate_sample_labels(self, sample_to_disease_file=None):
        """
        Validate sample labels using external disease info file
        """
        if sample_to_disease_file and os.path.exists(sample_to_disease_file):
            print(f"\n=== Validating Sample Labels ===")
            
            # Load external disease info
            disease_info = pd.read_csv(sample_to_disease_file, sep='\t')
            print(f"External disease info: {len(disease_info)} records")
            
            # Compare with our sample metadata
            our_samples = set(self.sample_metadata.index)
            external_samples = set(disease_info['run_id'])
            
            overlap = our_samples.intersection(external_samples)
            print(f"Sample overlap: {len(overlap)}/{len(our_samples)} ({len(overlap)/len(our_samples)*100:.1f}%)")
            
            if overlap:
                # Check label consistency for overlapping samples
                our_labels = self.sample_metadata.loc[list(overlap), 'phenotype']
                external_labels = disease_info.set_index('run_id').loc[list(overlap), 'disease']
                
                mismatches = sum(our_labels != external_labels)
                print(f"Label mismatches: {mismatches}/{len(overlap)} ({mismatches/len(overlap)*100:.1f}%)")
                
                if mismatches > 0:
                    print("Sample mismatches (first 5):")
                    mismatch_samples = our_labels[our_labels != external_labels].head()
                    for sample_id in mismatch_samples.index:
                        print(f"  {sample_id}: API={our_labels[sample_id]} vs External={external_labels[sample_id]}")
            
            return overlap
        else:
            print("No external disease info file provided for validation")
            return None
    
    def run_complete_pipeline(self, sample_to_disease_file=None, use_selected_features=True):
        """
        Run the complete pipeline: feature selection + real data collection + processing
        """
        print("Starting Complete Microbiome ML Pipeline")
        print("="*70)
        print("Phase 1: Statistical Feature Selection")
        print("Phase 2: Real API Data Collection") 
        print("Phase 3: Data Processing & ML Preparation")
        print("="*70)
        
        if use_selected_features:
            # Use pre-selected features from previous statistical analysis
            try:
                final_features_df = pd.read_csv('final_selected_features.csv')
                final_features = final_features_df['taxon_id'].tolist()
                print(f"Loaded {len(final_features)} pre-selected features from final_selected_features.csv")
            except FileNotFoundError:
                print("No pre-selected features found. Running statistical pipeline first...")
                use_selected_features = False
        
        if not use_selected_features:
            # Phase 1: Statistical Feature Selection
            print("\n" + "="*50)
            print("PHASE 1: STATISTICAL FEATURE SELECTION")
            print("="*50)
            
            # Step 1: Load data
            self.load_species_data()
            
            # Step 2: Create candidate pool
            candidate_taxon_ids = self.create_initial_candidate_pool()
            
            if not candidate_taxon_ids:
                print("No candidates found. Exiting.")
                return None, None, None
            
            # Step 3: Simulate abundance matrix for statistical testing
            self.simulate_abundance_matrix(candidate_taxon_ids)
            
            # Step 4: Perform Wilcoxon analysis
            self.perform_wilcoxon_analysis()
            
            # Step 5: Identify shared vs specific features
            self.identify_shared_and_specific_features()
            
            # Step 6: Select final feature set
            final_features, features_df, summary = self.select_final_feature_set()
        else:
            final_features = final_features_df['taxon_id'].tolist()
        
        # Phase 2: Real API Data Collection
        print("\n" + "="*50)
        print("PHASE 2: REAL API DATA COLLECTION")
        print("="*50)
        
        df_abundance = self.collect_real_abundance_matrix(final_features)
        
        if len(df_abundance) == 0:
            print("No abundance data collected. Exiting.")
            return None, None, None
        
        # Phase 3: Data Processing & ML Preparation
        print("\n" + "="*50)
        print("PHASE 3: DATA PROCESSING & ML PREPARATION")
        print("="*50)
        
        abundance_matrix_clr, sample_metadata = self.process_abundance_matrix(df_abundance, final_features)
        
        # Optional: Validate sample labels
        if sample_to_disease_file:
            self.validate_sample_labels(sample_to_disease_file)
        
        print("\n" + "="*70)
        print("COMPLETE PIPELINE FINISHED!")
        print("="*70)
        print("ML-Ready Data:")
        print(f"- Features (X): {abundance_matrix_clr.shape[1]} selected microbes")
        print(f"- Samples (n): {abundance_matrix_clr.shape[0]} total samples")
        print(f"- Classes (y): {sample_metadata['phenotype'].nunique()} disease categories")
        print("\nSample distribution:")
        print(sample_metadata['phenotype'].value_counts())
        print("\nKey Files:")
        print("- abundance_matrix_clr.csv: CLR-transformed features (use for ML)")
        print("- sample_metadata_final.csv: Sample labels")
        print("- final_selected_features.csv: Feature information")
        print("="*70)
        print("Ready for Models: RandomForest, XGBoost, LogisticRegression, SVM, kNN, NN")
        print("="*70)
        
        return abundance_matrix_clr, sample_metadata, final_features

    def run_statistical_pipeline_only(self):
        """
        Run only the statistical feature selection pipeline (for testing)
        """
        print("Starting Enhanced Statistical Feature Selection Pipeline")
        print("="*70)
        print("Approach: Shared vs Disease-Specific Features using Wilcoxon Tests")
        print(f"Diseases: {[self.phenotypes[code] for code in self.disease_codes]}")
        print("="*70)
        
        # Step 1: Load data
        self.load_species_data()
        
        # Step 2: Create candidate pool
        candidate_taxon_ids = self.create_initial_candidate_pool()
        
        if not candidate_taxon_ids:
            print("No candidates found. Exiting.")
            return None, None, None
        
        # Step 3: Simulate abundance matrix for statistical testing
        self.simulate_abundance_matrix(candidate_taxon_ids)
        
        # Step 4: Perform Wilcoxon analysis
        self.perform_wilcoxon_analysis()
        
        # Step 5: Identify shared vs specific features
        self.identify_shared_and_specific_features()
        
        # Step 6: Select final feature set
        final_features, features_df, summary = self.select_final_feature_set()
        
        print("="*70)
        print("Statistical Pipeline Completed!")
        print("Key outputs:")
        print("- wilcoxon_results_*.csv: Statistical test results for each disease")
        print("- shared_features_analysis.csv: Multi-disease shared features")
        print("- disease_specific_features_*.csv: Disease-specific features")
        print("- final_selected_features.csv: Final balanced feature set")
        print("="*70)
        print("Next step: Run complete pipeline to get real API data")
        print("="*70)
        
        return final_features, features_df, summary

if __name__ == "__main__":
    data_path = "/MultiDisease/readed_data" 
    sample_to_disease_file = "/MultiDisease/readed_data"  # validation file
    
    pipeline = EnhancedMicrobiomeFeatureSelection(data_path, api_delay=1)
    

    print("Running complete pipeline...")
    X, y_metadata, features = pipeline.run_complete_pipeline(
        sample_to_disease_file=sample_to_disease_file,
        use_selected_features=True  # Flag feature selection
    )
