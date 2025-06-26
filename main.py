"""
Main Controller for Multi-Class Microbiome Disease Classification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.simplefilter('ignore')

from first_feature_selection import BiologicalPrefiltering
from step2_feature_selection import run_step2
from SVM_model import run_svm_nested_cv
from kNN_model import run_knn_nested_cv
from XGBoost_model import run_xgboost_nested_cv
from lasso_model import run_lasso_nested_cv
from random_forest import run_random_forest_nested_cv
from neural_network_model import run_neural_network_nested_cv
from mrmr_feature_selection import run_mrmr_feature_selection

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

DATA_PATH = "gmrepo_cleaned_dataset.csv" 
BASE_RESULTS_DIR = "results" 

# Pipeline control switches - Set to True/False to enable/disable steps
# Feature Selection Method control switches - change preprocessing method to 'mrmr'/ 'xgb_rfecv'

RUN_CONFIG = {
    # Step 1: Biological pre-filtering
    'step1_biological_filtering': False,
    
    # Step 2: XGB_RFECV feature selection
    'step2_feature_selection': True,
    
    # Step 3: Individual ML models (nested cross-validation)
    'step3_svm': True,                    # SVM with nested CV
    'step3_knn': True,                    # KNN with nested CV
    'step3_lasso_regression': True,       # Lasso with nested CV
    'step3_random_forest': True,         # Random Forest with nested CV
    'step3_xgboost': True,                # XGBoost with nested CV
    'step3_neural_network': True,       # MLP Neural Network with nested CV
    
    # Step 4: Cross-validation and evaluation (deprecated - now done in Step 3)
    'step4_cross_validation':  True,     # Integrated into nested CV
    'step4_final_evaluation': True,      # Final comparison and analysis
    
    # Additional options
    'save_intermediate_results': True,
    'generate_visualizations': True,
    'verbose': True,
    'feature_selection_method': 'mrmr'  # !!! To change preprocessing method change: 'mrmr' OR 'xgb_rfecv' !!!
}

# Step 1 parameters
STEP1_PARAMS = {
    'min_prevalence': 0.03,         # 3% of samples
    'fdr_threshold': 0.1,           # FDR < 0.1
    'effect_size_threshold': 0.005  # eta-squared >= 0.005
}

# Step 2 parameters
STEP2_PARAMS = {
    'xgb_top_k': 164,              # Number of features to select
    'cv_folds': 5,                 # Cross-validation folds
    'random_state': 42,            # Random seed
    'phases': ['XGBoost_ranking', 'RFECV'],  # Two-phase approach
    'mrmr_k': 50,                 # Default number of features for mRMR
    'mrmr_candidates': [10, 20, 50, 100, 150, 200, 250]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directory_structure():
    """Create the directory structure for results"""
    directories = [
        BASE_RESULTS_DIR,
        f"{BASE_RESULTS_DIR}/step1_biological_filtering",
        f"{BASE_RESULTS_DIR}/step2_feature_selection",
        f"{BASE_RESULTS_DIR}/step3_models",
        f"{BASE_RESULTS_DIR}/step4_evaluation",
        f"{BASE_RESULTS_DIR}/final_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created directory structure in: {BASE_RESULTS_DIR}")

def log_pipeline_start():
    """Log the start of the pipeline"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_info = {
        'pipeline_start_time': timestamp,
        'configuration': RUN_CONFIG,
        'step1_parameters': STEP1_PARAMS,
        'step2_parameters': STEP2_PARAMS,
        'data_path': DATA_PATH
    }
    
    with open(f"{BASE_RESULTS_DIR}/pipeline_log.json", 'w') as f:
        json.dump(log_info, f, indent=2)
    
    print("="*80)
    print("MULTI-CLASS MICROBIOME DISEASE CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"Start time: {timestamp}")
    print(f"Data path: {DATA_PATH}")
    print(f"Results directory: {BASE_RESULTS_DIR}")
    print("="*80)

def save_pipeline_state(step_name, data=None, features=None, results=None):
    """Save the current state of the pipeline"""
    if not RUN_CONFIG['save_intermediate_results']:
        return
    
    state = {
        'step': step_name,
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape if data is not None else None,
        'n_features': len(features) if features is not None else None,
        'features': features if features is not None else None
    }
    
    if results is not None:
        state['results_summary'] = results
    
    # Save state
    state_file = f"{BASE_RESULTS_DIR}/{step_name}_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"Pipeline state saved: {state_file}")

def save_step1_output(X, y):
    """Persist Step 1 output so later steps can run independently."""
    if not RUN_CONFIG.get('save_intermediate_results', True):
        return
    output_path = f"{BASE_RESULTS_DIR}/step1_biological_filtering/step1_output.csv"
    df = X.copy()
    df['label'] = y
    df.to_csv(output_path, index=False)
    print(f"Step 1 data saved to {output_path}")

def load_step1_output():
    """Load the saved Step 1 dataset if available."""
    path = f"{BASE_RESULTS_DIR}/step1_biological_filtering/step1_output.csv"
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("Saved Step 1 output missing 'label' column")
    y = df['label']
    X = df.drop(columns=['label'])
    features = X.columns.tolist()
    print(f"Loaded Step 1 data from {path}")
    return X, y, features

def load_step2_features():
    """Load the feature list produced by Step 2."""
    path = f"{BASE_RESULTS_DIR}/step2_feature_selection/final_selected_features.txt"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"Loaded Step 2 features from {path}")
    return features

def load_mrmr_features():
    """Load the feature list produced by the mRMR selection."""
    path = f"{BASE_RESULTS_DIR}/step2_feature_selection/mrmr/mrmr_selected_features.txt"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"Loaded mRMR features from {path}")
    return features

# =============================================================================
# PIPELINE STEPS
# =============================================================================

def run_step_1(data_path):
    """
    Step 1: Biological pre-filtering
    """
    if not RUN_CONFIG['step1_biological_filtering']:
        print("Step 1: SKIPPED (disabled in configuration)")
        return None, None, None, None
    
    print("\n" + "="*60)
    print("STEP 1: BIOLOGICAL PRE-FILTERING")
    print("="*60)

    
    bio_filter = BiologicalPrefiltering(
        min_prevalence=STEP1_PARAMS['min_prevalence'],
        fdr_threshold=STEP1_PARAMS['fdr_threshold'],
        effect_size_threshold=STEP1_PARAMS['effect_size_threshold']
    )
    
    # Run the pipeline
    X_filtered, y, selected_features, results_df = bio_filter.run_complete_pipeline(data_path)
    
    save_pipeline_state('step1', X_filtered, selected_features, {
        'n_original_features': len(results_df) if results_df is not None else 0,
        'n_selected_features': len(selected_features),
        'parameters_used': STEP1_PARAMS
    })

    # Persist dataset for future runs
    save_step1_output(X_filtered, y)
    
    print(f"Step 1 completed: {len(selected_features)} features selected")
    return X_filtered, y, selected_features, results_df

def run_step_2(X_step1, y, features_step1):
    """
    Step 2: MODEL-INFORMED feature selection (2-phase approach)
    """
    if not RUN_CONFIG['step2_feature_selection']:
        print("Step 2: SKIPPED (disabled in configuration)")
        return None, None, None
    
    print("\n" + "="*60)
    print("STEP 2: MODEL-INFORMED FEATURE SELECTION")
    print("="*60)
    
    if len(features_step1) < 10:
        print(
            f"WARNING: Only {len(features_step1)} features from Step 1. Consider relaxing Step 1 parameters."
        )
        return features_step1, None, None

    method = RUN_CONFIG.get('feature_selection_method', 'xgb_rfecv').lower()
    results_dir = f"{BASE_RESULTS_DIR}/step2_feature_selection"

    if method == 'mrmr':
        mrmr_dir = os.path.join(results_dir, 'mrmr')
        candidates = STEP2_PARAMS.get('mrmr_candidates', [STEP2_PARAMS.get('mrmr_k', 50)])
        selected_features = run_mrmr_feature_selection(
            X_step1, y, n_features_list=candidates, results_dir=mrmr_dir
        )
        selector = None
        save_pipeline_state(
            'step2_mrmr',
            X_step1[selected_features],
            selected_features,
            {
                'n_input_features': len(features_step1),
                'n_selected_features': len(selected_features),
                'method': 'mRMR',
                'parameters_used': {'candidates': candidates},
            },
        )
        print(f"Step 2 completed using mRMR: {len(selected_features)} features")
    else:
        selected_features, selector = run_step2(X_step1, y, results_dir)
        save_pipeline_state(
            'step2',
            X_step1[selected_features],
            selected_features,
            {
                'n_input_features': len(features_step1),
                'n_selected_features': len(selected_features),
                'parameters_used': STEP2_PARAMS,
                'phases_completed': ['Phase 1: XGBoost + Cumulative Analysis', 'Phase 2: RFECV'],
            },
        )
        print(f"Step 2 completed using XGB+RFECV: {len(selected_features)} features")

    return selected_features, selector, method

def run_step_3_models(X_final, y, final_features, feature_set_name="default"):
    """
    Step 3: Train individual ML models with nested cross-validation
    """
    print("\n" + "="*60)
    print(f"STEP 3: MACHINE LEARNING MODELS (NESTED CV) - {feature_set_name}")
    print("="*60)
    
    # Get the final dataset
    X_model_ready = X_final[final_features]
    
    models_to_run = []
    if RUN_CONFIG['step3_svm']:
        models_to_run.append('svm_nested_cv')
    if RUN_CONFIG['step3_knn']:
        models_to_run.append('knn_nested_cv')
    if RUN_CONFIG['step3_lasso_regression']:
        models_to_run.append('lasso_regression')
    if RUN_CONFIG['step3_random_forest']:
        models_to_run.append('random_forest')
    if RUN_CONFIG['step3_xgboost']:
        models_to_run.append('xgboost')
    if RUN_CONFIG['step3_neural_network']:
        models_to_run.append('neural_network')
    
    if not models_to_run:
        print("Step 3: No models enabled in configuration")
        return {}
    
    print(f"Models to run: {models_to_run}")
    
    # Model results storage
    model_results = {}
    
    for model_name in models_to_run:
        print(f"\nRunning {model_name}...")
        
        try:
            if model_name == 'svm_nested_cv':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_svm_nested"
                nested_results, classifier = run_svm_nested_cv(X_model_ready, y, results_dir)
                
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }
                
            elif model_name == 'knn_nested_cv':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_knn_nested"
                nested_results, classifier = run_knn_nested_cv(X_model_ready, y, results_dir)
                
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }
            
            elif model_name == 'xgboost':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_xgboost_nested"
                nested_results, classifier = run_xgboost_nested_cv(X_model_ready, y, results_dir)
            
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'random_forest':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_random_forest_nested"
                nested_results, classifier = run_random_forest_nested_cv(X_model_ready, y, results_dir)

                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'neural_network':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_neural_network_nested"
                nested_results, classifier = run_neural_network_nested_cv(X_model_ready, y, results_dir)

                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'lasso_regression':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/{feature_set_name}_lasso_nested"
                nested_results, classifier = run_lasso_nested_cv(X_model_ready, y, results_dir)
            
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }
                
            else:
                # Placeholder for other models (implement as needed)
                model_results[model_name] = {
                    'status': 'placeholder',
                    'message': f'{model_name} script not yet implemented'
                }
                print(f"  {model_name}: Placeholder (implement step3_{model_name}.py)")
                
        except Exception as e:
            print(f"  ERROR running {model_name}: {e}")
            model_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    save_pipeline_state(f'step3_{feature_set_name}', X_model_ready, final_features, {
        'models_run': models_to_run,
        'model_results': model_results,
        'completed_models': [name for name, result in model_results.items() 
                           if result['status'] == 'completed']
    })
    
    print(f"\nStep 3 Summary:")
    completed_models = [name for name, result in model_results.items() 
                       if result['status'] == 'completed']
    print(f"  Successfully completed: {len(completed_models)} models")
    
    if completed_models:
        print(f"  Performance Summary (F1-macro):")
        for model_name in completed_models:
            f1_score = model_results[model_name]['best_f1_macro']
            print(f"    {model_name}: {f1_score:.4f}")

    return model_results

def visualize_feature_method_comparison(results_a, results_b, method_a, method_b):
    """Create bar plots comparing feature selection methods for RF and NN."""
    models = ['random_forest', 'neural_network']
    for model in models:
        if (model not in results_a) or (model not in results_b):
            continue
        if results_a[model]['status'] != 'completed' or results_b[model]['status'] != 'completed':
            continue
        data = pd.DataFrame({
            'method': [method_a, method_b],
            'f1_macro': [results_a[model]['best_f1_macro'], results_b[model]['best_f1_macro']]
        })
        plt.figure(figsize=(6,4))
        sns.barplot(data=data, x='method', y='f1_macro')
        plt.title(f'{model} F1-macro by Feature Selection')
        plt.ylabel('F1-macro')
        plt.xlabel('Feature Selection Method')
        out_path = f"{BASE_RESULTS_DIR}/step3_models/{model}_feature_comparison.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

def run_step_4_evaluation(model_results):
    """
    Step 4: Cross-validation and final evaluation
    """
    print("\n" + "="*60)
    print("STEP 4: EVALUATION AND CROSS-VALIDATION")
    print("="*60)
    
    evaluation_results = {}
    
    if RUN_CONFIG['step4_cross_validation']:
        print("Running cross-validation analysis...")
        # Placeholder for cross-validation
        evaluation_results['cross_validation'] = "Placeholder - implement cross-validation"
    
    if RUN_CONFIG['step4_final_evaluation']:
        print("Running final evaluation...")
        # Placeholder for final evaluation
        evaluation_results['final_evaluation'] = "Placeholder - implement final evaluation"

    save_pipeline_state('step4', results=evaluation_results)
    
    return evaluation_results

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please update DATA_PATH in the configuration section.")
        return

    create_directory_structure()
    
    log_pipeline_start()
    
    try:
        # Step 1: Biological pre-filtering
        X_step1, y, features_step1, results_step1 = run_step_1(DATA_PATH)
        
        if X_step1 is None:
            # Attempt to load previous Step 1 output
            X_step1, y, features_step1 = load_step1_output()
            if X_step1 is None:
                print("Step 1 was skipped and no saved output found. Cannot proceed.")
                return
        
        # Step 2: Feature selection
        selected_features, selector_step2, method_used = run_step_2(X_step1, y, features_step1)

        if not RUN_CONFIG['step2_feature_selection'] and selected_features is None:
            if method_used == 'mrmr':
                selected_features = load_mrmr_features()
            else:
                selected_features = load_step2_features()
            if selected_features is None:
                print("Step 2 was skipped and no saved features found. Cannot proceed.")
                return

        # Step 3: Machine learning models
        model_results = run_step_3_models(X_step1, y, selected_features, feature_set_name=method_used)
        
        # Step 4: Evaluation
        evaluation_results = run_step_4_evaluation(model_results)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Original features: {X_step1.shape[1] if X_step1 is not None else 'N/A'}")
        print(f"After Step 1: {len(features_step1) if features_step1 else 'N/A'}")
        print(f"After Step 2 ({method_used}): {len(selected_features) if selected_features else 'N/A'}")
        print(f"Models run: {len(model_results)}")
        print(f"Results saved in: {BASE_RESULTS_DIR}")
        print("="*80)

        final_summary = {
            'pipeline_completion_time': datetime.now().isoformat(),
            'status': 'completed',
            'original_features': X_step1.shape[1] if X_step1 is not None else None,
            'features_after_step1': len(features_step1) if features_step1 else None,
            'features_after_step2': len(selected_features) if selected_features else None,
            'final_features': selected_features if selected_features else None,
            'models_run': list(model_results.keys()) if model_results else [],
            'configuration_used': RUN_CONFIG,
            'step2_method': method_used
        }
        
        with open(f"{BASE_RESULTS_DIR}/final_summary.json", 'w') as f:
            json.dump(final_summary, f, indent=2)
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        print("Check the logs and configuration settings.")
        
        # Save error log
        error_log = {
            'error_time': datetime.now().isoformat(),
            'error_message': str(e),
            'error_type': type(e).__name__
        }
        
        with open(f"{BASE_RESULTS_DIR}/error_log.json", 'w') as f:
            json.dump(error_log, f, indent=2)
        
        raise

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def quick_test_step1_only():
    """Quick test function to run only Step 1"""
    # Temporarily modify config
    global RUN_CONFIG
    original_config = RUN_CONFIG.copy()
    
    RUN_CONFIG = {key: False for key in RUN_CONFIG}
    RUN_CONFIG['step1_biological_filtering'] = True
    RUN_CONFIG['save_intermediate_results'] = True
    RUN_CONFIG['verbose'] = True
    
    try:
        main()
    finally:
        RUN_CONFIG = original_config

def quick_test_step1_and_2():
    """test function to run Steps 1 and 2 only"""
    global RUN_CONFIG
    original_config = RUN_CONFIG.copy()
    
    RUN_CONFIG = {key: False for key in RUN_CONFIG}
    RUN_CONFIG['step1_biological_filtering'] = True
    RUN_CONFIG['step2_feature_selection'] = True
    RUN_CONFIG['save_intermediate_results'] = True
    RUN_CONFIG['verbose'] = True
    
    try:
        main()
    finally:
        RUN_CONFIG = original_config


if __name__ == "__main__":
    # Choose what to run:
    
    # Uncomment to run the full pipeline (default)
    main()
    
    # Uncomment to run only Step 1
    # quick_test_step1_only()
    
    # Uncomment to run Step 1 and 2
    # quick_test_step1_and_2()
    
    # Option 4: Modify RUN_CONFIG above to customize which steps to run
