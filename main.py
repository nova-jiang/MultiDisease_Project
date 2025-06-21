"""
Main Controller for Multi-Class Microbiome Disease Classification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
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

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

DATA_PATH = "gmrepo_cleaned_dataset.csv" 
BASE_RESULTS_DIR = "results" 

# Pipeline control switches - Set to True/False to enable/disable steps
RUN_CONFIG = {
    # Step 1: Biological pre-filtering
    'step1_biological_filtering': True,
    
    # Step 2: Model-informed feature selection
    'step2_feature_selection': True,
    
    # Step 3: Individual ML models (nested cross-validation)
    'step3_svm': False,                    # SVM with nested CV
    'step3_knn': False,                    # KNN with nested CV
    'step3_lasso_regression': False,       # Lasso with nested CV
    'step3_random_forest': False,         # Random Forest with nested CV
    'step3_xgboost': False,                # XGBoost with nested CV
    'step3_naive_bayes': False,          # TODO: Implement
    'step3_neural_network': True,       # MLP Neural Network with nested CV
    
    # Step 4: Cross-validation and evaluation (deprecated - now done in Step 3)
    'step4_cross_validation': False,     # Integrated into nested CV
    'step4_final_evaluation': True,      # Final comparison and analysis
    
    # Additional options
    'save_intermediate_results': True,
    'generate_visualizations': True,
    'verbose': True
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
    'phases': ['XGBoost_ranking', 'RFECV']  # Two-phase approach
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
    
    print(f"Step 1 completed: {len(selected_features)} features selected")
    return X_filtered, y, selected_features, results_df

def run_step_2(X_step1, y, features_step1):
    """
    Step 2: Model-informed feature selection (2-phase approach)
    """
    if not RUN_CONFIG['step2_feature_selection']:
        print("Step 2: SKIPPED (disabled in configuration)")
        return features_step1, None
    
    print("\n" + "="*60)
    print("STEP 2: MODEL-INFORMED FEATURE SELECTION")
    print("="*60)
    
    if len(features_step1) < 10:
        print(f"WARNING: Only {len(features_step1)} features from Step 1. Consider relaxing Step 1 parameters.")
        return features_step1, None
    
    # Run feature selection (now only 2 phases)
    results_dir = f"{BASE_RESULTS_DIR}/step2_feature_selection"
    selected_features, selector = run_step2(X_step1, y, results_dir)

    save_pipeline_state('step2', X_step1[selected_features], selected_features, {
        'n_input_features': len(features_step1),
        'n_selected_features': len(selected_features),
        'parameters_used': STEP2_PARAMS,
        'phases_completed': ['Phase 1: XGBoost + Cumulative Analysis', 'Phase 2: RFECV']
    })
    
    print(f"Step 2 completed: {len(selected_features)} features selected")
    return selected_features, selector

def run_step_3_models(X_final, y, final_features):
    """
    Step 3: Train individual ML models with nested cross-validation
    """
    print("\n" + "="*60)
    print("STEP 3: MACHINE LEARNING MODELS (NESTED CV)")
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
    if RUN_CONFIG['step3_naive_bayes']:
        models_to_run.append('naive_bayes')
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
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/svm_nested"
                nested_results, classifier = run_svm_nested_cv(X_model_ready, y, results_dir)
                
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }
                
            elif model_name == 'knn_nested_cv':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/knn_nested"
                nested_results, classifier = run_knn_nested_cv(X_model_ready, y, results_dir)
                
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }
            
            elif model_name == 'xgboost':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/xgboost_nested"
                nested_results, classifier = run_xgboost_nested_cv(X_model_ready, y, results_dir)
            
                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'random_forest':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/random_forest_nested"
                nested_results, classifier = run_random_forest_nested_cv(X_model_ready, y, results_dir)

                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'neural_network':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/neural_network_nested"
                nested_results, classifier = run_neural_network_nested_cv(X_model_ready, y, results_dir)

                model_results[model_name] = {
                    'status': 'completed',
                    'type': 'nested_cross_validation',
                    'performance': nested_results['overall_metrics'],
                    'best_f1_macro': nested_results['overall_metrics']['f1_macro'],
                    'results_dir': results_dir
                }

            elif model_name == 'lasso_regression':
                results_dir = f"{BASE_RESULTS_DIR}/step3_models/lasso_nested"
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
    
    save_pipeline_state('step3', X_model_ready, final_features, {
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
            print("Step 1 was skipped or failed. Cannot proceed.")
            return
        
        # Step 2: Model-informed feature selection
        features_step2, selector_step2 = run_step_2(X_step1, y, features_step1)
        
        # Step 3: Machine learning models
        model_results = run_step_3_models(X_step1, y, features_step2)
        
        # Step 4: Evaluation
        evaluation_results = run_step_4_evaluation(model_results)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Original features: {X_step1.shape[1] if X_step1 is not None else 'N/A'}")
        print(f"After Step 1: {len(features_step1) if features_step1 else 'N/A'}")
        print(f"After Step 2: {len(features_step2) if features_step2 else 'N/A'}")
        print(f"Models run: {len(model_results)}")
        print(f"Results saved in: {BASE_RESULTS_DIR}")
        print("="*80)

        final_summary = {
            'pipeline_completion_time': datetime.now().isoformat(),
            'status': 'completed',
            'original_features': X_step1.shape[1] if X_step1 is not None else None,
            'features_after_step1': len(features_step1) if features_step1 else None,
            'features_after_step2': len(features_step2) if features_step2 else None,
            'final_features': features_step2 if features_step2 else None,
            'models_run': list(model_results.keys()) if model_results else [],
            'configuration_used': RUN_CONFIG,
            'step2_method': '2-phase: XGBoost + Cumulative Analysis → RFECV'
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