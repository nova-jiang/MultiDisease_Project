"""
Complete Multi-Disease Classification Pipeline
==========================================

This is the main function to run the complete pipeline for microbiome-based multi-disease classification:
1. Step 1: Biological Statistical Pre-filtering
2. Step 2: SVM-Assisted Feature Selection
3. Step 3a: SVM Model Training
4. Step 3b: KNN Model Training
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from first_feature_selection import BiologicalPrefiltering
from SVM_feature_selection import SVMFeatureSelection
from SVM_model import SVMModelTraining
from kNN_model import KNNModelTraining
from XGBoost_model import XGBoostModelTraining
from lasso_model import LassoModelTraining

def main():
    """
    Run the complete multi-disease classification pipeline
    """
    print("="*80)
    print("MULTI-DISEASE CLASSIFICATION PIPELINE")
    print("Microbiome-based Disease Category Classification")
    print("="*80)
    
    # Configuration parameters
    DATA_FILE = 'gmrepo_cleaned_dataset.csv'
    RANDOM_STATE = 42
    
    try:
        # =====================================================================
        # STEP 1: BIOLOGICAL STATISTICAL PRE-FILTERING
        # =====================================================================
        print("\n🧬 STEP 1: BIOLOGICAL STATISTICAL PRE-FILTERING")
        print("-" * 60)
        
        # Initialize biological pre-filtering
        bio_filter = BiologicalPrefiltering(
            min_prevalence=0.1,      # Keep features present in ≥10% of samples
            fdr_threshold=0.05,      # FDR < 0.05
            effect_size_threshold=0.01  # Minimum effect size
        )
        
        # Run Step 1
        print("Running biological pre-filtering...")
        X_filtered, y, selected_features_step1, results_df_step1 = bio_filter.run_complete_pipeline(DATA_FILE)
        
        print(f"✅ Step 1 completed successfully!")
        print(f"   Features after biological filtering: {len(selected_features_step1)}")
        print(f"   Dataset shape: {X_filtered.shape}")
        
        # =====================================================================
        # STEP 2: SVM-ASSISTED FEATURE SELECTION
        # =====================================================================
        print("\n🎯 STEP 2: SVM-ASSISTED FEATURE SELECTION")
        print("-" * 60)
        
        # Initialize SVM feature selection
        svm_selector = SVMFeatureSelection(
            C=1.0,                          # SVM regularization parameter
            kernel='linear',                # Linear kernel for interpretability
            max_features=50,                # Maximum features to consider
            cv_folds=5,                     # 5-fold cross-validation
            feature_selection_method='coef' # Use coefficient-based selection
        )
        
        # Run Step 2
        print("Running SVM-assisted feature selection...")
        selected_features_final, feature_rankings, optimal_n = svm_selector.run_complete_pipeline(
            X_filtered, y
        )
        
        print(f"✅ Step 2 completed successfully!")
        print(f"   Optimal number of features: {optimal_n}")
        print(f"   Final selected features: {len(selected_features_final)}")
        
        # =====================================================================
        # STEP 3A: SVM MODEL TRAINING
        # =====================================================================
        print("\n🤖 STEP 3A: SVM MODEL TRAINING")
        print("-" * 60)
        
        # Initialize SVM trainer
        svm_trainer = SVMModelTraining(
            cv_folds=5,         # 5-fold cross-validation
            test_size=0.2,      # 20% for testing
            random_state=RANDOM_STATE
        )
        
        # Run SVM training
        print("Training SVM model...")
        svm_results = svm_trainer.run_complete_pipeline(
            X_filtered, y, selected_features_final
        )
        
        print(f"✅ SVM training completed successfully!")
        print(f"   SVM test accuracy: {svm_results['test_results']['accuracy']:.4f}")
        
        # =====================================================================
        # STEP 3B: KNN MODEL TRAINING
        # =====================================================================
        print("\n🎯 STEP 3B: KNN MODEL TRAINING")
        print("-" * 60)
        
        # Initialize KNN trainer
        knn_trainer = KNNModelTraining(
            cv_folds=5,         # 5-fold cross-validation
            test_size=0.2,      # 20% for testing
            random_state=RANDOM_STATE
        )
        
        # Run KNN training
        print("Training KNN model...")
        knn_results = knn_trainer.run_complete_pipeline(
            X_filtered, y, selected_features_final
        )
        
        print(f"✅ KNN training completed successfully!")
        print(f"   KNN test accuracy: {knn_results['test_results']['accuracy']:.4f}")

        # =====================================================================
        # STEP 3C: XGBOOST MODEL TRAINING
        # =====================================================================
        print("\n🎯 STEP 3C: XGBOOST MODEL TRAINING")
        print("-" * 60)

        # Initialize XGBoost trainer
        xgb_trainer = XGBoostModelTraining(
            cv_folds=5,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

        # Run XGBoost training
        print("Training XGBoost model...")
        xgb_results = xgb_trainer.run_complete_pipeline(
            X_filtered, y, selected_features_final
        )

        print(f"✅ XGBoost training completed successfully!")
        print(f"   XGBoost test accuracy: {xgb_results['test_results']['accuracy']:.4f}")

        # =====================================================================
        # STEP 3D: LASSO MODEL TRAINING
        # =====================================================================
        print("\n🧪 STEP 3D: LASSO MODEL TRAINING")
        print("-" * 60)

        # Initialize Lasso trainer
        lasso_trainer = LassoModelTraining(
            cv_folds=5,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

        # Run Lasso training
        print("Training Lasso model...")
        lasso_results = lasso_trainer.run_complete_pipeline(
            X_filtered, y, selected_features_final
        )

        print(f"✅ Lasso training completed successfully!")
        print(f"   Lasso test accuracy: {lasso_results['test_results']['accuracy']:.4f}")



        
        # =====================================================================
        # PIPELINE SUMMARY
        # =====================================================================
        print("\n📊 PIPELINE SUMMARY")
        print("=" * 80)
        
        # Create summary report
        pipeline_summary = {
            "Pipeline Configuration": {
                "Data file": DATA_FILE,
                "Random state": RANDOM_STATE,
                "Total samples": X_filtered.shape[0],
                "Disease categories": len(y.unique())
            },
            "Step 1 - Biological Filtering": {
                "Initial features": results_df_step1.shape[0] if results_df_step1 is not None else "N/A",
                "Features after filtering": len(selected_features_step1),
                "Filtering criteria": "Prevalence ≥10%, FDR <0.05, Effect size ≥0.01"
            },
            "Step 2 - Feature Selection": {
                "Input features": len(selected_features_step1),
                "Selected features": len(selected_features_final),
                "Selection method": "SVM coefficient-based",
                "Optimal number": optimal_n
            },
            "Step 3A - SVM Results": {
                "Best parameters": svm_results['best_params'],
                "CV accuracy": f"{svm_results['cv_results']['accuracy']['test_mean']:.4f} ± {svm_results['cv_results']['accuracy']['test_std']:.4f}",
                "Test accuracy": f"{svm_results['test_results']['accuracy']:.4f}",
                "Test F1-macro": f"{svm_results['test_results']['f1_macro']:.4f}"
            },
            "Step 3B - KNN Results": {
                "Best parameters": knn_results['best_params'],
                "CV accuracy": f"{knn_results['cv_results']['accuracy']['test_mean']:.4f} ± {knn_results['cv_results']['accuracy']['test_std']:.4f}",
                "Test accuracy": f"{knn_results['test_results']['accuracy']:.4f}",
                "Test F1-macro": f"{knn_results['test_results']['f1_macro']:.4f}",
                "Optimal k": knn_results['k_analysis']['optimal_k']
            },
            "Step 3C - XGBoost Results": {
                "Best parameters": xgb_results['best_params'],
                "CV accuracy": f"{xgb_results['cv_results']['accuracy']['test_mean']:.4f} ± {xgb_results['cv_results']['accuracy']['test_std']:.4f}",
                "Test accuracy": f"{xgb_results['test_results']['accuracy']:.4f}",
                "Test F1-macro": f"{xgb_results['test_results']['f1_macro']:.4f}"
            },
            "Step 3D - Lasso Results": {
                "Best parameters": lasso_results['best_params'],
                "CV accuracy": f"{lasso_results['cv_results']['accuracy']['test_mean']:.4f} ± {lasso_results['cv_results']['accuracy']['test_std']:.4f}",
                "Test accuracy": f"{lasso_results['test_results']['accuracy']:.4f}",
                "Test F1-macro": f"{lasso_results['test_results']['f1_macro']:.4f}"
            }

        }
        
        # Print summary
        for section, details in pipeline_summary.items():
            print(f"\n{section}:")
            print("-" * len(section))
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # Determine best model
        svm_accuracy = svm_results['test_results']['accuracy']
        knn_accuracy = knn_results['test_results']['accuracy']
        xgb_accuracy = xgb_results['test_results']['accuracy']
        lasso_accuracy = lasso_results['test_results']['accuracy']

        best_model = max(
            [("SVM", svm_accuracy), ("KNN", knn_accuracy), ("XGBoost", xgb_accuracy), ("Lasso", lasso_accuracy)],
            key=lambda x: x[1]
        )

        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")
        print(f"   Best test accuracy: {best_model[1]:.4f}")

        
        # Save pipeline summary
        import json
        from datetime import datetime
        
        summary_filename = f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_filename, 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_summary = {}
            for section, details in pipeline_summary.items():
                json_summary[section] = {}
                for key, value in details.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_summary[section][key] = value.item()
                    else:
                        json_summary[section][key] = value
            
            json.dump(json_summary, f, indent=2)
        
        print(f"\n💾 Pipeline summary saved to: {summary_filename}")
        
        # =====================================================================
        # FINAL FEATURE LIST
        # =====================================================================
        print(f"\n📋 FINAL SELECTED FEATURES ({len(selected_features_final)}):")
        print("-" * 60)
        for i, feature in enumerate(selected_features_final, 1):
            print(f"  {i:2d}. {feature}")
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            'step1_results': (X_filtered, y, selected_features_step1, results_df_step1),
            'step2_results': (selected_features_final, feature_rankings, optimal_n),
            'svm_results': svm_results,
            'knn_results': knn_results,
            'pipeline_summary': pipeline_summary
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with error: {str(e)}")
        print("Please check your data.")
        raise e

def validate_environment():
    """
    Validate that all required files and dependencies are available
    """
    print("🔍 Validating environment...")
    
    # Check required files
    required_files = [
        'gmrepo_cleaned_dataset.csv',
        'first_feature_selection.py',
        'SVM_feature_selection.py', 
        'SVM_model.py',
        'kNN_model.py'
    ]
    
    import os
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check required packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        return False
    
    print("✅ Environment validation passed!")
    return True

if __name__ == "__main__":
    # Validate environment first
    if validate_environment():
        # Run the complete pipeline
        results = main()
    else:
        print("Please check.")