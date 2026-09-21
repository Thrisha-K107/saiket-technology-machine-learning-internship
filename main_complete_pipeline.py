"""
COMPLETE MACHINE LEARNING PIPELINE
Customer Churn Analysis and Prediction - Telco Dataset
Saiket Systems Machine Learning Internship

This script runs all 4 main tasks in sequence:
1. Data Preparation
2. Data Splitting
3. Feature Selection
4. Model Training & Evaluation
"""

import sys
import os

# Import all task modules
from task_1_data_preparation import DataPreparation
from task_2_data_splitting import DataSplitting
from task_3_feature_selection import FeatureSelection
from task_4_5_model_training_evaluation import ModelTrainingEvaluation

def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "╔" + "=" * 98 + "╗")
    print("║" + title.center(98) + "║")
    print("╚" + "=" * 98 + "╝\n")

def print_separator(title, level=1):
    """Print a separator"""
    if level == 1:
        print("\n" + "█" * 100)
        print(f"█ {title.upper()}")
        print("█" * 100 + "\n")

def run_all_tasks(dataset_path, output_dir='.'):
    """Run all tasks in the pipeline"""
    
    print_banner("SAIKET SYSTEMS - CUSTOMER CHURN PREDICTION")
    print_banner("Complete Machine Learning Pipeline")
    
    print(f"\nDataset: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    
    try:
        # ============================================================================
        # TASK 1: DATA PREPARATION
        # ============================================================================
        print_separator("Task 1: Data Preparation")
        
        data_prep = DataPreparation(dataset_path)
        prepared_data = data_prep.execute_full_pipeline(f'{output_dir}/task1_prepared_data.csv')
        
        print("\n✓ Task 1 completed! File saved: task1_prepared_data.csv\n")
        input("Press Enter to continue to Task 2...")
        
        # ============================================================================
        # TASK 2: DATA SPLITTING
        # ============================================================================
        print_separator("Task 2: Split Data for Training and Testing")
        
        data_splitter = DataSplitting(f'{output_dir}/task1_prepared_data.csv')
        X_train, X_test, y_train, y_test, paths = data_splitter.execute_full_pipeline(output_dir)
        
        print("\n✓ Task 2 completed! Files saved:")
        for key, path in paths.items():
            print(f"  - {path}")
        print()
        input("Press Enter to continue to Task 3...")
        
        # ============================================================================
        # TASK 3: FEATURE SELECTION
        # ============================================================================
        print_separator("Task 3: Feature Selection")
        
        feature_selector = FeatureSelection(
            f'{output_dir}/X_train.csv',
            f'{output_dir}/X_test.csv',
            f'{output_dir}/y_train.csv',
            f'{output_dir}/y_test.csv'
        )
        selected_features, importance, feat_paths = feature_selector.execute_full_pipeline(
            n_features=None,
            output_dir=output_dir
        )
        
        print("\n✓ Task 3 completed! Files saved:")
        for key, path in feat_paths.items():
            print(f"  - {path}")
        print()
        input("Press Enter to continue to Task 4 & 5...")
        
        # ============================================================================
        # TASK 4 & 5: MODEL TRAINING AND EVALUATION
        # ============================================================================
        print_separator("Task 4 & 5: Model Training and Evaluation")
        
        model_trainer = ModelTrainingEvaluation(
            f'{output_dir}/X_train.csv',
            f'{output_dir}/X_test.csv',
            f'{output_dir}/y_train.csv',
            f'{output_dir}/y_test.csv'
        )
        models, metrics, best_model = model_trainer.execute_full_pipeline(output_dir)
        
        print("\n✓ Task 4 & 5 completed! Models and metrics saved.\n")
        
        # ============================================================================
        # SUMMARY
        # ============================================================================
        print_separator("Complete Pipeline Summary")
        
        print("\n✓ ALL TASKS COMPLETED SUCCESSFULLY!\n")
        
        print("Output Files Generated:")
        print("\nTask 1 - Data Preparation:")
        print("  - task1_prepared_data.csv")
        
        print("\nTask 2 - Data Splitting:")
        print("  - X_train.csv")
        print("  - X_test.csv")
        print("  - y_train.csv")
        print("  - y_test.csv")
        
        print("\nTask 3 - Feature Selection:")
        print("  - X_train_selected.csv")
        print("  - X_test_selected.csv")
        print("  - X_train_selected_scaled.csv")
        print("  - X_test_selected_scaled.csv")
        print("  - selected_features.txt")
        print("  - feature_importance.csv")
        
        print("\nTask 4 & 5 - Model Training & Evaluation:")
        print("  - logistic_regression_model.pkl")
        print("  - decision_tree_model.pkl")
        print("  - random_forest_model.pkl")
        print("  - gradient_boosting_model.pkl")
        print("  - model_performance_metrics.csv")
        
        print(f"\n★ Best Model Recommended: {best_model}")
        
        print("\n" + "=" * 100)
        print("Ready to push to GitHub!")
        print("=" * 100 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Get dataset path
    dataset_path = '/mnt/user-data/uploads/Telco_Customer_Churn_Dataset___1_.csv'
    output_dir = '/home/claude'
    
    # Run pipeline
    success = run_all_tasks(dataset_path, output_dir)
    
    if success:
        print("\n✓ Complete pipeline executed successfully!")
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)
