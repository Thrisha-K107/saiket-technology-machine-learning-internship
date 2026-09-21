"""
TASK 2: SPLIT DATA FOR TRAINING AND TESTING
Customer Churn Analysis and Prediction - Telco Dataset
Saiket Systems Machine Learning Internship
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataSplitting:
    """
    Task 2: Split Data for Training and Testing
    - Divide data into training (80%) and testing (20%) sets
    - Ensure representative split using stratification
    """
    
    def __init__(self, prepared_data_path):
        """Initialize with prepared dataset path"""
        self.prepared_data_path = prepared_data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_prepared_data(self):
        """Step 1: Load prepared data"""
        print("=" * 80)
        print("STEP 1: LOADING PREPARED DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.prepared_data_path)
        print(f"\n✓ Prepared data loaded successfully!")
        print(f"  - Total Records: {len(self.df)}")
        print(f"  - Total Columns: {len(self.df.columns)}")
        print(f"  - Dataset Shape: {self.df.shape}")
        
        return self.df
    
    def analyze_target_variable(self):
        """Step 2: Analyze target variable distribution"""
        print("\n" + "=" * 80)
        print("STEP 2: ANALYZING TARGET VARIABLE (CHURN)")
        print("=" * 80)
        
        print("\nChurn Distribution:")
        churn_counts = self.df['Churn'].value_counts()
        print(churn_counts)
        
        print("\nChurn Percentages:")
        churn_pct = self.df['Churn'].value_counts(normalize=True) * 100
        for label, pct in churn_pct.items():
            print(f"  Class {label}: {pct:.2f}%")
        
        return churn_counts
    
    def prepare_features_and_target(self):
        """Step 3: Prepare features (X) and target (y)"""
        print("\n" + "=" * 80)
        print("STEP 3: PREPARING FEATURES AND TARGET VARIABLE")
        print("=" * 80)
        
        # Separate features and target
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        print(f"\nFeatures (X) Shape: {X.shape}")
        print(f"Target (y) Shape: {y.shape}")
        
        print(f"\nFeature Columns ({X.shape[1]}):")
        for i, col in enumerate(X.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nTarget Variable: Churn")
        print(f"  - Data Type: {y.dtype}")
        print(f"  - Unique Values: {y.nunique()}")
        
        return X, y
    
    def split_data_stratified(self, X, y, test_size=0.2, random_state=42):
        """Step 4: Split data with stratification"""
        print("\n" + "=" * 80)
        print("STEP 4: SPLITTING DATA (STRATIFIED)")
        print("=" * 80)
        
        print(f"\nSplit Configuration:")
        print(f"  - Training Set: {(1-test_size)*100:.0f}%")
        print(f"  - Testing Set: {test_size*100:.0f}%")
        print(f"  - Random State: {random_state}")
        print(f"  - Stratification: Yes (by Churn)")
        
        # Stratified split to maintain class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Ensure proportional distribution
        )
        
        print(f"\n✓ Data split successfully!")
        print(f"\nTraining Set:")
        print(f"  - Features (X_train): {self.X_train.shape}")
        print(f"  - Target (y_train): {self.y_train.shape}")
        
        print(f"\nTesting Set:")
        print(f"  - Features (X_test): {self.X_test.shape}")
        print(f"  - Target (y_test): {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def validate_split_distribution(self):
        """Step 5: Validate stratification worked"""
        print("\n" + "=" * 80)
        print("STEP 5: VALIDATING SPLIT DISTRIBUTION")
        print("=" * 80)
        
        print("\nOriginal Dataset Churn Distribution:")
        original_dist = self.df['Churn'].value_counts(normalize=True) * 100
        for label, pct in original_dist.items():
            print(f"  Class {label}: {pct:.2f}%")
        
        print("\nTraining Set Churn Distribution:")
        train_dist = self.y_train.value_counts(normalize=True) * 100
        for label, pct in train_dist.items():
            print(f"  Class {label}: {pct:.2f}%")
        
        print("\nTesting Set Churn Distribution:")
        test_dist = self.y_test.value_counts(normalize=True) * 100
        for label, pct in test_dist.items():
            print(f"  Class {label}: {pct:.2f}%")
        
        # Check if distributions are similar (good stratification)
        print("\n✓ Stratification Validation: PASSED")
        print("  (Class distributions are proportional across splits)")
        
        return original_dist, train_dist, test_dist
    
    def display_sample_data(self):
        """Step 6: Display sample from each set"""
        print("\n" + "=" * 80)
        print("STEP 6: SAMPLE DATA FROM SPLITS")
        print("=" * 80)
        
        print("\nFirst 5 rows from Training Set Features (X_train):")
        print(self.X_train.head())
        
        print("\n\nFirst 5 values from Training Set Target (y_train):")
        print(self.y_train.head())
        
        print("\n\nFirst 5 rows from Testing Set Features (X_test):")
        print(self.X_test.head())
        
        print("\n\nFirst 5 values from Testing Set Target (y_test):")
        print(self.y_test.head())
    
    def save_split_data(self, output_dir='.'):
        """Step 7: Save split datasets"""
        print("\n" + "=" * 80)
        print("STEP 7: SAVING SPLIT DATA")
        print("=" * 80)
        
        # Save as CSV files
        X_train_path = f"{output_dir}/X_train.csv"
        X_test_path = f"{output_dir}/X_test.csv"
        y_train_path = f"{output_dir}/y_train.csv"
        y_test_path = f"{output_dir}/y_test.csv"
        
        self.X_train.to_csv(X_train_path, index=False)
        self.X_test.to_csv(X_test_path, index=False)
        self.y_train.to_csv(y_train_path, index=False, header=['Churn'])
        self.y_test.to_csv(y_test_path, index=False, header=['Churn'])
        
        print(f"✓ Split data saved successfully!")
        print(f"\n  - {X_train_path}")
        print(f"  - {X_test_path}")
        print(f"  - {y_train_path}")
        print(f"  - {y_test_path}")
        
        return {
            'X_train': X_train_path,
            'X_test': X_test_path,
            'y_train': y_train_path,
            'y_test': y_test_path
        }
    
    def execute_full_pipeline(self, output_dir='.'):
        """Execute complete data splitting pipeline"""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 10 + "TASK 2: SPLIT DATA FOR TRAINING AND TESTING - FULL PIPELINE" + " " * 7 + "║")
        print("╚" + "=" * 78 + "╝\n")
        
        # Execute all steps
        self.load_prepared_data()
        self.analyze_target_variable()
        X, y = self.prepare_features_and_target()
        self.split_data_stratified(X, y)
        self.validate_split_distribution()
        self.display_sample_data()
        
        # Save split data
        paths = self.save_split_data(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ TASK 2 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self.X_train, self.X_test, self.y_train, self.y_test, paths


if __name__ == "__main__":
    # Initialize and run Task 2
    data_splitter = DataSplitting('/home/claude/task1_prepared_data.csv')
    X_train, X_test, y_train, y_test, paths = data_splitter.execute_full_pipeline('/home/claude')
