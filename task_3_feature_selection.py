"""
TASK 3: FEATURE SELECTION
Customer Churn Analysis and Prediction - Telco Dataset
Saiket Systems Machine Learning Internship
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureSelection:
    """
    Task 3: Feature Selection
    - Identify and select relevant features influencing churn prediction
    - Use statistical methods for feature importance
    """
    
    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path):
        """Initialize with training and testing data paths"""
        self.X_train_path = X_train_path
        self.X_test_path = X_test_path
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importance = None
        self.selected_features = None
        
    def load_split_data(self):
        """Step 1: Load split data"""
        print("=" * 80)
        print("STEP 1: LOADING SPLIT DATA")
        print("=" * 80)
        
        self.X_train = pd.read_csv(self.X_train_path)
        self.X_test = pd.read_csv(self.X_test_path)
        self.y_train = pd.read_csv(self.y_train_path).iloc[:, 0]
        self.y_test = pd.read_csv(self.y_test_path).iloc[:, 0]
        
        print(f"✓ Data loaded successfully!")
        print(f"\n  Training Features: {self.X_train.shape}")
        print(f"  Testing Features: {self.X_test.shape}")
        print(f"  Training Target: {self.y_train.shape}")
        print(f"  Testing Target: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def explore_features(self):
        """Step 2: Explore feature statistics"""
        print("\n" + "=" * 80)
        print("STEP 2: EXPLORING FEATURES")
        print("=" * 80)
        
        print(f"\nTotal Features: {len(self.X_train.columns)}")
        print(f"\nFeature List:")
        for i, col in enumerate(self.X_train.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nFeature Statistics (Training Data):")
        print(self.X_train.describe())
        
        return self.X_train.columns
    
    def scale_features(self):
        """Step 3: Normalize/Scale features"""
        print("\n" + "=" * 80)
        print("STEP 3: SCALING FEATURES")
        print("=" * 80)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.X_train.columns)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.X_test.columns)
        
        print(f"✓ Features scaled successfully!")
        print(f"\nScaled Training Data Statistics:")
        print(self.X_train_scaled.describe())
        
        return self.X_train_scaled, self.X_test_scaled, scaler
    
    def calculate_feature_importance(self):
        """Step 4: Calculate feature importance using F-score"""
        print("\n" + "=" * 80)
        print("STEP 4: CALCULATING FEATURE IMPORTANCE (F-SCORE)")
        print("=" * 80)
        
        # Calculate F-scores
        f_scores = f_classif(self.X_train_scaled, self.y_train)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'F-Score': f_scores[0],
            'P-Value': f_scores[1]
        }).sort_values('F-Score', ascending=False).reset_index(drop=True)
        
        print(f"\nFeature Importance Rankings:")
        print(feature_importance.to_string())
        
        self.feature_importance = feature_importance
        
        return feature_importance
    
    def calculate_mutual_information(self):
        """Step 5: Calculate mutual information"""
        print("\n" + "=" * 80)
        print("STEP 5: CALCULATING MUTUAL INFORMATION")
        print("=" * 80)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(self.X_train_scaled, self.y_train, random_state=42)
        
        # Create MI dataframe
        mi_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'MI-Score': mi_scores
        }).sort_values('MI-Score', ascending=False).reset_index(drop=True)
        
        print(f"\nMutual Information Rankings:")
        print(mi_importance.to_string())
        
        self.mi_importance = mi_importance
        
        return mi_importance
    
    def select_top_features(self, n_features=None):
        """Step 6: Select top N features"""
        print("\n" + "=" * 80)
        print("STEP 6: SELECTING TOP FEATURES")
        print("=" * 80)
        
        if n_features is None:
            # Select top 70% of features
            n_features = max(5, len(self.X_train.columns) // 2)
        
        print(f"\nSelecting top {n_features} features...")
        
        # Get top features based on F-score
        top_features = self.feature_importance.head(n_features)['Feature'].tolist()
        
        print(f"\n✓ Top {n_features} Selected Features:")
        for i, feat in enumerate(top_features, 1):
            f_score = self.feature_importance[self.feature_importance['Feature'] == feat]['F-Score'].values[0]
            print(f"  {i}. {feat} (F-Score: {f_score:.4f})")
        
        self.selected_features = top_features
        
        return top_features
    
    def identify_key_attributes(self):
        """Step 7: Identify key attributes affecting churn"""
        print("\n" + "=" * 80)
        print("STEP 7: KEY ATTRIBUTES AFFECTING CHURN")
        print("=" * 80)
        
        print(f"\nTop Features Influencing Churn Prediction:")
        
        top_5 = self.feature_importance.head(5)
        for idx, row in top_5.iterrows():
            print(f"\n  {idx+1}. {row['Feature']}")
            print(f"     - F-Score: {row['F-Score']:.4f}")
            print(f"     - P-Value: {row['P-Value']:.6f}")
        
        print(f"\nDomain Insights:")
        print(f"  • Contract type, Tenure, and Monthly Charges are key predictors")
        print(f"  • Service preferences (Internet, Phone) significantly affect churn")
        print(f"  • Customer support services correlate with retention")
        
        return self.feature_importance.head(5)
    
    def create_feature_reduced_datasets(self):
        """Step 8: Create datasets with selected features"""
        print("\n" + "=" * 80)
        print("STEP 8: CREATING FEATURE-REDUCED DATASETS")
        print("=" * 80)
        
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_selected_scaled = self.X_train_scaled[self.selected_features]
        X_test_selected_scaled = self.X_test_scaled[self.selected_features]
        
        print(f"\n✓ Feature-reduced datasets created!")
        print(f"\nOriginal Dataset:")
        print(f"  - X_train shape: {self.X_train.shape}")
        print(f"  - X_test shape: {self.X_test.shape}")
        
        print(f"\nReduced Dataset (Selected {len(self.selected_features)} features):")
        print(f"  - X_train_selected shape: {X_train_selected.shape}")
        print(f"  - X_test_selected shape: {X_test_selected.shape}")
        
        return X_train_selected, X_test_selected, X_train_selected_scaled, X_test_selected_scaled
    
    def save_selected_features(self, output_dir='.'):
        """Step 9: Save selected features data"""
        print("\n" + "=" * 80)
        print("STEP 9: SAVING SELECTED FEATURES")
        print("=" * 80)
        
        # Get feature-reduced datasets
        X_train_selected, X_test_selected, X_train_scaled, X_test_scaled = \
            self.create_feature_reduced_datasets()
        
        # Save files
        X_train_selected_path = f"{output_dir}/X_train_selected.csv"
        X_test_selected_path = f"{output_dir}/X_test_selected.csv"
        X_train_scaled_path = f"{output_dir}/X_train_selected_scaled.csv"
        X_test_scaled_path = f"{output_dir}/X_test_selected_scaled.csv"
        features_list_path = f"{output_dir}/selected_features.txt"
        importance_path = f"{output_dir}/feature_importance.csv"
        
        X_train_selected.to_csv(X_train_selected_path, index=False)
        X_test_selected.to_csv(X_test_selected_path, index=False)
        X_train_scaled.to_csv(X_train_scaled_path, index=False)
        X_test_scaled.to_csv(X_test_scaled_path, index=False)
        
        # Save features list
        with open(features_list_path, 'w') as f:
            f.write("Selected Features for Churn Prediction:\n")
            f.write("=" * 50 + "\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i}. {feat}\n")
        
        # Save importance
        self.feature_importance.to_csv(importance_path, index=False)
        
        print(f"✓ Selected features saved!")
        print(f"\n  - {X_train_selected_path}")
        print(f"  - {X_test_selected_path}")
        print(f"  - {X_train_scaled_path}")
        print(f"  - {X_test_scaled_path}")
        print(f"  - {features_list_path}")
        print(f"  - {importance_path}")
        
        return {
            'X_train_selected': X_train_selected_path,
            'X_test_selected': X_test_selected_path,
            'X_train_scaled': X_train_scaled_path,
            'X_test_scaled': X_test_scaled_path,
            'features_list': features_list_path,
            'importance': importance_path
        }
    
    def execute_full_pipeline(self, n_features=None, output_dir='.'):
        """Execute complete feature selection pipeline"""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "TASK 3: FEATURE SELECTION - FULL PIPELINE" + " " * 17 + "║")
        print("╚" + "=" * 78 + "╝\n")
        
        # Execute all steps
        self.load_split_data()
        self.explore_features()
        self.scale_features()
        self.calculate_feature_importance()
        self.calculate_mutual_information()
        self.select_top_features(n_features)
        self.identify_key_attributes()
        
        # Save results
        paths = self.save_selected_features(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self.selected_features, self.feature_importance, paths


if __name__ == "__main__":
    # Initialize and run Task 3
    feature_selector = FeatureSelection(
        '/home/claude/X_train.csv',
        '/home/claude/X_test.csv',
        '/home/claude/y_train.csv',
        '/home/claude/y_test.csv'
    )
    selected_features, importance, paths = feature_selector.execute_full_pipeline(output_dir='/home/claude')
