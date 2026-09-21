"""
TASK 4 & 5: MODEL SELECTION, TRAINING AND EVALUATION
Customer Churn Analysis and Prediction - Telco Dataset
Saiket Systems Machine Learning Internship
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainingEvaluation:
    """
    Task 4 & 5: Model Selection, Training and Evaluation
    - Select suitable binary classification algorithms
    - Train models on training dataset
    - Evaluate on testing dataset
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
        self.models = {}
        self.predictions = {}
        self.performance_metrics = {}
        
    def load_data(self):
        """Step 1: Load data"""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        
        self.X_train = pd.read_csv(self.X_train_path)
        self.X_test = pd.read_csv(self.X_test_path)
        self.y_train = pd.read_csv(self.y_train_path).iloc[:, 0]
        self.y_test = pd.read_csv(self.y_test_path).iloc[:, 0]
        
        print(f"✓ Data loaded successfully!")
        print(f"\n  Training set: {self.X_train.shape}")
        print(f"  Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def explore_data(self):
        """Step 2: Explore data"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA EXPLORATION")
        print("=" * 80)
        
        print(f"\nTraining Data:")
        print(f"  - Shape: {self.X_train.shape}")
        print(f"  - Columns: {list(self.X_train.columns)}")
        
        print(f"\nTarget Variable Distribution (Training):")
        print(self.y_train.value_counts())
        
        print(f"\nTarget Variable Distribution (Testing):")
        print(self.y_test.value_counts())
    
    def build_logistic_regression(self):
        """Step 3a: Build Logistic Regression Model"""
        print("\n" + "=" * 80)
        print("STEP 3A: LOGISTIC REGRESSION MODEL")
        print("=" * 80)
        
        print("\nModel Configuration:")
        print("  - Algorithm: Logistic Regression")
        print("  - Type: Linear Binary Classification")
        print("  - Best for: Quick baseline, interpretable results")
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        print("\n  Training Logistic Regression...")
        lr_model.fit(self.X_train, self.y_train)
        print("  ✓ Logistic Regression trained!")
        
        self.models['Logistic Regression'] = lr_model
        
        return lr_model
    
    def build_decision_tree(self):
        """Step 3b: Build Decision Tree Model"""
        print("\n" + "=" * 80)
        print("STEP 3B: DECISION TREE MODEL")
        print("=" * 80)
        
        print("\nModel Configuration:")
        print("  - Algorithm: Decision Tree Classifier")
        print("  - Type: Tree-based Non-linear Classification")
        print("  - Best for: Interpretability, feature importance")
        
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        
        print("\n  Training Decision Tree...")
        dt_model.fit(self.X_train, self.y_train)
        print("  ✓ Decision Tree trained!")
        
        self.models['Decision Tree'] = dt_model
        
        return dt_model
    
    def build_random_forest(self):
        """Step 3c: Build Random Forest Model"""
        print("\n" + "=" * 80)
        print("STEP 3C: RANDOM FOREST MODEL")
        print("=" * 80)
        
        print("\nModel Configuration:")
        print("  - Algorithm: Random Forest Classifier")
        print("  - Type: Ensemble Method (Multiple Decision Trees)")
        print("  - Best for: High accuracy, feature importance")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n  Training Random Forest...")
        rf_model.fit(self.X_train, self.y_train)
        print("  ✓ Random Forest trained!")
        
        self.models['Random Forest'] = rf_model
        
        return rf_model
    
    def build_gradient_boosting(self):
        """Step 3d: Build Gradient Boosting Model"""
        print("\n" + "=" * 80)
        print("STEP 3D: GRADIENT BOOSTING MODEL")
        print("=" * 80)
        
        print("\nModel Configuration:")
        print("  - Algorithm: Gradient Boosting Classifier")
        print("  - Type: Boosting Ensemble Method")
        print("  - Best for: Best accuracy, but slower training")
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print("\n  Training Gradient Boosting...")
        gb_model.fit(self.X_train, self.y_train)
        print("  ✓ Gradient Boosting trained!")
        
        self.models['Gradient Boosting'] = gb_model
        
        return gb_model
    
    def train_all_models(self):
        """Step 4: Train all models"""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "BUILDING AND TRAINING ALL MODELS" + " " * 25 + "║")
        print("╚" + "=" * 78 + "╝")
        
        self.build_logistic_regression()
        self.build_decision_tree()
        self.build_random_forest()
        self.build_gradient_boosting()
        
        print("\n" + "=" * 80)
        print(f"✓ ALL MODELS TRAINED! Total Models: {len(self.models)}")
        print("=" * 80)
        
        return self.models
    
    def make_predictions(self):
        """Step 5: Make predictions on test set"""
        print("\n" + "=" * 80)
        print("STEP 5: MAKING PREDICTIONS")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n  Making predictions with {model_name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  ✓ Predictions made for {model_name}")
        
        print(f"\n✓ Predictions completed for all {len(self.models)} models!")
        
        return self.predictions
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
        }
        return metrics
    
    def evaluate_models(self):
        """Step 6: Evaluate all models"""
        print("\n" + "=" * 80)
        print("STEP 6: EVALUATING MODELS")
        print("=" * 80)
        
        for model_name, predictions in self.predictions.items():
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name}")
            print(f"{'='*80}")
            
            y_pred = predictions['y_pred']
            y_pred_proba = predictions['y_pred_proba']
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            self.performance_metrics[model_name] = metrics
            
            # Display metrics
            print(f"\nPerformance Metrics:")
            print(f"  • Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
            print(f"  • Precision: {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
            print(f"  • Recall:    {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
            print(f"  • F1-Score:  {metrics['F1-Score']:.4f}")
            print(f"  • ROC-AUC:   {metrics['ROC-AUC']:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  [[TN: {cm[0][0]}, FP: {cm[0][1]}]")
            print(f"   [FN: {cm[1][0]}, TP: {cm[1][1]}]]")
            
            # Classification Report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        return self.performance_metrics
    
    def compare_models(self):
        """Step 7: Compare all models"""
        print("\n" + "=" * 80)
        print("STEP 7: MODEL COMPARISON")
        print("=" * 80)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.performance_metrics).T
        comparison_df = comparison_df.round(4)
        
        print("\nPerformance Comparison:")
        print(comparison_df.to_string())
        
        # Find best model
        best_model_f1 = comparison_df['F1-Score'].idxmax()
        best_model_auc = comparison_df['ROC-AUC'].idxmax()
        best_model_acc = comparison_df['Accuracy'].idxmax()
        
        print(f"\n\nBEST MODELS:")
        print(f"  • Highest Accuracy:  {best_model_acc}")
        print(f"  • Highest F1-Score:  {best_model_f1}")
        print(f"  • Highest ROC-AUC:   {best_model_auc}")
        
        return comparison_df
    
    def recommend_model(self):
        """Step 8: Recommend best model"""
        print("\n" + "=" * 80)
        print("STEP 8: MODEL RECOMMENDATION")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(self.performance_metrics).T
        
        # Use F1-score for recommendation (balanced metric)
        best_model = comparison_df['F1-Score'].idxmax()
        best_f1 = comparison_df.loc[best_model, 'F1-Score']
        best_auc = comparison_df.loc[best_model, 'ROC-AUC']
        best_acc = comparison_df.loc[best_model, 'Accuracy']
        
        print(f"\n★ RECOMMENDED MODEL: {best_model}")
        print(f"\nReason:")
        print(f"  • Best balanced performance (F1-Score: {best_f1:.4f})")
        print(f"  • High ROC-AUC: {best_auc:.4f}")
        print(f"  • High Accuracy: {best_acc:.4f}")
        
        print(f"\nWhy this model:")
        if best_model == "Logistic Regression":
            print(f"  • Simple, fast, and interpretable")
            print(f"  • Good for business stakeholders")
        elif best_model == "Decision Tree":
            print(f"  • Highly interpretable")
            print(f"  • Easy to explain churn factors")
        elif best_model == "Random Forest":
            print(f"  • Excellent accuracy")
            print(f"  • Handles feature interactions well")
        elif best_model == "Gradient Boosting":
            print(f"  • Best overall performance")
            print(f"  • Captures complex patterns")
        
        return best_model
    
    def save_models(self, output_dir='.'):
        """Step 9: Save trained models"""
        print("\n" + "=" * 80)
        print("STEP 9: SAVING TRAINED MODELS")
        print("=" * 80)
        
        model_paths = {}
        
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)
            model_paths[model_name] = model_path
            print(f"  ✓ {model_name} saved to {model_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame(self.performance_metrics).T
        metrics_path = f"{output_dir}/model_performance_metrics.csv"
        metrics_df.to_csv(metrics_path)
        print(f"\n  ✓ Performance metrics saved to {metrics_path}")
        
        return model_paths, metrics_path
    
    def execute_full_pipeline(self, output_dir='.'):
        """Execute complete model training and evaluation pipeline"""
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 15 + "TASK 4 & 5: MODEL TRAINING & EVALUATION - FULL PIPELINE" + " " * 7 + "║")
        print("╚" + "=" * 78 + "╝\n")
        
        # Execute all steps
        self.load_data()
        self.explore_data()
        self.train_all_models()
        self.make_predictions()
        self.evaluate_models()
        comparison_df = self.compare_models()
        best_model = self.recommend_model()
        model_paths, metrics_path = self.save_models(output_dir)
        
        print("\n" + "=" * 80)
        print("✓ TASK 4 & 5 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self.models, self.performance_metrics, best_model


if __name__ == "__main__":
    # Initialize and run Tasks 4 & 5
    model_trainer = ModelTrainingEvaluation(
        '/home/claude/X_train.csv',
        '/home/claude/X_test.csv',
        '/home/claude/y_train.csv',
        '/home/claude/y_test.csv'
    )
    models, metrics, best_model = model_trainer.execute_full_pipeline('/home/claude')
