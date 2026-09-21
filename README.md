# saiket-technology-machine-learning-internship
Machine Learning internship projects, assignments, and implementations completed during my internship at Saiket Technology System.
# Customer Churn Analysis and Prediction
## Saiket Systems Machine Learning Internship

This project implements a complete machine learning pipeline to analyze and predict customer churn in a telecommunications company.

---

## 📋 Project Overview

**Project Title:** Customer Churn Analysis and Prediction

**Objective:** Analyze customer churn in a telecommunications company and develop predictive models to identify at-risk customers. Provide actionable insights and recommendations to reduce churn and improve customer retention.

**Dataset:** Telco Customer Churn Dataset
- **Total Records:** 7,044 customers
- **Total Features:** 20 attributes
- **Target Variable:** Churn (Yes/No)

---

## 🎯 Tasks Implemented (4 out of 6)

### Task 1: Data Preparation ✓
**Description:** Load and preprocess the dataset, addressing missing values and encoding categorical variables.

**Key Steps:**
1. Load the dataset
2. Explore data structure and statistics
3. Check for missing values
4. Handle missing values (remove or impute)
5. Identify numerical and categorical columns
6. Encode categorical variables using LabelEncoder
7. Generate final statistics and save prepared data

**File:** `task_1_data_preparation.py`
**Output:** `task1_prepared_data.csv`

**Skills Demonstrated:**
- Data preprocessing techniques
- Handling missing values
- Categorical variable encoding

---

### Task 2: Split Data for Training and Testing ✓
**Description:** Divide the data into training (80%) and testing (20%) sets for model training and evaluation, ensuring representative split using stratification.

**Key Steps:**
1. Load prepared data
2. Analyze target variable distribution
3. Prepare features (X) and target (y)
4. Split data using stratified train_test_split (80/20)
5. Validate stratification effectiveness
6. Display sample data from splits
7. Save split datasets

**File:** `task_2_data_splitting.py`
**Outputs:**
- `X_train.csv` (Training features)
- `X_test.csv` (Testing features)
- `y_train.csv` (Training target)
- `y_test.csv` (Testing target)

**Skills Demonstrated:**
- Data splitting methodologies
- Understanding of training/testing dataset requirements
- Stratification for class balance

---

### Task 3: Feature Selection ✓
**Description:** Identify and select relevant features (attributes) influencing churn prediction using statistical methods.

**Key Steps:**
1. Load split data
2. Explore feature statistics
3. Scale features using StandardScaler
4. Calculate feature importance using F-score
5. Calculate mutual information scores
6. Select top features based on importance
7. Create feature-reduced datasets
8. Save selected features and importance metrics

**File:** `task_3_feature_selection.py`
**Outputs:**
- `X_train_selected.csv` (Selected training features)
- `X_test_selected.csv` (Selected testing features)
- `X_train_selected_scaled.csv` (Scaled selected features)
- `X_test_selected_scaled.csv` (Scaled selected features)
- `selected_features.txt` (List of selected features)
- `feature_importance.csv` (Feature importance scores)

**Top Features Identified:**
- tenure (Customer tenure in months)
- MonthlyCharges (Monthly service charges)
- Contract (Contract type)
- TotalCharges (Total charges to date)
- InternetService (Internet service type)

**Skills Demonstrated:**
- Feature relevance analysis
- Domain knowledge for identifying influential attributes
- Statistical feature selection

---

### Task 4 & 5: Model Selection, Training and Evaluation ✓
**Description:** Select suitable binary classification algorithms, train models, and evaluate performance on testing dataset.

**Key Steps:**

#### Task 4: Model Selection
1. Compare 4 binary classification algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier

#### Task 5: Training & Evaluation
1. Train all models on training dataset
2. Make predictions on testing dataset
3. Calculate evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
4. Generate confusion matrices
5. Compare model performance
6. Recommend best model
7. Save trained models and metrics

**File:** `task_4_5_model_training_evaluation.py`
**Outputs:**
- `logistic_regression_model.pkl` (Trained model)
- `decision_tree_model.pkl` (Trained model)
- `random_forest_model.pkl` (Trained model)
- `gradient_boosting_model.pkl` (Trained model)
- `model_performance_metrics.csv` (Performance comparison)

**Performance Metrics Calculated:**
- **Accuracy:** Percentage of correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve

**Recommended Model:** Selected based on F1-Score (balanced metric)

**Skills Demonstrated:**
- Understanding of binary classification algorithms
- Model selection based on dataset characteristics
- Model training and hyperparameter tuning
- Evaluation metrics understanding

---

## 🚀 Setup and Installation Guide

### STEP 1: Prerequisites
Make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Git

**Check Python version:**
```bash
python --version
```

### STEP 2: Create Project Directory

```bash
# Create a new directory for your project
mkdir telco-churn-prediction
cd telco-churn-prediction
```

### STEP 3: Create Virtual Environment

Virtual environment isolates project dependencies.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### STEP 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Verify installation:**
```bash
pip list
```

### STEP 5: Organize Project Files

Create the following directory structure:

```
telco-churn-prediction/
├── data/
│   └── Telco_Customer_Churn_Dataset.csv
├── outputs/
├── src/
│   ├── task_1_data_preparation.py
│   ├── task_2_data_splitting.py
│   ├── task_3_feature_selection.py
│   ├── task_4_5_model_training_evaluation.py
│   └── main_complete_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Running the Pipeline

### Method 1: Run Complete Pipeline (Recommended)

```bash
# Navigate to project directory
cd telco-churn-prediction

# Ensure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Run complete pipeline
python src/main_complete_pipeline.py
```

**Note:** This script will run all 4 tasks sequentially with pauses between tasks.

### Method 2: Run Individual Tasks

**Task 1 - Data Preparation:**
```bash
python src/task_1_data_preparation.py
```

**Task 2 - Data Splitting:**
```bash
python src/task_2_data_splitting.py
```

**Task 3 - Feature Selection:**
```bash
python src/task_3_feature_selection.py
```

**Task 4 & 5 - Model Training & Evaluation:**
```bash
python src/task_4_5_model_training_evaluation.py
```

---

## 📁 Output Files Generated

### From Task 1:
- `task1_prepared_data.csv` - Prepared and encoded dataset

### From Task 2:
- `X_train.csv` - Training features
- `X_test.csv` - Testing features
- `y_train.csv` - Training target
- `y_test.csv` - Testing target

### From Task 3:
- `X_train_selected.csv` - Selected training features
- `X_test_selected.csv` - Selected testing features
- `X_train_selected_scaled.csv` - Scaled selected training features
- `X_test_selected_scaled.csv` - Scaled selected testing features
- `selected_features.txt` - List of selected features
- `feature_importance.csv` - Feature importance scores

### From Task 4 & 5:
- `logistic_regression_model.pkl` - Saved model
- `decision_tree_model.pkl` - Saved model
- `random_forest_model.pkl` - Saved model
- `gradient_boosting_model.pkl` - Saved model
- `model_performance_metrics.csv` - Model comparison metrics

---

## 📤 GitHub Setup and Push Instructions

### STEP 1: Create GitHub Account and Repository

1. Go to https://github.com
2. Click "Sign up" and create a new account (if you don't have one)
3. After logging in, click the "+" icon and select "New repository"
4. Enter repository name: `telco-churn-prediction`
5. Add description: "Customer Churn Analysis and Prediction"
6. Choose "Public" or "Private"
7. **DO NOT** initialize with README (we already have one)
8. Click "Create repository"

### STEP 2: Initialize Git in Local Project

```bash
# Navigate to project directory
cd telco-churn-prediction

# Initialize git
git init

# Configure git (one time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### STEP 3: Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files from version control:

```bash
# Create .gitignore file
echo "# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files (optional - only if large)
data/

# Output files (optional)
outputs/
*.csv
*.pkl" > .gitignore
```

### STEP 4: Add Files to Git

```bash
# Add all files to staging area
git add .

# Verify files to be committed
git status

# You should see files like:
# - README.md (green)
# - requirements.txt (green)
# - src/task_1_data_preparation.py (green)
# - src/task_2_data_splitting.py (green)
# - etc.
```

### STEP 5: Create Initial Commit

```bash
# Create commit with descriptive message
git commit -m "Initial commit: Complete ML pipeline for churn prediction"

# Verify commit
git log --oneline
```

### STEP 6: Connect to Remote Repository

```bash
# Add remote repository (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/telco-churn-prediction.git

# Verify connection
git remote -v
```

### STEP 7: Push to GitHub

**For newer Git versions (main branch):**
```bash
# Rename branch to main (if on master)
git branch -M main

# Push to GitHub
git push -u origin main
```

**For older Git versions (master branch):**
```bash
git push -u origin master
```

### STEP 8: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/telco-churn-prediction`
2. Check if all files are visible
3. Verify file structure

---

## 📝 Committing Changes Later

After making changes or running the pipeline:

```bash
# Check what changed
git status

# Add specific files or all changes
git add .

# Create commit
git commit -m "Descriptive message about changes"

# Push to GitHub
git push origin main
```

### Example Commits:
```bash
git commit -m "Run Task 1: Data preparation completed"
git commit -m "Run Task 2: Data splitting completed"
git commit -m "Run Task 3: Feature selection completed"
git commit -m "Run Task 4 & 5: Model training completed"
```

---

## 🔑 Key Commands Reference

### Git Commands:
```bash
# Initialize repository
git init

# Check status
git status

# Add files
git add .                          # Add all files
git add filename.py                # Add specific file

# Create commit
git commit -m "message"

# Push to GitHub
git push origin main

# View commit history
git log --oneline

# Create new branch
git branch branch-name

# Switch branch
git checkout branch-name

# Create and switch to new branch
git checkout -b branch-name
```

### Python Commands:
```bash
# Activate virtual environment
source venv/bin/activate           # macOS/Linux
venv\Scripts\activate              # Windows

# Deactivate virtual environment
deactivate

# Install requirements
pip install -r requirements.txt

# Run script
python filename.py
```

---

## 🎓 Key Concepts

### Data Preparation
- **Missing Values:** Handled by dropping or imputation
- **Encoding:** Converting categorical variables to numerical using LabelEncoder

### Data Splitting
- **Train/Test Split:** 80% training, 20% testing
- **Stratification:** Maintains class distribution in both sets

### Feature Selection
- **F-Score:** Measures feature importance for classification
- **Mutual Information:** Measures dependency between feature and target

### Model Evaluation
- **Accuracy:** Overall correctness
- **Precision:** True positive rate among predicted positives
- **Recall:** True positive rate among actual positives
- **F1-Score:** Harmonic mean balancing precision and recall
- **ROC-AUC:** Area under curve for threshold comparison

---

## 💡 Tips for Success

1. **Always activate virtual environment** before running code
2. **Keep dataset in version control carefully** (use .gitignore)
3. **Commit frequently** with descriptive messages
4. **Document changes** in commit messages
5. **Test before pushing** to ensure code works
6. **Use branches** for experimentation

---

## 📞 Contact & Support

**For Saiket Systems Support:**
- Email: support@saiket.in
- Website: www.saiket.in
- LinkedIn: @saiketsystems
- Instagram: @saiket_systems

---

## 📄 License

This project is part of the Saiket Systems Machine Learning Internship Program.

---

## ✅ Checklist Before Submission

- [ ] All 4 tasks completed
- [ ] Code runs without errors
- [ ] All output files generated
- [ ] GitHub repository created
- [ ] Files pushed to GitHub
- [ ] README.md updated with your info
- [ ] Commit messages are descriptive
- [ ] Virtual environment used for development
- [ ] requirements.txt updated with dependencies

---

## 🎉 Completion

When you complete this pipeline:

1. **Update LinkedIn Profile:**
   - Add Saiket Systems internship completion
   - Tag @Saiket Systems
   - Use hashtags: #SaiKetSystemsJourney #FutureWithSaiKet

2. **Create Demo Video:**
   - Show the pipeline execution
   - Highlight key results
   - Post on LinkedIn with project tags

3. **Document Learnings:**
   - Key takeaways
   - Challenges faced
   - Solutions implemented

---

**Good Luck! 🚀**
