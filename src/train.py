"""
train.py
Created by: Addisu Taye
Date Created: July 18, 2025
Purpose: Train and evaluate machine learning models for fraud detection.
"""

import pandas as pd
import numpy as np # Added for potential NaN handling if needed
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, make_scorer, f1_score
import joblib # For saving/loading models
import warnings

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For issues like division by zero in metrics if a class is empty

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path='data/cleaned/processed_fraud_data.csv'):
    """
    Load preprocessed data and split into training and testing sets.
    
    Args:
        path (str): Path to the preprocessed CSV data.

    Returns:
        tuple: X_train, X_test, y_train, y_test DataFrames/Series.
    """
    logging.info(f"Loading data from {path}...")
    if not os.path.exists(path):
        logging.error(f"Data file not found at {path}. Please ensure preprocessing is complete.")
        raise FileNotFoundError(f"Data file not found at {path}")

    try:
        df = pd.read_csv(path)
        if df.empty:
            logging.error(f"Loaded data from {path} is empty. Cannot proceed with training.")
            raise ValueError("Input DataFrame is empty.")

        # Ensure 'class' column exists
        if 'class' not in df.columns:
            logging.error("Target column 'class' not found in the DataFrame. Cannot proceed.")
            raise KeyError("Target column 'class' not found.")
        
        # Drop rows where the target 'class' is NaN
        initial_rows = len(df)
        df.dropna(subset=['class'], inplace=True)
        if len(df) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df)} rows due to NaN values in the 'class' column.")
        
        if df.empty:
            logging.error("DataFrame became empty after dropping rows with NaN in 'class'. Cannot proceed.")
            raise ValueError("DataFrame is empty after cleaning target column.")

        X = df.drop('class', axis=1)
        y = df['class']

        # Ensure X contains only numeric types and no NaNs before splitting and training
        # This is a good safety net if preprocessing missed some NaNs or non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            logging.warning(f"Non-numeric columns found in features (X) which will be dropped: {list(non_numeric_cols)}")
            X = X.select_dtypes(include=[np.number])
        
        # Drop rows with NaNs in X if any remain. Model training typically cannot handle NaNs.
        initial_X_rows = len(X)
        X.dropna(inplace=True)
        if len(X) < initial_X_rows:
            logging.warning(f"Dropped {initial_X_rows - len(X)} rows from features (X) due to remaining NaN values. Target 'y' will be aligned.")
            y = y.loc[X.index] # Align y to the new index of X

        if X.empty or y.empty:
            logging.error("Features (X) or target (y) DataFrame/Series became empty after NaN cleaning. Cannot proceed.")
            raise ValueError("Features or target is empty after final cleaning.")

        # Ensure there are at least two classes for stratification
        if y.nunique() < 2:
            logging.error(f"Only {y.nunique()} unique class(es) found in target variable. Stratified split requires at least 2 classes.")
            raise ValueError("Insufficient number of classes for stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        logging.info(f"Data loaded and split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.critical(f"Failed to load or split data: {e}")
        raise # Re-raise the exception after logging

def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, results_dir='reports', models_dir='models'):
    """
    Train a single model, evaluate it, and save results and the model.

    Args:
        name (str): Name of the model.
        model (estimator): Scikit-learn estimator object.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        results_dir (str): Directory to save evaluation reports.
        models_dir (str): Directory to save trained models.

    Returns:
        dict: Dictionary containing evaluation metrics for the model.
    """
    logging.info(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        logging.info(f"{name} training complete.")
    except Exception as e:
        logging.error(f"Error during {name} training: {e}")
        return None # Indicate failure

    # Save the trained model
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.joblib")
    try:
        joblib.dump(model, model_filename)
        logging.info(f"Saved {name} model to {model_filename}")
    except Exception as e:
        logging.warning(f"Could not save {name} model: {e}")

    logging.info(f"Evaluating {name}...")
    try:
        y_pred = model.predict(X_test)
        # Handle cases where predict_proba is not available (e.g., some classifiers)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = np.nan # Or choose a default value, or skip ROC/PR AUC

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        f1 = report_dict['1']['f1-score'] if '1' in report_dict and 'f1-score' in report_dict['1'] else np.nan
        roc_auc = roc_auc_score(y_test, y_proba) if not np.isnan(y_proba).all() else np.nan

        pr_auc = np.nan
        if not np.isnan(y_proba).all():
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)

        result = {
            "Model": name,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "AUC-PR": pr_auc
        }

        # Print detailed report to console
        print(f"\n{name} Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc if not np.isnan(roc_auc) else 'N/A'}")
        print(f"AUC-PR: {pr_auc if not np.isnan(pr_auc) else 'N/A'}")
        
        logging.info(f"{name} evaluation complete.")
        return result
    except Exception as e:
        logging.error(f"Error during {name} evaluation: {e}")
        return None # Indicate failure

def run_training_pipeline():
    """Main function to orchestrate data loading, training, and evaluation."""
    try:
        X_train, X_test, y_train, y_test = load_data()
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n❌ Training pipeline aborted due to data loading/splitting error: {e}")
        return # Exit if data loading fails

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000), # Add solver and max_iter
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1) # Add n_estimators, n_jobs
    }

    results = []
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True) # Create reports directory

    for name, model in models.items():
        model_results = train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, results_dir=reports_dir)
        if model_results: # Only add if evaluation was successful
            results.append(model_results)

    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(reports_dir, "model_comparison.csv")
        try:
            results_df.to_csv(results_path, index=False)
            logging.info(f"Model comparison results saved to {results_path}")
            print(f"\n✅ Model training complete. Results saved to {results_path}")
        except Exception as e:
            logging.error(f"Error saving model comparison results: {e}")
            print(f"\n❌ Model training complete, but failed to save results: {e}")
    else:
        logging.warning("No models were successfully trained and evaluated.")
        print("\n❌ No models were successfully trained and evaluated.")

if __name__ == "__main__":
    run_training_pipeline()