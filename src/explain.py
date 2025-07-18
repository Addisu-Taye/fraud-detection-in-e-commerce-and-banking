"""
explain.py
Created by: Addisu Taye
Date Created: July 30, 2025
Purpose: Interpret model predictions using SHAP values.
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import logging
import numpy as np # Import numpy for NaN handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path='data/cleaned/processed_fraud_data.csv'):
    """
    Load preprocessed data (features X and target y).

    Args:
        path (str): Path to the preprocessed CSV data.

    Returns:
        tuple: (pd.DataFrame X, pd.Series y) if successful, raises error otherwise.
    """
    logging.info(f"Loading data from {path} for SHAP explanation...")
    if not os.path.exists(path):
        logging.error(f"Data file not found at {path}. Please ensure preprocessing is complete.")
        raise FileNotFoundError(f"Data file not found at {path}")

    try:
        df = pd.read_csv(path)
        if df.empty:
            logging.error(f"Loaded data from {path} is empty. Cannot proceed with SHAP explanation.")
            raise ValueError("Input DataFrame is empty.")

        # Ensure 'class' column exists and handle any NaNs in it
        if 'class' not in df.columns:
            logging.error("Target column 'class' not found in the DataFrame. Cannot proceed.")
            raise KeyError("Target column 'class' not found.")
        
        initial_rows = len(df)
        df.dropna(subset=['class'], inplace=True)
        if len(df) < initial_rows:
            logging.warning(f"Dropped {initial_rows - len(df)} rows due to NaN values in the 'class' column during data loading for SHAP.")
        
        if df.empty:
            logging.error("DataFrame became empty after cleaning 'class' column. Cannot proceed with SHAP.")
            raise ValueError("DataFrame is empty after cleaning target column for SHAP.")

        X = df.drop('class', axis=1)
        y = df['class']

        # Ensure X contains only numeric types and handle NaNs, as SHAP typically expects numeric input
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            logging.warning(f"Non-numeric columns found in features (X) for SHAP, attempting to drop: {list(non_numeric_cols)}")
            X = X.select_dtypes(include=[np.number])
        
        initial_X_rows = len(X)
        X.dropna(inplace=True)
        if len(X) < initial_X_rows:
            logging.warning(f"Dropped {initial_X_rows - len(X)} rows from features (X) due to remaining NaN values before SHAP explanation.")
            # Align y if X rows were dropped
            y = y.loc[X.index] 

        if X.empty:
            logging.error("Features (X) DataFrame became empty after cleaning. Cannot proceed with SHAP.")
            raise ValueError("Features DataFrame (X) is empty after cleaning.")

        logging.info(f"Data loaded successfully for SHAP. X shape: {X.shape}")
        return X, y
    except Exception as e:
        logging.critical(f"Failed to load or prepare data for SHAP: {e}")
        raise

def load_model(path='models/random_forest_model.joblib'): # Adjusted default path based on optimized train.py
    """
    Load a trained machine learning model.

    Args:
        path (str): Path to the saved model file (e.g., .joblib).

    Returns:
        object: The loaded scikit-learn model object.
    """
    logging.info(f"Loading model from {path}...")
    if not os.path.exists(path):
        logging.error(f"Model file not found at {path}. Please ensure training is complete and model is saved.")
        raise FileNotFoundError(f"Model file not found at {path}")
    
    try:
        model = joblib.load(path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.critical(f"Failed to load model from {path}: {e}")
        raise

def explain_model(model, X, output_dir='visuals'):
    """
    Generate SHAP explanations and save plots.

    This function attempts to select the appropriate SHAP explainer based on the model type.

    Args:
        model (object): The trained machine learning model.
        X (pd.DataFrame): The feature DataFrame used for explanation.
        output_dir (str): Directory to save SHAP plots.
    """
    logging.info("Generating SHAP explanations...")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    try:
        # Determine appropriate SHAP explainer
        if "XGB" in str(type(model)): # XGBoost
            explainer = shap.TreeExplainer(model)
            logging.info("Using shap.TreeExplainer for XGBoost model.")
        elif "LightGBM" in str(type(model)): # LightGBM
            explainer = shap.TreeExplainer(model)
            logging.info("Using shap.TreeExplainer for LightGBM model.")
        elif "CatBoost" in str(type(model)): # CatBoost
            explainer = shap.TreeExplainer(model)
            logging.info("Using shap.TreeExplainer for CatBoost model.")
        elif "RandomForest" in str(type(model)) or "GradientBoosting" in str(type(model)): # Tree-based ensemble
            explainer = shap.TreeExplainer(model)
            logging.info("Using shap.TreeExplainer for tree-based ensemble model (e.g., RandomForest).")
        elif "LogisticRegression" in str(type(model)) or "LinearSVC" in str(type(model)): # Linear models
            # For linear models, you often want to use LinearExplainer
            # It expects model.coef_ and a background dataset, but can often work directly with the model
            explainer = shap.Explainer(model, X) # KernelExplainer or LinearExplainer are options
            logging.info("Using shap.Explainer (or similar, like LinearExplainer/KernelExplainer) for linear model.")
        else: # Default to KernelExplainer for model-agnostic explanation (slower)
            # KernelExplainer requires a background dataset, usually a sample of the training data
            # For simplicity here, we use X directly, but a small sample is often better for performance
            # You might pass X_train_sampled as a background dataset from your training script
            logging.warning("Model type not explicitly recognized for SHAP. Falling back to shap.KernelExplainer. This might be slow.")
            # Using a small sample of X for the background dataset for performance
            # If X is very large, consider X_train_sampled from your training set
            background_data = shap.sample(X, 100) if len(X) > 100 else X
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, background_data)
        
        # Calculate SHAP values
        logging.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X)

        # Handle multi-output SHAP values (e.g., from multi-class classification, or predict_proba)
        # For binary classification, predict_proba usually gives (N, 2) array, so shap_values is a list of 2 arrays.
        # We typically want the SHAP values for the positive class (class 1).
        if isinstance(shap_values, list):
            logging.info(f"SHAP values are a list (likely multi-class or predict_proba output). Taking values for positive class (index 1).")
            shap_values_for_plot = shap_values[1] # For binary classification, index 1 is usually the positive class
        else:
            shap_values_for_plot = shap_values

        # Summary plot (bar plot of feature importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_for_plot, X, plot_type="bar", show=False)
        summary_plot_path = os.path.join(output_dir, "shap_summary_plot_bar.png")
        plt.savefig(summary_plot_path, bbox_inches='tight') # bbox_inches='tight' prevents labels from being cut off
        plt.close()
        logging.info(f"SHAP summary (bar) plot saved to {summary_plot_path}")

        # Summary plot (dot plot for feature importance and impact)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_for_plot, X, show=False) # Default is dot plot
        dot_plot_path = os.path.join(output_dir, "shap_summary_plot_dot.png")
        plt.savefig(dot_plot_path, bbox_inches='tight')
        plt.close()
        logging.info(f"SHAP summary (dot) plot saved to {dot_plot_path}")

        logging.info("SHAP explanation complete.")
        print(f"\n✅ SHAP explanation complete. Plots saved to {output_dir}/")

    except Exception as e:
        logging.critical(f"An error occurred during SHAP explanation: {e}")
        print(f"\n❌ SHAP explanation FAILED: {e}")

if __name__ == "__main__":
    try:
        X_data, y_data = load_data()
        
        # Specify the path to the model saved by your optimized train.py script
        # Assuming you saved a RandomForest model, for example:
        model_path = "models/random_forest_model.joblib" 
        # Or "models/logistic_regression_model.joblib" if you want to explain that one
        
        trained_model = load_model(model_path)
        
        explain_model(trained_model, X_data) # Pass the loaded model and data
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n❌ Explanation pipeline aborted due to an error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")