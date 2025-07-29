"""
explain.py
Created by: Addisu Taye
Date Created: July 25, 2025
Purpose: Interpret the best-performing fraud detection model using SHAP (SHapley Additive exPlanations).
         Generates global and local explanations for the Random Forest model.
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set style for plots
plt.style.use('default')
shap.initjs()  # Required for SHAP JavaScript visualizations


BASE_PATH = 'fraud-detection-in-e-commerce-and-banking'  
MODEL_PATH = f'{BASE_PATH}/models/best_model.pkl'
DATA_PATH = f'{BASE_PATH}/data/cleaned/processed_fraud_data.csv'
PLOTS_DIR = f'{BASE_PATH}/plots'

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_model_and_data():
    """Load the trained model and preprocessed data."""
    print("📂 Loading trained model and data...")
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        
        X = df.drop('class', axis=1)
        y = df['class']
        
        print(f"✅ Model loaded from {MODEL_PATH}")
        print(f"✅ Data loaded from {DATA_PATH}")
        print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return model, X, y
    except Exception as e:
        print(f"❌ Error loading model or data: {e}")
        raise

def generate_summary_plot(model, X):
    """Generate and save SHAP summary plot (global feature importance)."""
    print("\n📊 Generating SHAP Summary Plot...")
    
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    
    # For binary classification, shap_values is a list: [negative class, positive class]
    # We want the SHAP values for the positive class (fraud)
    shap_values_positive = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_positive, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Global) - Top Drivers of Fraud")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP Summary Plot (bar) saved to {PLOTS_DIR}/shap_summary_plot.png")
    
    # Summary plot (detailed)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_positive, X, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on Fraud Prediction")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_summary_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP Summary Plot (detailed) saved to {PLOTS_DIR}/shap_summary_detailed.png")

def generate_force_plot(model, X):
    """Generate and save SHAP force plot for a single transaction."""
    print("\n🔍 Generating SHAP Force Plot (Local Explanation)...")
    
    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    
    # Select a transaction (first one for demonstration)
    sample_idx = 0
    shap_values_positive = shap_values[1][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx]
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    # Force plot
    shap.force_plot(
        expected_value,
        shap_values_positive,
        X.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot: Local Explanation for Transaction #{sample_idx}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_force_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP Force Plot saved to {PLOTS_DIR}/shap_force_plot.png")

def generate_waterfall_plot(model, X):
    """Generate and save SHAP waterfall plot for detailed local explanation."""
    print("\n📊 Generating SHAP Waterfall Plot (Local Breakdown)...")
    
    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    
    # Select a transaction
    sample_idx = 0
    shap_values_positive = shap_values[1][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx]
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_positive,
            base_values=expected_value,
            data=X.iloc[sample_idx],
            feature_names=X.columns.tolist()
        ),
        max_display=10,
        show=False
    )

    plt.title(f"SHAP Waterfall Plot: Breakdown for Transaction #{sample_idx}")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_waterfall_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP Waterfall Plot saved to {PLOTS_DIR}/shap_waterfall_plot.png")

def main():
    """Main function to run SHAP analysis."""
    print("🚀 Starting SHAP Analysis for Fraud Detection Model")
    
    # Load model and data
    model, X, y = load_model_and_data()
    
    # Generate global explanation
    generate_summary_plot(model, X)
    
    # Generate local explanations
    generate_force_plot(model, X)
    generate_waterfall_plot(model, X)
    
    # Final message
    print("\n✅ SHAP Analysis Complete!")

    print(f"📊 All plots saved in the '{PLOTS_DIR}' folder.")
    
    print("📌 Use these plots in your final report to explain model decisions.")

if __name__ == "__main__":
    main()