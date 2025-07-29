"""
app.py
Created by: Addisu Taye
Date Created: July 26, 2025
Purpose: Backend Flask API for the Fraud Detection Project.
         Provides endpoints to predict fraud and get model explanations (SHAP).
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import shap
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Define paths - Updated to match your Google Drive folder
BASE_PATH = 'FraudDetectionProject'
MODEL_PATH = f'{BASE_PATH}/models/best_model.pkl'
DATA_PATH = f'{BASE_PATH}/data/cleaned/processed_fraud_data.csv'

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load a sample of the dataset to get feature names (required for prediction)
try:
    df = pd.read_csv(DATA_PATH)
    X = df.drop('class', axis=1)
    feature_names = X.columns.tolist()
    print(f"✅ Data loaded from {DATA_PATH} for feature names")
except Exception as e:
    print(f"❌ Error loading  {e}")
    feature_names = []

# Initialize SHAP explainer (lazily to save memory)
explainer = None

def get_shap_explainer():
    """Lazy load SHAP explainer."""
    global explainer
    if explainer is None and model is not None:
        explainer = shap.TreeExplainer(model)
    return explainer

@app.route('/')
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Welcome to the Fraud Detection API",
        "endpoints": {
            "/predict": "POST - Predict fraud probability for a transaction",
            "/explain": "POST - Get SHAP explanation for a prediction"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a single transaction."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Reindex to match training data (add missing columns, remove extra ones)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()  # [P(0), P(1)]

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability[1]),
            "legitimate_probability": float(probability[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/explain', methods=['POST'])
def explain():
    """Get SHAP explanation for a prediction."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Get SHAP explainer
        explainer = get_shap_explainer()
        if explainer is None:
            return jsonify({"error": "SHAP explainer could not be initialized"}), 500

        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df, check_additivity=False)
        expected_value = explainer.expected_value

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Focus on positive class
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        # Convert to list for JSON serialization
        shap_values_list = shap_values[0].tolist()  # SHAP values for each feature
        base_value = float(expected_value)

        # Create feature importance list
        feature_importance = [
            {"feature": feature, "shap_value": float(shap_val)}
            for feature, shap_val in zip(feature_names, shap_values_list)
        ]

        # Sort by absolute SHAP value
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return jsonify({
            "base_value": base_value,
            "shap_values": shap_values_list,
            "feature_importance": feature_importance,
            "model_prediction": int(np.argmax(model.predict_proba(input_df)[0])),
            "fraud_probability": float(model.predict_proba(input_df)[0][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": len(feature_names) > 0
    })

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)