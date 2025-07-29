# 🧠 Improved Fraud Detection for E-Commerce and Bank Transactions  
**A Data Science Project at Adey Innovations Inc.**  
*Submitted by: Addisu Taye*  
*Date: July 27, 2025*

---

## 📌 Overview

This project aims to improve fraud detection in **e-commerce** and **banking** by building accurate, explainable machine learning models. As a data scientist at **Adey Innovations Inc.**, I addressed class imbalance, engineered behavioral and temporal features, and trained models to minimize financial loss while preserving user experience.

The solution includes:
- Robust preprocessing and geolocation mapping
- Feature engineering (e.g., `time_since_signup`, IP-to-country)
- Handling extreme class imbalance with SMOTE
- Training and comparing **Logistic Regression** and **Random Forest**
- Model evaluation using **F1-Score, AUC-PR, and AUC-ROC**
- Model interpretation with **SHAP** for business transparency

---

## 📁 Project Structure
```
fraud-detection-ecom-banking/
├── data/
│   ├── raw/                # Original datasets
│   └── cleaned/            # Preprocessed datasets
├── src/
│   ├── preprocess.py       # Data cleaning & feature engineering
│   ├── train.py            # Model training
│   └── explain.py          # SHAP-based interpretation
├── plots/                  # EDA and model visualizations
├── data_summary/           # Summary statistics (CSV)
├── models/                 # Trained model binaries
├── reports/                # Interim and final reports
├── notebooks/              # Jupyter/Colab notebooks
└── README.md
```

---

## 🛠️ Data Analysis & Preprocessing

### ✅ Key Steps
- **Data Cleaning**: Removed duplicates, converted timestamps, handled missing values.
- **IP Geolocation Mapping**: Converted IP addresses to integers and merged with `IpAddress_to_Country.csv` to add `country` feature.
- **Feature Engineering**:
  - `time_since_signup` (hours between signup and purchase)
  - `hour_of_day` and `day_of_week` from `purchase_time`
  - One-Hot Encoding for `source`, `browser`, `sex`, `country`
- **Class Imbalance**: Applied **SMOTE** to balance the minority fraud class in training data.
- **Scaling**: Used `StandardScaler` on numerical features.

### 📊 EDA Insights
- Severe class imbalance: <1% fraud in both datasets.
- Fraud spikes during late-night hours (`hour_of_day`).
- High `purchase_value` and certain countries show higher fraud rates.
- Users with very short `time_since_signup` are more likely to be fraudulent.

---

## 🤖 Model Building & Training

### 📦 Models Trained
| Model                | Purpose                   |
|---------------------|---------------------------|
| **Logistic Regression** | Interpretable baseline     |
| **Random Forest**       | High-performance ensemble |

### 📈 Evaluation Metrics (Imbalanced Data)
- **F1-Score**: Balances precision and recall
- **AUC-PR**: More reliable than ROC-AUC for rare events
- **Confusion Matrix**: Analyze false positives/negatives

### 📊 Performance Results

| Dataset        | Model              | F1-Score | AUC-ROC | AUC-PR |
|----------------|--------------------|----------|---------|--------|
| **E-commerce** | Logistic Regression| 0.69     | 0.77    | 0.78   |
|                | **Random Forest**  | **0.97** | **0.98**| **0.99**|
| **Credit Card**| Logistic Regression| 0.95     | 0.99    | 0.99   |
|                | **Random Forest**  | **1.00** | **1.00**| **1.00**|

### 🏆 Final Model: **Random Forest**
**Justification**:
- Significantly outperforms Logistic Regression on all metrics.
- High **AUC-PR (0.99)** ensures strong performance on the minority fraud class.
- Handles non-linear patterns in user behavior and transaction timing.
- Will be interpreted using **SHAP** for transparency (Task 3).

---

## 🔍 Model Explainability

- Use **SHAP (SHapley Additive exPlanations)** to interpret Random Forest predictions.
- Generate:
  - **Summary Plot**: Global feature importance
  - **Force Plot**: Local explanation for individual transactions
- Extract business insights:
  - Which features drive fraud predictions?
  - How can stakeholders act on these insights?

---

## 🚀 How to Run

1. **Clone the repo**:
```bash
git clone https://github.com/Addisu-Taye/fraud-detection-in-e-commerce-and-banking.git
cd fraud-detection-in-e-commerce-and-banking
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run preprocessing**:
```bash
python src/preprocess.py
```

4. **Train models**:
```bash
python src/train.py
```

5. **Generate explanations (upcoming)**:
```bash
python src/explain.py
```

---

## 📦 `requirements.txt`
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
jupyter>=1.0.0
```

Install with:
```bash
pip install -r requirements.txt
```
