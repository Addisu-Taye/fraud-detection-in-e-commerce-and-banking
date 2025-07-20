# Fraud Detection Data Preprocessing and EDA Pipeline

## Overview

This repository contains `preprocess.py`, a comprehensive Python script designed to prepare raw transaction data for fraud detection machine learning models. It handles two distinct datasets: e-commerce transaction data (`Fraud_Data.csv`) and credit card transaction data (`creditcard.csv`). The script performs robust data cleaning, feature engineering tailored to each dataset, exploratory data analysis (EDA), and addresses class imbalance, producing clean, structured datasets ready for model training.

## Features

The `preprocess.py` script includes the following key functionalities:

* **Data Loading**: Loads `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` from the `data/raw` directory.
* **E-commerce Data Preprocessing (`Fraud_Data.csv`)**:
    * **Timestamp Conversion & Cleaning**: Converts `signup_time` and `purchase_time` to datetime objects, handling errors and missing values.
    * **IP Address-to-Country Mapping**: Converts IP addresses to integers and uses `IpAddress_to_Country.csv` to map transactions to their respective countries, providing geographical context.
    * **Time-based Feature Engineering**: Creates new features like `time_since_signup` (in hours), `hour_of_day`, and `day_of_week` from timestamps to capture temporal patterns.
    * **Categorical Feature Encoding**: Applies One-Hot Encoding (`source`, `browser`, `sex`, `country`) to transform nominal features into a numerical format.
    * **Numerical Feature Scaling**: Standardizes numerical features (`purchase_value`, `age`, `time_since_signup`, `hour_of_day`, `day_of_week`) using `StandardScaler`.
* **Credit Card Data Preprocessing (`creditcard.csv`)**:
    * **Numerical Feature Scaling**: Scales `Time` and `Amount` features using `StandardScaler` to bring them to a comparable range with the anonymized `V` features.
* **Class Imbalance Handling**: Applies the **Synthetic Minority Over-sampling Technique (SMOTE)** to both e-commerce and credit card datasets to balance the highly skewed class distributions (fraud vs. non-fraud), crucial for effective model training.
* **Exploratory Data Analysis (EDA)**:
    * Generates various plots (e.g., class distributions, age distribution, time patterns, correlation matrices) for both datasets.
    * Saves descriptive statistics and class summaries to CSV files.
* **Output Management**: Saves processed data, plots, and summary statistics to designated output directories.
* **Comprehensive Logging**: Provides detailed logs for each step of the preprocessing pipeline, aiding in monitoring and debugging.

## Datasets

Ensure the following raw data files are placed in the `data/raw` directory:

* `Fraud_Data.csv`: E-commerce transaction data, including user and transaction details, and a fraud label (`class`).
* `IpAddress_to_Country.csv`: A lookup table for mapping IP address ranges to countries.
* `creditcard.csv`: Bank transaction data with anonymized features (`V1-V28`), transaction time, amount, and a fraud label (`Class`).

## Installation

1.  **Clone the repository** (or download `preprocess.py`):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python libraries**:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn python-docx
    ```

## Usage

1.  Place the raw datasets (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`) into a directory named `data/raw` in the same location as `preprocess.py`.
    ```
    .
    ├── data/
    │   └── raw/
    │       ├── Fraud_Data.csv
    │       ├── IpAddress_to_Country.csv
    │       └── creditcard.csv
    └── preprocess.py
    ```
2.  Run the preprocessing script from your terminal:
    ```bash
    python preprocess.py
    ```

## Output

Upon successful execution, the script will create the following directories and files:

* `data/cleaned/`: Contains the processed datasets.
    * `processed_fraud_data.csv`: Cleaned, feature-engineered, scaled, and balanced e-commerce fraud data.
    * `processed_creditcard.csv`: Cleaned, scaled, and balanced credit card transaction data.
* `plots/`: Contains various EDA visualizations in PNG format.
    * `fraud_data_class_distribution.png`
    * `fraud_data_age_distribution.png`
    * `fraud_data_hour_of_day_distribution.png`
    * `fraud_data_day_of_week_distribution.png`
    * `fraud_data_feature_correlation.png`
    * `creditcard_data_class_distribution.png`
    * `creditcard_data_feature_correlation.png`
    * (and potentially other plots if added in the future)
* `data_summary/`: Contains summary statistics and class distribution reports in CSV format.
    * `fraud_data_descriptive_statistics.csv`
    * `fraud_data_class_summary.csv`
    * `fraud_data_country_distribution.csv`
    * `fraud_data_source_distribution.csv`
    * `fraud_data_browser_distribution.csv`
    * `creditcard_data_descriptive_statistics.csv`
    * `creditcard_data_class_summary.csv`
    * (and potentially other summaries)

## Project Structure