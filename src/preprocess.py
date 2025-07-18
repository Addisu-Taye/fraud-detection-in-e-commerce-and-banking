"""
preprocess.py
Created by: Addisu Taye
Date Created: July 18, 2025
Updated: July 30, 2025
Purpose: Preprocess e-commerce and bank transaction data for fraud detection.
         Includes EDA visualizations and summary statistics to CSV files.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Set style for plots
sns.set(style='whitegrid')

def load_datasets(data_dir='data/raw'):
    """Load all datasets."""
    fraud = pd.read_csv(f"{data_dir}/Fraud_Data.csv")
    ip_map = pd.read_csv(f"{data_dir}/IpAddress_to_Country.csv")
    credit = pd.read_csv(f"{data_dir}/creditcard.csv")
    return fraud, ip_map, credit

def convert_ip(ip):
    """Convert IP address string to integer."""
    parts = ip.split('.')
    return int(parts[0])*256**3 + int(parts[1])*256**2 + int(parts[2])*256 + int(parts[3])

def preprocess_fraud_data(fraud_df, ip_df):
    """Preprocess e-commerce fraud data."""
    # Convert timestamps
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

    # Time-based features
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek

    # Convert IP to integer
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(convert_ip)
    ip_df['lower'] = ip_df['lower_bound_ip_address'].apply(convert_ip)
    ip_df['upper'] = ip_df['upper_bound_ip_address'].apply(convert_ip)

    # Map country based on IP range
    fraud_df['country'] = None
    for _, row in ip_df.iterrows():
        mask = (fraud_df['ip_int'] >= row['lower']) & (fraud_df['ip_int'] <= row['upper'])
        fraud_df.loc[mask, 'country'] = row['country']

    # Drop unneeded cols
    fraud_df.drop(['lower', 'upper', 'ip_int'], axis=1, inplace=True)
    fraud_df.dropna(subset=['country'], inplace=True)

    return fraud_df

def plot_eda(fraud_df):
    """Generate and save EDA plots."""
    os.makedirs('plots', exist_ok=True)

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=fraud_df)
    plt.title('Class Distribution (Before SMOTE)')
    plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
    plt.ylabel('Count')
    plt.savefig('plots/class_distribution.png')
    plt.close()

    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_df['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('plots/age_distribution.png')
    plt.close()

    # Hour of Day
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour_of_day', data=fraud_df)
    plt.title('Transaction Count by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.savefig('plots/hour_of_day_distribution.png')
    plt.close()

    # Day of Week
    plt.figure(figsize=(10, 6))
    sns.countplot(x='day_of_week', data=fraud_df)
    plt.title('Transaction Count by Day of Week')
    plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
    plt.ylabel('Count')
    plt.savefig('plots/day_of_week_distribution.png')
    plt.close()

    # Feature correlation
    plt.figure(figsize=(12, 10))
    corr = fraud_df[['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'class']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('plots/feature_correlation.png')
    plt.close()

def save_data_summary(fraud_df):
    """Save summary statistics to CSV files."""
    os.makedirs('data_summary', exist_ok=True)

    # Class distribution
    class_summary = fraud_df['class'].value_counts().reset_index()
    class_summary.columns = ['class', 'count']
    class_summary['percentage'] = (class_summary['count'] / class_summary['count'].sum()) * 100
    class_summary.to_csv('data_summary/fraud_class_summary.csv', index=False)

    # Country distribution
    country_summary = fraud_df['country'].value_counts().reset_index()
    country_summary.columns = ['country', 'count']
    country_summary.to_csv('data_summary/country_distribution.csv', index=False)

    # Source distribution
    source_summary = fraud_df['source'].value_counts().reset_index()
    source_summary.columns = ['source', 'count']
    source_summary.to_csv('data_summary/source_distribution.csv', index=False)

    # Browser distribution
    browser_summary = fraud_df['browser'].value_counts().reset_index()
    browser_summary.columns = ['browser', 'count']
    browser_summary.to_csv('data_summary/browser_distribution.csv', index=False)

    # Correlation matrix
    corr = fraud_df[['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'class']].corr()
    corr.to_csv('data_summary/feature_correlation.csv')

def handle_class_imbalance(X, y):
    """Apply SMOTE to balance classes."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

if __name__ == "__main__":
    # Load data
    fraud_data, ip_data, credit_data = load_datasets()

    # Process e-commerce fraud data
    fraud_processed = preprocess_fraud_data(fraud_data.copy(), ip_data.copy())

    # Generate EDA plots
    plot_eda(fraud_processed)

    # Save summary statistics
    save_data_summary(fraud_processed)

    # One-hot encode categoricals
    encoder = OneHotEncoder(drop='first')
    cat_features = ['source', 'browser', 'sex', 'country']
    encoded = encoder.fit_transform(fraud_processed[cat_features]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_features), index=fraud_processed.index)
    fraud_encoded = pd.concat([fraud_processed.drop(cat_features, axis=1), encoded_df], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    num_features = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week']
    fraud_encoded[num_features] = scaler.fit_transform(fraud_encoded[num_features])

    # Handle class imbalance
    X = fraud_encoded.drop('class', axis=1)
    y = fraud_encoded['class']
    X_balanced, y_balanced = handle_class_imbalance(X, y)

    # Recombine
    fraud_balanced = pd.concat([X_balanced, y_balanced], axis=1)

    # Save processed data
    fraud_balanced.to_csv("data/cleaned/processed_fraud_data.csv", index=False)
    credit_data.to_csv("data/cleaned/processed_creditcard.csv", index=False)

    print("✅ Preprocessing complete. Files saved to:", "data/cleaned")
    print("📈 Plots saved to:", "plots/")
    print("📊 Summary statistics saved to:", "data_summary/")