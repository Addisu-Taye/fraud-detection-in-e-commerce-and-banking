"""
preprocess.py
Created by: Addisu Taye
Date Created: July 18, 2025
Updated: July 19, 2025 (Fixing 'country_y' column reference after merge_asof)
Updated: July 20, 2025 (Added preprocessing for creditcard.csv)
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set style for plots
sns.set(style='whitegrid')

def load_datasets(data_dir='data/raw'):
    """Load all datasets."""
    logging.info(f"Loading datasets from {data_dir}...")
    try:
        fraud = pd.read_csv(f"{data_dir}/Fraud_Data.csv")
        ip_map = pd.read_csv(f"{data_dir}/IpAddress_to_Country.csv")
        credit = pd.read_csv(f"{data_dir}/creditcard.csv")
        logging.info("Datasets loaded successfully.")
        return fraud, ip_map, credit
    except FileNotFoundError as e:
        logging.error(f"Error loading datasets: {e}. Make sure the data directory and files exist.")
        raise

def convert_ip(ip):
    """Convert IP address string (dot-decimal or already int/float) to integer."""
    if pd.isna(ip) or not isinstance(ip, (str, float, int)):
        logging.debug(f"Input IP is NaN/None or unexpected type: {ip}. Returning NaN.")
        return np.nan
    try:
        if isinstance(ip, (float, int)):
            return int(ip)
        
        parts = str(ip).strip().split('.')
        if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            if str(ip).isdigit():
                return int(ip)
            logging.debug(f"Malformed or out-of-range IPv4 address encountered: '{ip}'. Returning NaN.")
            return np.nan
        return int(parts[0])*256**3 + int(parts[1])*256**2 + int(parts[2])*256 + int(parts[3])
    except ValueError as e:
        logging.debug(f"ValueError during IP conversion for '{ip}': {e}. Returning NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Unexpected error converting IP address '{ip}': {e}. Returning NaN.")
        return np.nan

def preprocess_fraud_data(fraud_df, ip_df):
    """Preprocess e-commerce fraud data."""
    logging.info("Starting preprocessing of e-commerce fraud data.")
    logging.debug(f"Initial fraud_df shape: {fraud_df.shape}")

    # Convert timestamps
    logging.debug("Converting timestamp columns...")
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce') 
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')
    
    initial_rows_ts_conv = len(fraud_df)
    fraud_df.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
    if len(fraud_df) < initial_rows_ts_conv:
        logging.warning(f"Dropped {initial_rows_ts_conv - len(fraud_df)} rows due to invalid signup_time or purchase_time. Current shape: {fraud_df.shape}")
    
    logging.debug("Timestamp columns converted and NaTs handled.")
    
    if fraud_df.empty:
        logging.error("fraud_df is empty after timestamp handling. Cannot proceed.")
        fraud_df['country'] = pd.Series(dtype='object')
        return fraud_df

    # Time-based features
    logging.debug("Creating time-based features...")
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    logging.debug("Time-based features created.")
    
    if fraud_df['time_since_signup'].isnull().any():
        logging.warning("NaNs found in 'time_since_signup'. Filling with median.")
        median_time_since_signup = fraud_df['time_since_signup'].median()
        fraud_df['time_since_signup'].fillna(median_time_since_signup, inplace=True)

    # --- IP ADDRESS MAPPING USING IpAddress_to_Country.csv ---
    logging.info("Starting IP address to country mapping using IpAddress_to_Country.csv.")
    
    logging.info(f"Sample of 'ip_address' column before mapping (first 10 unique non-null):")
    logging.info(fraud_df['ip_address'].dropna().head(10).tolist())
    logging.info(f"Number of unique IPs: {fraud_df['ip_address'].nunique()}")
    logging.info(f"Number of null IPs: {fraud_df['ip_address'].isnull().sum()}")

    logging.debug("Converting fraud_df['ip_address'] to integer...")
    initial_fraud_rows_ip_conv = len(fraud_df)
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(convert_ip)
    
    fraud_df.dropna(subset=['ip_int'], inplace=True)
    rows_dropped_ip_int = initial_fraud_rows_ip_conv - len(fraud_df)
    if rows_dropped_ip_int > 0:
        logging.warning(f"Dropped {rows_dropped_ip_int} rows from fraud_df due to invalid 'ip_address' values (could not convert to int). Current fraud_df shape: {fraud_df.shape}")
    
    if fraud_df.empty:
        logging.error("fraud_df is empty after converting and cleaning 'ip_int'. Cannot proceed with merge.")
        fraud_df['country'] = pd.Series(dtype='object')
        return fraud_df
    logging.debug(f"fraud_df['ip_int'] head:\n{fraud_df['ip_int'].head()}")
    logging.debug(f"fraud_df['ip_int'] null count: {fraud_df['ip_int'].isnull().sum()}")

    logging.debug("Converting IP addresses to integers for ip_df (lower/upper bounds)...")
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].apply(convert_ip)
    ip_df['upper_bound_ip_address_int'] = ip_df['upper_bound_ip_address'].apply(convert_ip)
    logging.debug("IP addresses in ip_df converted to integers.")

    initial_ip_map_rows = len(ip_df)
    ip_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)
    ip_rows_dropped = initial_ip_map_rows - len(ip_df)
    if ip_rows_dropped > 0:
        logging.warning(f"Dropped {ip_rows_dropped} rows from ip_df due to NaN values in IP bound integers. Current ip_df shape: {ip_df.shape}")
    
    if ip_df.empty:
        logging.error("ip_df is empty after dropping rows with invalid IP bounds. Country mapping will not occur effectively.")
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        fraud_df['country'] = 'Unknown'
        return fraud_df

    fraud_df['ip_int'] = pd.to_numeric(fraud_df['ip_int'], errors='coerce')
    ip_df['lower_bound_ip_address_int'] = pd.to_numeric(ip_df['lower_bound_ip_address_int'], errors='coerce')
    
    fraud_df.dropna(subset=['ip_int'], inplace=True)
    ip_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)

    if fraud_df.empty or ip_df.empty:
        logging.error("One of the dataframes is empty after final IP integer cleanup and sorting. Merge_asof will fail.")
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        fraud_df['country'] = 'Unknown'
        return fraud_df

    ip_df.sort_values('lower_bound_ip_address_int', inplace=True)
    fraud_df.sort_values('ip_int', inplace=True)
    logging.debug(f"fraud_df shape after sorting: {fraud_df.shape}, ip_df shape after sorting: {ip_df.shape}")

    logging.info("Mapping countries to IP addresses using merge_asof...")
    try:
        merged_df = pd.merge_asof(
            fraud_df,
            ip_df[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
            left_on='ip_int',
            right_on='lower_bound_ip_address_int',
            direction='backward'
        )
        logging.debug(f"Shape after initial merge_asof: {merged_df.shape}")
        logging.debug(f"Columns after merge_asof: {merged_df.columns.tolist()}") # Diagnostic: print columns here

        # Corrected: Use 'country' directly as it's the result of the merge, not 'country_y'
        merged_df['country'] = np.where(
            (merged_df['ip_int'] >= merged_df['lower_bound_ip_address_int']) &
            (merged_df['ip_int'] <= merged_df['upper_bound_ip_address_int']),
            merged_df['country'], # Corrected from merged_df['country_y']
            None
        )
        logging.info("Country mapping validation complete.")

        if 'country' not in merged_df.columns:
            merged_df['country'] = None
        merged_df['country'].fillna('Unknown', inplace=True)
        
        logging.info(f"Sample of 'country' column after mapping (first 10 unique non-null):")
        logging.info(merged_df['country'].dropna().head(10).tolist())
        logging.info(f"Value counts for 'country' immediately after mapping:")
        logging.info(merged_df['country'].value_counts(dropna=False))

    except Exception as e:
        logging.error(f"Error during merge_asof or country validation: {e}. Returning fraud_df with 'country' set to 'Unknown'.")
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        fraud_df['country'] = 'Unknown'
        return fraud_df

    logging.debug("Dropping unneeded columns from merged_df...")
    cols_to_drop_from_merged = [
        'ip_address', 
        'ip_int',     
        'lower_bound_ip_address_int', 
        'upper_bound_ip_address_int', 
        'country_x' # Keep this for robustness if a 'country_x' somehow appeared.
    ]
    existing_cols_to_drop = [col for col in cols_to_drop_from_merged if col in merged_df.columns]
    merged_df.drop(existing_cols_to_drop, axis=1, errors='ignore', inplace=True) 
    
    logging.debug(f"Fraud data final shape after country mapping and NaN handling: {merged_df.shape}")
    logging.debug(f"Final 'country' value counts before return:\n{merged_df['country'].value_counts(dropna=False)}")
    logging.info("E-commerce fraud data preprocessing complete.")
    return merged_df

def preprocess_creditcard_data(credit_df):
    """Preprocess credit card transaction data."""
    logging.info("Starting preprocessing of credit card transaction data.")
    logging.debug(f"Initial credit_df shape: {credit_df.shape}")

    if credit_df.empty:
        logging.warning("Credit card DataFrame is empty. Skipping preprocessing.")
        return credit_df

    # Check for missing values (Credit Card dataset is typically very clean)
    if credit_df.isnull().sum().sum() > 0:
        logging.warning(f"Missing values found in credit_df. Count: {credit_df.isnull().sum().sum()}. Dropping rows with any NaNs.")
        initial_credit_rows = len(credit_df)
        credit_df.dropna(inplace=True)
        if len(credit_df) < initial_credit_rows:
            logging.warning(f"Dropped {initial_credit_rows - len(credit_df)} rows due to NaNs.")
        if credit_df.empty:
            logging.error("Credit card DataFrame became empty after NaN handling. Cannot proceed.")
            return credit_df

    # Separate features (X) and target (y)
    if 'Class' not in credit_df.columns:
        logging.error("'Class' column not found in credit card data. Cannot preprocess.")
        return credit_df
        
    X = credit_df.drop('Class', axis=1)
    y = credit_df['Class']

    logging.info("Scaling 'Time' and 'Amount' features for credit card data...")
    scaler = StandardScaler()
    
    # Scale 'Time' and 'Amount' features
    if 'Time' in X.columns:
        X['Time'] = scaler.fit_transform(X[['Time']])
        logging.debug("'Time' feature scaled.")
    else:
        logging.warning("'Time' column not found in credit card features for scaling.")
    
    if 'Amount' in X.columns:
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        logging.debug("'Amount' feature scaled.")
    else:
        logging.warning("'Amount' column not found in credit card features for scaling.")

    # The 'V' features (V1-V28) are already results of PCA and typically do not require
    # further scaling in this context as they are already standardized implicitly by PCA.
    # However, if you want *all* features to be on the same scale, you could scale them.
    # For now, we'll assume V features are good as they are from PCA.

    # Ensure X only contains numeric types before SMOTE
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.empty:
        logging.error("No numeric features found in credit card data after initial processing for SMOTE.")
        return pd.DataFrame() # Return empty if no numeric data for SMOTE

    logging.info("Handling class imbalance for credit card data using SMOTE...")
    X_balanced, y_balanced = handle_class_imbalance(X_numeric, y)

    # Combine X_balanced and y_balanced into a DataFrame
    processed_credit_df = pd.DataFrame(X_balanced, columns=X_numeric.columns)
    processed_credit_df['Class'] = y_balanced.values # Use .values to avoid index misalignment

    logging.debug(f"Credit card data final shape after preprocessing: {processed_credit_df.shape}")
    logging.debug(f"Final 'Class' value counts for credit card data:\n{processed_credit_df['Class'].value_counts(dropna=False)}")
    logging.info("Credit card transaction data preprocessing complete.")
    return processed_credit_df


def plot_eda(df, filename_prefix):
    """Generate and save EDA plots for a given dataframe."""
    logging.info(f"Generating EDA plots for {filename_prefix} data...")
    plots_output_dir = 'plots'
    os.makedirs(plots_output_dir, exist_ok=True)

    if df.empty:
        logging.warning(f"Skipping EDA plots for {filename_prefix} as the DataFrame is empty.")
        return

    # Dynamically select class column name
    class_col = 'class' if 'class' in df.columns else ('Class' if 'Class' in df.columns else None)

    if class_col:
        logging.debug(f"Generating class distribution plot for {filename_prefix}.")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=class_col, data=df)
        plt.title(f'{filename_prefix.replace("_", " ").title()} Class Distribution (Before SMOTE if applicable)')
        plt.xlabel(f'Class (0 = Non-Fraud, 1 = Fraud)')
        plt.ylabel('Count')
        plt.savefig(f'{plots_output_dir}/{filename_prefix}_class_distribution.png')
        plt.close()
        logging.debug(f"Saved plot: {plots_output_dir}/{filename_prefix}_class_distribution.png")
    else:
        logging.warning(f"Skipped class distribution plot for {filename_prefix}: Class column not found.")

    # Specific plots for fraud_data (if applicable)
    if filename_prefix == "fraud_data":
        plots_to_generate = {
            'age_distribution': {'type': 'histplot', 'x': 'age', 'bins': 30, 'kde': True, 'title': 'Age Distribution', 'xlabel': 'Age', 'ylabel': 'Count'},
            'hour_of_day_distribution': {'type': 'countplot', 'x': 'hour_of_day', 'title': 'Transaction Count by Hour of Day', 'xlabel': 'Hour of Day', 'ylabel': 'Count'},
            'day_of_week_distribution': {'type': 'countplot', 'x': 'day_of_week', 'title': 'Transaction Count by Day of Week', 'xlabel': 'Day of Week (0 = Monday, 6 = Sunday)', 'ylabel': 'Count'}
        }

        for plot_name, params in plots_to_generate.items():
            if params['x'] in df.columns and not df[params['x']].isnull().all():
                logging.debug(f"Generating plot: {filename_prefix}_{plot_name}.png")
                plt.figure(figsize=(10, 6))
                if params['type'] == 'countplot':
                    sns.countplot(x=params['x'], data=df)
                elif params['type'] == 'histplot':
                    sns.histplot(df[params['x']], bins=params['bins'], kde=params['kde'])
                plt.title(f"{filename_prefix.replace('_', ' ').title()} {params['title']}")
                plt.xlabel(params['xlabel'])
                plt.ylabel(params['ylabel'])
                plt.savefig(f'{plots_output_dir}/{filename_prefix}_{plot_name}.png')
                plt.close()
                logging.debug(f"Saved plot: {plots_output_dir}/{filename_prefix}_{plot_name}.png")
            else:
                logging.warning(f"Skipping plot '{plot_name}' for {filename_prefix} as column '{params['x']}' not found or is all NaN.")

    # Correlation Matrix (for relevant numerical features)
    logging.debug(f"Generating feature correlation matrix plot for {filename_prefix}.")
    plt.figure(figsize=(12, 10))
    
    if filename_prefix == "fraud_data":
        corr_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week']
        if class_col: corr_cols.append(class_col)
    elif filename_prefix == "creditcard_data":
        corr_cols = [col for col in df.columns if col.startswith('V') or col in ['Time', 'Amount']]
        if class_col: corr_cols.append(class_col)
    else:
        corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if class_col and class_col in corr_cols: # Ensure class is last for correlation
            corr_cols.remove(class_col)
            corr_cols.append(class_col)


    existing_corr_cols = [col for col in corr_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(existing_corr_cols) > 1:
        corr = df[existing_corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'{filename_prefix.replace("_", " ").title()} Feature Correlation Matrix')
        plt.savefig(f'{plots_output_dir}/{filename_prefix}_feature_correlation.png')
        plt.close()
        logging.debug(f"Saved plot: {plots_output_dir}/{filename_prefix}_feature_correlation.png")
    else:
        logging.warning(f"Not enough numeric columns to generate feature correlation plot for {filename_prefix}.")

    logging.info(f"EDA plots generation for {filename_prefix} complete.")

def save_data_summary(df, filename_prefix):
    """Save summary statistics to CSV files."""
    logging.info(f"Saving summary statistics for {filename_prefix} to CSV files...")
    summary_output_dir = 'data_summary'
    os.makedirs(summary_output_dir, exist_ok=True)

    if df.empty:
        logging.warning(f"Skipping data summary saving for {filename_prefix} as the DataFrame is empty.")
        return

    # Overall descriptive statistics
    df.describe(include='all').to_csv(f'{summary_output_dir}/{filename_prefix}_descriptive_statistics.csv')
    logging.debug(f"Saved {summary_output_dir}/{filename_prefix}_descriptive_statistics.csv")

    # Class distribution summary
    class_col = 'class' if 'class' in df.columns else ('Class' if 'Class' in df.columns else None)
    if class_col:
        value_counts_series = df[class_col].value_counts(dropna=False)
        summary_df = value_counts_series.reset_index()
        summary_df.columns = [class_col, 'count']
        summary_df['percentage'] = (summary_df['count'] / summary_df['count'].sum()) * 100
        summary_df.to_csv(f'{summary_output_dir}/{filename_prefix}_class_summary.csv', index=False)
        logging.debug(f"Saved {summary_output_dir}/{filename_prefix}_class_summary.csv")
    else:
        logging.warning(f"Skipped class summary for {filename_prefix}: Class column not found.")


    # Specific summaries for fraud_data
    if filename_prefix == "fraud_data":
        summaries = {
            'country_distribution': {'column': 'country', 'columns': ['country', 'count']},
            'source_distribution': {'column': 'source', 'columns': ['source', 'count']},
            'browser_distribution': {'column': 'browser', 'columns': ['browser', 'count']}
        }

        for name, config in summaries.items():
            if config['column'] in df.columns and not df[config['column']].isnull().all():
                value_counts_series = df[config['column']].value_counts(dropna=False)
                if value_counts_series.empty:
                    logging.warning(f"Value counts for '{config['column']}' are empty. Saving an empty CSV for '{filename_prefix}_{name}'.")
                    summary_df = pd.DataFrame(columns=config['columns'])
                else:
                    summary_df = value_counts_series.reset_index()
                    summary_df.columns = config['columns']
                summary_df.to_csv(f'{summary_output_dir}/{filename_prefix}_{name}.csv', index=False)
                logging.debug(f"Saved {summary_output_dir}/{filename_prefix}_{name}.csv")
            else:
                logging.warning(f"Skipping summary '{name}' for {filename_prefix} as column '{config['column']}' not found or is all NaN.")
    
    logging.info(f"Summary statistics saving for {filename_prefix} complete.")


def handle_class_imbalance(X, y):
    """Apply SMOTE to balance classes."""
    logging.info("Applying SMOTE to handle class imbalance...")
    
    if X.empty or y.empty:
        logging.warning("Cannot apply SMOTE: Input DataFrame (X) or Series (y) is empty.")
        return X, y

    unique_classes = y.nunique()
    if unique_classes < 2:
        logging.warning(f"Cannot apply SMOTE: Only {unique_classes} unique class(es) found in target variable. SMOTE requires at least 2.")
        return X, y
    
    class_counts = y.value_counts()
    if any(count < 2 for count in class_counts.values):
        logging.warning("Cannot apply SMOTE: At least one class has fewer than 2 samples. SMOTE requires a minimum number of samples for each class to generate synthetic data.")
        return X, y

    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logging.info(f"Class imbalance handled. Original samples: {len(y)}, Resampled samples: {len(y_res)}")
        return X_res, y_res
    except ValueError as e:
        logging.error(f"Error applying SMOTE: {e}. This might happen if a class has too few samples for strategy, or if feature scaling issues exist.")
        return X, y
    except Exception as e:
        logging.error(f"An unexpected error occurred during SMOTE: {e}. Returning original data.")
        return X, y

if __name__ == "__main__":
    logging.info("Starting fraud detection data preprocessing script.")

    os.makedirs('data/cleaned', exist_ok=True)
    os.makedirs('plots', exist_ok=True) # Ensure plots directory is created
    os.makedirs('data_summary', exist_ok=True) # Ensure data_summary directory is created

    try:
        fraud_data, ip_data, credit_data = load_datasets() 
    except FileNotFoundError:
        logging.critical("Required data files not found. Exiting preprocessing script.")
        print("\n" + "="*50)
        print("❌ Preprocessing FAILED: Data files not found.")
        print("Please ensure 'Fraud_Data.csv', 'IpAddress_to_Country.csv', and 'creditcard.csv' are in 'data/raw'.")
        print("==================================================")
        exit() 

    # --- Process Fraud Data ---
    fraud_processed = preprocess_fraud_data(fraud_data.copy(), ip_data.copy())

    if fraud_processed.empty:
        logging.critical("Fraud data DataFrame is empty after preprocessing. Cannot generate EDA or save.")
    else:
        plot_eda(fraud_processed, "fraud_data")
        save_data_summary(fraud_processed, "fraud_data")

        logging.info("One-hot encoding categorical features for fraud data...")
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        cat_features = ['source', 'browser', 'sex', 'country'] 
        existing_cat_features = [col for col in cat_features if col in fraud_processed.columns]
        
        fraud_encoded = fraud_processed.copy() 
        
        if existing_cat_features:
            logging.debug(f"Categorical features to encode: {existing_cat_features}")
            df_for_encoding = fraud_processed[existing_cat_features].copy()
            
            if df_for_encoding.empty:
                logging.warning("DataFrame for encoding is empty for fraud data. Skipping one-hot encoding.")
            else:
                for col in existing_cat_features:
                    df_for_encoding[col] = df_for_encoding[col].astype(str).fillna('Missing') 

                encoded = encoder.fit_transform(df_for_encoding).toarray()
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(existing_cat_features), index=fraud_processed.index)
                
                fraud_encoded = fraud_processed.drop(existing_cat_features, axis=1)
                fraud_encoded = fraud_encoded.reindex(encoded_df.index)
                fraud_encoded = pd.concat([fraud_encoded, encoded_df], axis=1)
                
                logging.info(f"Categorical features one-hot encoded for fraud data. New shape: {fraud_encoded.shape}")
        else:
            logging.warning("No categorical features found to one-hot encode for fraud data. Using fraud_processed as is.")

        logging.info("Scaling numerical features for fraud data...")
        fraud_balanced = fraud_encoded.copy() 

        if fraud_encoded.empty:
            logging.warning("fraud_encoded is empty for fraud data. Skipping numerical feature scaling and SMOTE.")
        else:
            scaler = StandardScaler()
            num_features = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week']
            existing_num_features = [col for col in num_features if col in fraud_encoded.columns and pd.api.types.is_numeric_dtype(fraud_encoded[col])]
            
            if existing_num_features:
                logging.debug(f"Numerical features to scale for fraud data: {existing_num_features}")
                for col in existing_num_features:
                    if fraud_encoded[col].isnull().any():
                        logging.warning(f"Column '{col}' for fraud data has NaNs before scaling. Filling with median for robust scaling.")
                        fraud_encoded[col].fillna(fraud_encoded[col].median(), inplace=True)

                fraud_encoded[existing_num_features] = scaler.fit_transform(fraud_encoded[existing_num_features])
                logging.info("Numerical features scaled for fraud data.")
            else:
                logging.warning("No numerical features found to scale for fraud data or none are numeric. Skipping scaling.")

            if 'class' in fraud_encoded.columns:
                if not pd.api.types.is_numeric_dtype(fraud_encoded['class']):
                    logging.warning("Class column for fraud data is not numeric. Attempting to convert to numeric, coercing errors.")
                    fraud_encoded['class'] = pd.to_numeric(fraud_encoded['class'], errors='coerce')
                    
                initial_rows_class_nan = len(fraud_encoded)
                fraud_encoded.dropna(subset=['class'], inplace=True)
                if len(fraud_encoded) < initial_rows_class_nan:
                    logging.warning(f"Dropped {initial_rows_class_nan - len(fraud_encoded)} rows from fraud data due to NaN values in 'class' column after conversion.")
                
                if fraud_encoded.empty:
                    logging.warning("fraud_encoded became empty after 'class' NaN handling for fraud data. Skipping SMOTE.")
                else:
                    X = fraud_encoded.drop('class', axis=1)
                    y = fraud_encoded['class']
                    
                    X_numeric = X.select_dtypes(include=[np.number])
                    
                    cols_before_all_nan_drop = X_numeric.shape[1]
                    X_numeric.dropna(axis=1, how='all', inplace=True)
                    if X_numeric.shape[1] < cols_before_all_nan_drop:
                        logging.warning(f"Dropped {cols_before_all_nan_drop - X_numeric.shape[1]} columns from X_numeric for fraud data that were all NaN before SMOTE.")

                    initial_X_rows = len(X_numeric)
                    X_numeric.dropna(inplace=True) 
                    if len(X_numeric) < initial_X_rows:
                        rows_dropped_X_smote = initial_X_rows - len(X_numeric)
                        logging.warning(f"Dropped {rows_dropped_X_smote} rows from X (features) for fraud data due to NaNs before SMOTE. Target 'y' will be aligned.")
                        y = y.loc[X_numeric.index] 

                    if X_numeric.empty:
                        logging.warning("X (features) DataFrame for fraud data is empty after NaN handling. Skipping SMOTE.")
                    elif y.empty:
                        logging.warning("y (target) Series for fraud data is empty after NaN handling for X. Skipping SMOTE.")
                    else:
                        X_balanced, y_balanced = handle_class_imbalance(X_numeric, y)

                        fraud_balanced = pd.DataFrame(X_balanced, columns=X_numeric.columns)
                        fraud_balanced['class'] = y_balanced.values
                        
                        logging.info(f"Final balanced fraud data shape: {fraud_balanced.shape}")
            else:
                logging.error("'class' column not found for imbalance handling in fraud data. Skipping SMOTE and saving fraud_encoded as fraud_balanced.")
                fraud_balanced = fraud_encoded.copy()
    
    # --- Process Credit Card Data ---
    
    credit_processed = preprocess_creditcard_data(credit_data.copy())

    if credit_processed.empty:
        logging.critical("Credit card DataFrame is empty after preprocessing. Cannot generate EDA or save.")
    else:
        plot_eda(credit_processed, "creditcard_data") # Use a different prefix for credit card plots
        save_data_summary(credit_processed, "creditcard_data") # Use a different prefix for credit card summaries


    # --- Save Processed Data ---
    try:
        if fraud_processed.empty:
            logging.warning("fraud_balanced DataFrame is empty. Skipping saving of processed_fraud_data.csv.")
        else:
            # Save the final fraud_balanced data, or fraud_processed if SMOTE was skipped
            if 'fraud_balanced' in locals() and not fraud_balanced.empty:
                 fraud_balanced.to_csv("data/cleaned/processed_fraud_data.csv", index=False)
                 logging.info("Processed fraud data saved successfully.")
            elif not fraud_processed.empty: # Fallback if fraud_balanced wasn't created
                 fraud_processed.to_csv("data/cleaned/processed_fraud_data.csv", index=False)
                 logging.info("Processed fraud data (without full balancing) saved successfully.")
            else:
                 logging.warning("No fraud data to save.")

        if credit_processed.empty:
            logging.warning("Credit card data DataFrame is empty. Skipping saving of processed_creditcard.csv.")
        else:
            credit_processed.to_csv("data/cleaned/processed_creditcard.csv", index=False)
            logging.info("Processed credit card data saved successfully.")

    except Exception as e:
        logging.error(f"Error saving processed data: {e}")

    logging.info("Fraud detection data preprocessing script finished.")
    print("\n" + "="*50)
    print("✅ Preprocessing complete for both Fraud and Credit Card data.")
    print(f"📁 Processed data saved to: {os.path.join(os.getcwd(), 'data/cleaned')}")
    print(f"📈 Plots saved to: {os.path.join(os.getcwd(), 'plots/')}")
    print(f"📊 Summary statistics saved to: {os.path.join(os.getcwd(), 'data_summary/')}")
    print("==================================================")