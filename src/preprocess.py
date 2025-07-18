"""
preprocess.py
Created by: Addisu Taye
Date Created: July 18, 2025
Updated: July 18, 2025
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

# Keep convert_ip as is, it's correct for dot-decimal IPs (like in ip_map)
def convert_ip(ip):
    """Convert IP address string (dot-decimal) to integer."""
    if pd.isna(ip):
        logging.debug(f"Input IP is NaN/None: {ip}. Returning NaN.")
        return np.nan
    try:
        parts = str(ip).strip().split('.')
        if len(parts) != 4 or not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
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
    # Using errors='coerce' to turn invalid parsing into NaT
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce') 
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')
    
    # Drop rows where timestamp conversion failed
    initial_rows_ts_conv = len(fraud_df)
    fraud_df.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
    if len(fraud_df) < initial_rows_ts_conv:
        logging.warning(f"Dropped {initial_rows_ts_conv - len(fraud_df)} rows due to invalid signup_time or purchase_time. Current shape: {fraud_df.shape}")
    
    logging.debug("Timestamp columns converted and NaTs handled.")
    
    if fraud_df.empty:
        logging.error("fraud_df is empty after timestamp handling. Cannot proceed.")
        return fraud_df

    # Time-based features
    logging.debug("Creating time-based features...")
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    logging.debug("Time-based features created.")
    
    # Fill any NaNs created by timedelta (e.g., if signup > purchase time, though this will be negative)
    # Using median is a reasonable strategy for time-since-signup.
    if fraud_df['time_since_signup'].isnull().any():
        logging.warning("NaNs found in 'time_since_signup'. Filling with median.")
        median_time_since_signup = fraud_df['time_since_signup'].median()
        fraud_df['time_since_signup'].fillna(median_time_since_signup, inplace=True)

    # --- CRITICAL CHANGE FOR IP ADDRESS HANDLING ---
    # Based on the sample image, fraud_df['ip_address'] is already a float/integer representation.
    # We need to convert it directly to an integer without using the convert_ip function.
    logging.info("Converting fraud_df['ip_address'] directly to integer as it appears pre-converted...")
    initial_fraud_rows_ip = len(fraud_df)
    # Convert to numeric, coercing errors to NaN. Then fill NaNs or drop.
    fraud_df['ip_int'] = pd.to_numeric(fraud_df['ip_address'], errors='coerce')
    # If there's a decimal part like .8, we need to convert to int.
    # Using .astype(int) will truncate, but .round().astype(int) is safer if it's truly float.
    # Assuming .8 is just noise or a very small decimal, direct int conversion is okay.
    fraud_df['ip_int'] = fraud_df['ip_int'].fillna(-1).astype(int) # Temporarily fill NaN with -1 for astype(int)
    fraud_df['ip_int'] = fraud_df['ip_int'].replace(-1, np.nan) # Revert -1 back to NaN

    # Drop rows that failed IP integer conversion (e.g., were completely non-numeric)
    fraud_df.dropna(subset=['ip_int'], inplace=True)
    rows_dropped_ip_int = initial_fraud_rows_ip - len(fraud_df)
    if rows_dropped_ip_int > 0:
        logging.warning(f"Dropped {rows_dropped_ip_int} rows from fraud_df due to non-numeric/invalid 'ip_address' values (could not convert to int). Current fraud_df shape: {fraud_df.shape}")
    
    if fraud_df.empty:
        logging.error("fraud_df is empty after converting and cleaning 'ip_int'. Cannot proceed with merge.")
        return fraud_df

    # Apply convert_ip ONLY to ip_map, which should have dot-decimal IPs
    logging.debug("Converting IP addresses to integers for ip_df (lower/upper bounds)...")
    ip_df['lower_bound_ip_address_int'] = ip_df['lower_bound_ip_address'].apply(convert_ip)
    ip_df['upper_bound_ip_address_int'] = ip_df['upper_bound_ip_address'].apply(convert_ip)
    logging.debug("IP addresses in ip_df converted to integers.")

    # --- Handle NaN values in merge keys before merge_asof ---
    initial_ip_map_rows = len(ip_df)
    ip_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)
    ip_rows_dropped = initial_ip_map_rows - len(ip_df)
    if ip_rows_dropped > 0:
        logging.warning(f"Dropped {ip_rows_dropped} rows from ip_df due to NaN values in IP bound integers (invalid/unconvertible dot-decimal IPs). Current ip_df shape: {ip_df.shape}")
    
    if ip_df.empty:
        logging.error("ip_df is empty after dropping rows with invalid IP bounds. Country mapping will not occur effectively.")
        # If ip_df is empty, we can't map countries. 
        # We need to drop original 'ip_address' column from fraud_df. 'ip_int' is kept for potential future use.
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        # Add a 'country' column filled with None as a placeholder if no mapping is possible
        fraud_df['country'] = None 
        return fraud_df # Return fraud_df as is, but with 'country' column
    # --- END Handling NaNs ---

    # Sort IP map for merge_asof. Important: Ensure columns are numeric before sorting.
    logging.debug("Sorting IP mapping data for efficient merging...")
    # Ensure columns used for sorting are numeric (already done, but re-confirm)
    ip_df['lower_bound_ip_address_int'] = pd.to_numeric(ip_df['lower_bound_ip_address_int'], errors='coerce')
    fraud_df['ip_int'] = pd.to_numeric(fraud_df['ip_int'], errors='coerce')
    
    # Drop NaNs again after to_numeric, just in case
    fraud_df.dropna(subset=['ip_int'], inplace=True)
    ip_df.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)

    if fraud_df.empty or ip_df.empty:
        logging.error("One of the dataframes is empty after final IP integer cleanup and sorting. Merge_asof will fail.")
        # Ensure ip_address is dropped and country is added as None if merge cannot happen
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        fraud_df['country'] = None
        return fraud_df

    ip_df.sort_values('lower_bound_ip_address_int', inplace=True)
    fraud_df.sort_values('ip_int', inplace=True) # Important for merge_asof
    logging.debug(f"fraud_df shape after sorting: {fraud_df.shape}, ip_df shape after sorting: {ip_df.shape}")

    # Map country based on IP range using merge_asof for efficiency
    logging.info("Mapping countries to IP addresses using merge_asof (optimized)...")
    try:
        # Perform merge_asof
        merged_df = pd.merge_asof(
            fraud_df,
            ip_df[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
            left_on='ip_int',
            right_on='lower_bound_ip_address_int',
            direction='backward'
        )
        logging.debug(f"Shape after initial merge_asof: {merged_df.shape}")

        # Check if the IP falls within the upper bound for the found country
        # This will set 'country' to None if the IP is not within the matched range
        merged_df['country'] = np.where(
            (merged_df['ip_int'] >= merged_df['lower_bound_ip_address_int']) &
            (merged_df['ip_int'] <= merged_df['upper_bound_ip_address_int']),
            merged_df['country_y'], # 'country_y' is from ip_df after merge
            None
        )
        logging.info("Country mapping complete.")

    except Exception as e:
        logging.error(f"Error during merge_asof or country validation: {e}. Check if merge keys are clean and sorted. Returning fraud_df with attempted IP drops only.")
        # If merge_asof fails, ensure ip_address is dropped and country is added as None
        fraud_df.drop(['ip_address'], axis=1, errors='ignore', inplace=True)
        fraud_df['country'] = None
        return fraud_df # Return fraud_df which has been cleaned up to this point

    # Drop unneeded cols from the merged_df
    logging.debug("Dropping unneeded columns from merged_df...")
    cols_to_drop_from_merged = ['ip_address', 'ip_int', 'lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country_x', 'country_y']
    existing_cols_to_drop = [col for col in cols_to_drop_from_merged if col in merged_df.columns]
    merged_df.drop(existing_cols_to_drop, axis=1, errors='ignore', inplace=True) 
    
    # Handle NaNs in the final 'country' column
    initial_merged_rows = len(merged_df)
    merged_df.dropna(subset=['country'], inplace=True)
    rows_dropped_after_country = initial_merged_rows - len(merged_df)
    if rows_dropped_after_country > 0:
        logging.warning(f"Dropped {rows_dropped_after_country} rows from fraud data due to no country match found after IP mapping (final clean-up). Current shape: {merged_df.shape}")

    if merged_df.empty:
        logging.error("fraud_processed DataFrame is empty after country mapping and dropping NaNs. Subsequent steps will fail.")

    logging.debug(f"Fraud data final shape after country mapping and dropping NaNs: {merged_df.shape}")
    logging.info("E-commerce fraud data preprocessing complete.")
    return merged_df # Return the merged_df, which is now the processed fraud_df

def plot_eda(fraud_df):
    """Generate and save EDA plots."""
    logging.info("Generating EDA plots...")
    os.makedirs('plots', exist_ok=True)

    if fraud_df.empty:
        logging.warning("Skipping EDA plots as the DataFrame is empty.")
        return

    plots_to_generate = {
        'class_distribution': {'type': 'countplot', 'x': 'class', 'title': 'Class Distribution (Before SMOTE)', 'xlabel': 'Class (0 = Non-Fraud, 1 = Fraud)', 'ylabel': 'Count'},
        'age_distribution': {'type': 'histplot', 'x': 'age', 'bins': 30, 'kde': True, 'title': 'Age Distribution', 'xlabel': 'Age', 'ylabel': 'Count'},
        'hour_of_day_distribution': {'type': 'countplot', 'x': 'hour_of_day', 'title': 'Transaction Count by Hour of Day', 'xlabel': 'Hour of Day', 'ylabel': 'Count'},
        'day_of_week_distribution': {'type': 'countplot', 'x': 'day_of_week', 'title': 'Transaction Count by Day of Week', 'xlabel': 'Day of Week (0 = Monday, 6 = Sunday)', 'ylabel': 'Count'}
    }

    for plot_name, params in plots_to_generate.items():
        # Check if column exists AND has non-null values
        if params['x'] not in fraud_df.columns or fraud_df[params['x']].isnull().all():
            logging.warning(f"Skipping plot '{plot_name}' as column '{params['x']}' not found or is all NaN in DataFrame.")
            continue
        logging.debug(f"Generating plot: {plot_name}.png")
        plt.figure(figsize=(10, 6))
        if params['type'] == 'countplot':
            sns.countplot(x=params['x'], data=fraud_df)
        elif params['type'] == 'histplot':
            sns.histplot(fraud_df[params['x']], bins=params['bins'], kde=params['kde'])
        plt.title(params['title'])
        plt.xlabel(params['xlabel'])
        plt.ylabel(params['ylabel'])
        plt.savefig(f'plots/{plot_name}.png')
        plt.close()
        logging.debug(f"Saved plot: plots/{plot_name}.png")

    # Feature correlation
    logging.debug("Generating feature correlation matrix plot.")
    plt.figure(figsize=(12, 10))
    # Ensure all columns exist and are numeric before calculating correlation
    corr_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'class']
    existing_corr_cols = [col for col in corr_cols if col in fraud_df.columns and pd.api.types.is_numeric_dtype(fraud_df[col])]
    
    if len(existing_corr_cols) > 1: # Need at least 2 columns for correlation
        corr = fraud_df[existing_corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.savefig('plots/feature_correlation.png')
        plt.close()
        logging.debug("Saved plot: plots/feature_correlation.png")
    else:
        logging.warning("Not enough numeric columns to generate feature correlation plot.")

    logging.info("EDA plots generation complete.")

def save_data_summary(fraud_df):
    """Save summary statistics to CSV files."""
    logging.info("Saving summary statistics to CSV files...")
    os.makedirs('data_summary', exist_ok=True)

    if fraud_df.empty:
        logging.warning("Skipping data summary saving as the DataFrame is empty.")
        return

    summaries = {
        'fraud_class_summary': {'column': 'class', 'columns': ['class', 'count'], 'post_process': lambda df: df.assign(percentage=(df['count'] / df['count'].sum()) * 100)},
        'country_distribution': {'column': 'country', 'columns': ['country', 'count']},
        'source_distribution': {'column': 'source', 'columns': ['source', 'count']},
        'browser_distribution': {'column': 'browser', 'columns': ['browser', 'count']}
    }

    for name, config in summaries.items():
        # Check if column exists AND has non-null values
        if config['column'] not in fraud_df.columns or fraud_df[config['column']].isnull().all():
            logging.warning(f"Skipping summary '{name}' as column '{config['column']}' not found or is all NaN in DataFrame.")
            continue
        logging.debug(f"Saving {name}.csv")
        summary_df = fraud_df[config['column']].value_counts().reset_index()
        summary_df.columns = config['columns']
        if 'post_process' in config:
            summary_df = config['post_process'](summary_df)
        summary_df.to_csv(f'data_summary/{name}.csv', index=False)
        logging.debug(f"Saved data_summary/{name}.csv")

    # Correlation matrix
    logging.debug("Saving feature correlation matrix to CSV.")
    corr_cols = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week', 'class']
    existing_corr_cols = [col for col in corr_cols if col in fraud_df.columns and pd.api.types.is_numeric_dtype(fraud_df[col])]
    if len(existing_corr_cols) > 1:
        corr = fraud_df[existing_corr_cols].corr()
        corr.to_csv('data_summary/feature_correlation.csv')
        logging.debug("Saved data_summary/feature_correlation.csv")
    else:
        logging.warning("Not enough numeric columns to save feature correlation to CSV.")

    logging.info("Summary statistics saving complete.")

def handle_class_imbalance(X, y):
    """Apply SMOTE to balance classes."""
    logging.info("Applying SMOTE to handle class imbalance...")
    
    if X.empty or y.empty:
        logging.warning("Cannot apply SMOTE: Input DataFrame (X) or Series (y) is empty.")
        return X, y

    # Check if there's enough data for SMOTE for both classes
    unique_classes = y.nunique()
    if unique_classes < 2:
        logging.warning(f"Cannot apply SMOTE: Only {unique_classes} unique class(es) found in target variable. SMOTE requires at least 2.")
        return X, y
    
    class_counts = y.value_counts()
    # SMOTE requires at least 2 samples for the minority class to generate synthetic samples.
    # It also needs more than 1 sample for the majority class to be able to fit.
    if any(count < 2 for count in class_counts.values): # Use .values to get counts
        logging.warning("Cannot apply SMOTE: At least one class has fewer than 2 samples. SMOTE requires at least 2 samples for each class.")
        return X, y

    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logging.info(f"Class imbalance handled. Original samples: {len(y)}, Resampled samples: {len(y_res)}")
        return X_res, y_res
    except ValueError as e:
        logging.error(f"Error applying SMOTE: {e}. This might happen if a class has too few samples for strategy, or feature scaling issues.")
        return X, y # Return original data if SMOTE fails
    except Exception as e:
        logging.error(f"An unexpected error occurred during SMOTE: {e}. Returning original data.")
        return X, y

if __name__ == "__main__":
    logging.info("Starting fraud detection data preprocessing script.")

    # Create directories if they don't exist
    os.makedirs('data/cleaned', exist_ok=True)

    # Load data
    fraud_data, ip_data, credit_data = load_datasets()

    # Process e-commerce fraud data
    fraud_processed = preprocess_fraud_data(fraud_data.copy(), ip_data.copy())

    # --- Crucial check if fraud_processed is empty ---
    if fraud_processed.empty:
        logging.critical("Fraud data DataFrame is empty after preprocessing. Exiting script.")
        print("\n" + "="*50)
        print("❌ Preprocessing FAILED: fraud_processed DataFrame is empty.")
        print("Please check your raw data and preprocessing steps, especially IP mapping and NaN handling.")
        print("==================================================")
        exit() # Exit the script if no data to process

    # Generate EDA plots
    plot_eda(fraud_processed)

    # Save summary statistics
    save_data_summary(fraud_processed)

    # One-hot encode categoricals
    logging.info("One-hot encoding categorical features...")
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    cat_features = ['source', 'browser', 'sex', 'country']
    existing_cat_features = [col for col in cat_features if col in fraud_processed.columns]
    
    if existing_cat_features:
        logging.debug(f"Categorical features to encode: {existing_cat_features}")
        df_for_encoding = fraud_processed[existing_cat_features].copy() 
        
        # --- Check df_for_encoding for emptiness before fit_transform ---
        if df_for_encoding.empty:
            logging.warning("df_for_encoding is empty. Skipping one-hot encoding.")
            fraud_encoded = fraud_processed.copy()
        else:
            # Check if all features are indeed categorical (object/string type)
            non_cat_in_encoding_df = [col for col in df_for_encoding.columns if not pd.api.types.is_object_dtype(df_for_encoding[col]) and not pd.api.types.is_string_dtype(df_for_encoding[col])]
            if non_cat_in_encoding_df:
                logging.warning(f"Non-categorical columns found in df_for_encoding: {non_cat_in_encoding_df}. Attempting to convert to string.")
                for col in non_cat_in_encoding_df:
                    df_for_encoding[col] = df_for_encoding[col].astype(str)

            # Drop rows with NaNs in categorical columns if they exist, before encoding.
            initial_rows_encoding = len(df_for_encoding)
            df_for_encoding.dropna(inplace=True)
            if len(df_for_encoding) < initial_rows_encoding:
                logging.warning(f"Dropped {initial_rows_encoding - len(df_for_encoding)} rows from df_for_encoding due to NaNs before one-hot encoding. Current shape: {df_for_encoding.shape}")
            
            if df_for_encoding.empty:
                logging.warning("df_for_encoding became empty after dropping NaNs. Skipping one-hot encoding.")
                fraud_encoded = fraud_processed.copy()
            else:
                encoded = encoder.fit_transform(df_for_encoding).toarray()
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(existing_cat_features), index=df_for_encoding.index) 
                
                # Align the base DataFrame's index with the index of the encoded features
                fraud_encoded_base = fraud_processed.drop(existing_cat_features, axis=1)
                # Use .reindex to align fraud_encoded_base to encoded_df's index, filling NaNs for missing rows if any
                fraud_encoded_base = fraud_encoded_base.reindex(encoded_df.index) 
                # Concatenate the base DataFrame with the encoded features
                fraud_encoded = pd.concat([fraud_encoded_base, encoded_df], axis=1)
                
                logging.info(f"Categorical features one-hot encoded. New shape: {fraud_encoded.shape}")
    else:
        fraud_encoded = fraud_processed.copy()
        logging.warning("No categorical features found to one-hot encode. Using fraud_processed as is.")

    # Scale numerical features
    logging.info("Scaling numerical features...")
    if fraud_encoded.empty:
        logging.warning("fraud_encoded is empty. Skipping numerical feature scaling.")
        fraud_balanced = fraud_encoded.copy()
    else:
        scaler = StandardScaler()
        num_features = ['purchase_value', 'age', 'time_since_signup', 'hour_of_day', 'day_of_week']
        existing_num_features = [col for col in num_features if col in fraud_encoded.columns and pd.api.types.is_numeric_dtype(fraud_encoded[col])]
        if existing_num_features:
            logging.debug(f"Numerical features to scale: {existing_num_features}")
            for col in existing_num_features:
                if fraud_encoded[col].isnull().any():
                    logging.warning(f"Column '{col}' has NaNs before scaling. Filling with median for robust scaling. Column: {col}")
                    fraud_encoded[col].fillna(fraud_encoded[col].median(), inplace=True)

            fraud_encoded[existing_num_features] = scaler.fit_transform(fraud_encoded[existing_num_features])
            logging.info("Numerical features scaled.")
        else:
            logging.warning("No numerical features found to scale or none are numeric. Skipping scaling.")

        # Handle class imbalance
        if 'class' in fraud_encoded.columns:
            if not pd.api.types.is_numeric_dtype(fraud_encoded['class']):
                logging.warning("Class column is not numeric. Attempting to convert to numeric, coercing errors.")
                fraud_encoded['class'] = pd.to_numeric(fraud_encoded['class'], errors='coerce')
                
            initial_rows_class_nan = len(fraud_encoded)
            fraud_encoded.dropna(subset=['class'], inplace=True) 
            if len(fraud_encoded) < initial_rows_class_nan:
                logging.warning(f"Dropped {initial_rows_class_nan - len(fraud_encoded)} rows due to NaN values in 'class' column after conversion.")
            
            if fraud_encoded.empty:
                logging.warning("fraud_encoded became empty after 'class' NaN handling. Skipping SMOTE.")
                fraud_balanced = fraud_encoded.copy()
            else:
                X = fraud_encoded.drop('class', axis=1)
                y = fraud_encoded['class']
                
                X_numeric = X.select_dtypes(include=[np.number])
                
                cols_before_all_nan_drop = X_numeric.shape[1]
                X_numeric.dropna(axis=1, how='all', inplace=True)
                if X_numeric.shape[1] < cols_before_all_nan_drop:
                    logging.warning(f"Dropped {cols_before_all_nan_drop - X_numeric.shape[1]} columns from X_numeric that were all NaN.")

                initial_X_rows = len(X_numeric)
                X_numeric.dropna(inplace=True)
                if len(X_numeric) < initial_X_rows:
                    rows_dropped_X_smote = initial_X_rows - len(X_numeric)
                    logging.warning(f"Dropped {rows_dropped_X_smote} rows from X (features) due to NaNs before SMOTE. Target 'y' will be aligned.")
                    y = y.loc[X_numeric.index]

                if X_numeric.empty:
                    logging.warning("X (features) DataFrame is empty after NaN handling. Skipping SMOTE.")
                    fraud_balanced = fraud_encoded.copy()
                elif y.empty:
                    logging.warning("y (target) Series is empty after NaN handling for X. Skipping SMOTE.")
                    fraud_balanced = fraud_encoded.copy()
                else:
                    X_balanced, y_balanced = handle_class_imbalance(X_numeric, y)

                    fraud_balanced = pd.DataFrame(X_balanced, columns=X_numeric.columns)
                    fraud_balanced['class'] = y_balanced.values
                    
                    logging.info(f"Final balanced fraud data shape: {fraud_balanced.shape}")
        else:
            logging.error("'class' column not found for imbalance handling. Skipping SMOTE and saving fraud_encoded as fraud_balanced.")
            fraud_balanced = fraud_encoded.copy()

    # Save processed data
    try:
        if fraud_balanced.empty:
            logging.warning("fraud_balanced DataFrame is empty. Skipping saving of processed_fraud_data.csv.")
        else:
            fraud_balanced.to_csv("data/cleaned/processed_fraud_data.csv", index=False)
            logging.info("Processed fraud data saved successfully.")
        
        if credit_data.empty:
             logging.warning("Credit card data DataFrame is empty. Skipping saving of processed_creditcard.csv.")
        else:
            credit_data.to_csv("data/cleaned/processed_creditcard.csv", index=False)
            logging.info("Credit card data saved successfully.")

    except Exception as e:
        logging.error(f"Error saving processed data: {e}")

    logging.info("Fraud detection data preprocessing script finished.")
    print("\n" + "="*50)
    print("✅ Preprocessing complete.")
    print(f"📁 Processed data saved to: {os.path.join(os.getcwd(), 'data/cleaned')}")
    print(f"📈 Plots saved to: {os.path.join(os.getcwd(), 'plots/')}")
    print(f"📊 Summary statistics saved to: {os.path.join(os.getcwd(), 'data_summary/')}")
    print("="*50)