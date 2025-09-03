import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from the path
try:
    houses_data = pd.read_csv('P2_Imputation_Norm/house_prices_mod.csv', sep=',')
    print("Dataset loaded successfully.\n")

    # Make a copy for imputation to preserve the original data
    df_imputed = houses_data.copy()

    # --- Preprocessing: Clean and convert the 'Amount' column to numeric ---
    def clean_amount(x):
        if pd.isna(x) or isinstance(x, (int, float)):
            return x
        s = str(x).lower().replace(' ', '')
        if 'lac' in s:
            return float(s.replace('lac', '')) * 100000
        if 'crore' in s:
            return float(s.replace('crore', '')) * 10000000
        return pd.to_numeric(s, errors='coerce')

    df_imputed['Amount (in rupees)'] = df_imputed['Amount(in rupees)'].apply(clean_amount)

    # --- Ensure all columns are numeric ---
    numerical_cols = ['Price (in rupees)', 'Carpet Area', 'Super Area', 'Bathroom', 'Balcony', 'Car Parking']
    for col in numerical_cols:
        df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
        
    print("\nNumerical columns successfully converted to numeric type.")
    print("New null values per column after cleaning:")
    print(df_imputed.isnull().sum())

    # --- Imputation by Mean/Mode (Simple Deterministic) ---
    print("\n--- Applying Mean/Mode Imputation ---")
    
    for col in numerical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mean_value = df_imputed[col].mean()
            df_imputed[f'{col}_imputed_mean'] = df_imputed[col].fillna(mean_value)
            print(f"Column '{col}' imputed with mean value: {mean_value:.2f}")

    categorical_cols = ['Transaction', 'Furnishing', 'facing', 'overlooking']
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0 and not df_imputed[col].mode().empty:
            mode_value = df_imputed[col].mode()[0]
            df_imputed[f'{col}_imputed_mode'] = df_imputed[col].fillna(mode_value)
            print(f"Column '{col}' imputed with mode value: {mode_value}")
        elif df_imputed[col].isnull().sum() > 0:
            print(f"Warning: Column '{col}' has no mode, skipping mode imputation.")

    # --- Imputation by Regression (Advanced Deterministic) ---
    print("\n--- Applying Regression Imputation for 'Price (in rupees)' ---")

    features_for_regression = ['Carpet Area', 'Bathroom']
    df_train = df_imputed.dropna(subset=['Price (in rupees)'] + features_for_regression)
    
    if not df_train.empty:
        model = LinearRegression()
        model.fit(df_train[features_for_regression], df_train['Price (in rupees)'])
        df_missing = df_imputed[df_imputed['Price (in rupees)'].isnull()]
        if not df_missing.empty:
            df_imputed.loc[df_missing.index, 'Price_imputed_regression'] = model.predict(df_missing[features_for_regression])
            print("Missing 'Price (in rupees)' values imputed using Linear Regression.")
        else:
            print("No missing values found in 'Price (in rupees)' to impute.")
    else:
        print("Not enough complete data to train the regression model.")

    # --- Imputation with Random Values (Stochastic) ---
    print("\n--- Applying Random Imputation (Stochastic) for numerical columns ---")
    
    for col in numerical_cols:
        if df_imputed[col].isnull().sum() > 0:
            data_to_impute = df_imputed[col].dropna()
            mean_val = data_to_impute.mean()
            std_val = data_to_impute.std()
            missing_indices = df_imputed[col].isnull()
            num_missing = missing_indices.sum()
            random_imputation = np.random.normal(mean_val, std_val, size=num_missing)
            df_imputed.loc[missing_indices, f'{col}_imputed_stochastic'] = random_imputation
            print(f"Column '{col}' imputed with {num_missing} random values.")
    
    # --- NEW STEP: Data Normalization (Min-Max) ---
    print("\n--- Normalizing Imputed Numerical Data (Min-Max) ---")
    # Identify the columns to normalize (using the mean imputed values for demonstration)
    cols_to_normalize = ['Price (in rupees)_imputed_mean', 'Carpet Area_imputed_mean', 'Bathroom_imputed_mean']
    
    for col in cols_to_normalize:
        min_val = df_imputed[col].min()
        max_val = df_imputed[col].max()
        df_imputed[f'{col.replace("_imputed_mean", "")}_normalized'] = (df_imputed[col] - min_val) / (max_val - min_val)
        print(f"Column '{col}' successfully normalized.")
        
    # --- Verify and Export the Imputed and Normalized Dataset ---
    print("\n--- Verifying and Exporting Data ---")
    print("New null values count after all operations:")
    print(df_imputed.isnull().sum())

    # Create a new DataFrame with the original, imputed and normalized columns
    columns_to_export = [
        'Index', 'Title', 'Description', 'Amount (in rupees)', 
        'Price (in rupees)_imputed_mean', 'Price (in rupees)_normalized',
        'location', 'Carpet Area_imputed_mean', 'Carpet Area_normalized',
        'Status', 'Floor',
        'Transaction_imputed_mode', 'Furnishing_imputed_mode', 'facing_imputed_mode', 
        'overlooking_imputed_mode', 'Society', 'Bathroom_imputed_mean', 'Bathroom_normalized', 
        'Balcony_imputed_mean', 'Car Parking_imputed_mean', 'Ownership', 'Super Area_imputed_mean', 'Dimensions', 'Plot Area'
    ]
    
    # Conditionally add the new columns if they were created
    if 'Price_imputed_regression' in df_imputed.columns:
        columns_to_export.insert(5, 'Price_imputed_regression')
    if 'Price (in rupees)_imputed_stochastic' in df_imputed.columns:
        columns_to_export.insert(6, 'Price (in rupees)_imputed_stochastic')
        
    df_final = df_imputed[columns_to_export]

    # Export the cleaned data to a new CSV file
    output_filename = 'P2_Imputation_Norm/house_prices_imputed_normalized.csv'
    df_final.to_csv(output_filename, index=False)
    print(f"\nSuccessfully created and saved the new CSV file: '{output_filename}'")
    
except FileNotFoundError:
    print("Error: The file 'P2_Imputation_Norm/house_prices_mod.csv' was not found. Please ensure the path is correct.\n")