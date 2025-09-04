import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from the path
try:
    houses_data = pd.read_csv('P2_Imputation_Norm/house_prices_mod.csv', sep=',')
    print("Dataset loaded successfully.\n")

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

    houses_data['Amount (in rupees)'] = houses_data['Amount(in rupees)'].apply(clean_amount)

    # --- Ensure all columns are numeric ---
    numerical_cols = ['Price (in rupees)', 'Carpet Area', 'Super Area', 'Bathroom', 'Balcony', 'Car Parking']
    for col in numerical_cols:
        houses_data[col] = pd.to_numeric(houses_data[col], errors='coerce')
        
    print("\nNumerical columns successfully converted to numeric type.")
    print("New null values per column after cleaning:")
    print(houses_data.isnull().sum())

    # --- Normalization helper function ---
    def normalize_dataframe(df, cols):
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm[cols] = scaler.fit_transform(df_norm[cols])
        return df_norm

    # --- Imputation by Mean/Mode (Simple Deterministic) ---
    print("\n--- Applying Mean/Mode Imputation ---")
    df_mean_mode = houses_data.copy()
    
    for col in numerical_cols:
        if df_mean_mode[col].isnull().sum() > 0:
            mean_value = df_mean_mode[col].mean()
            df_mean_mode[f'{col}_imputed_mean'] = df_mean_mode[col].fillna(mean_value)
            print(f"Column '{col}' imputed with mean value: {mean_value:.2f}")

    categorical_cols = ['Transaction', 'Furnishing', 'facing', 'overlooking']
    for col in categorical_cols:
        if df_mean_mode[col].isnull().sum() > 0 and not df_mean_mode[col].mode().empty:
            mode_value = df_mean_mode[col].mode()[0]
            df_mean_mode[f'{col}_imputed_mode'] = df_mean_mode[col].fillna(mode_value)
            print(f"Column '{col}' imputed with mode value: {mode_value}")
        elif df_mean_mode[col].isnull().sum() > 0:
            print(f"Warning: Column '{col}' has no mode, skipping mode imputation.")

    # Normalize after imputation
    df_mean_mode = normalize_dataframe(df_mean_mode, numerical_cols)
    output_filename_mean_mode = 'P2_Imputation_Norm/house_prices_imputed_mean_mode.csv'
    df_mean_mode.to_csv(output_filename_mean_mode, index=False)
    print(f"\nSuccessfully created and saved the new CSV file: '{output_filename_mean_mode}'")

    # --- Imputation by KNN (Advanced Deterministic) ---
    print("\n--- Applying KNN Imputation ---")
    df_knn = houses_data.copy()
    
    sample_size = min(10000, len(df_knn))
    df_sample = df_knn[numerical_cols].sample(n=sample_size, random_state=42)

    valid_numeric_cols = [col for col in numerical_cols if col in df_sample.columns and df_sample[col].notna().any()]
    
    if valid_numeric_cols:
        cols_with_missing_values = [col for col in valid_numeric_cols if df_sample[col].isnull().sum() > 0]

        if cols_with_missing_values:
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df_sample[valid_numeric_cols])
            df_imputed_knn_sample = pd.DataFrame(imputed_data, columns=valid_numeric_cols, index=df_sample.index)

            for col in cols_with_missing_values:
                df_knn[f'{col}_imputed_knn'] = df_knn[col].fillna(df_imputed_knn_sample[col])
                print(f"Column '{col}' imputed using KNN (from a sample).")
        else:
            print("No missing values found to impute with KNN.")
    else:
        print("No valid numeric columns available for KNN imputation.")

    # Normalize after imputation
    df_knn = normalize_dataframe(df_knn, numerical_cols)
    output_filename_knn = 'P2_Imputation_Norm/house_prices_imputed_knn.csv'
    df_knn.to_csv(output_filename_knn, index=False)
    print(f"\nSuccessfully created and saved the new CSV file: '{output_filename_knn}'")

    # --- Imputation with Hot Deck (Sequential/Stochastic) ---
    print("\n--- Applying Hot Deck (Sequential) Imputation for numerical columns ---")
    df_hotdeck = houses_data.copy()
    
    for col in numerical_cols:
        if df_hotdeck[col].isnull().sum() > 0:
            available_values = df_hotdeck[col].dropna().values
            if len(available_values) > 0:
                missing_indices = df_hotdeck[df_hotdeck[col].isnull()].index
                imputed_values = np.random.choice(available_values, size=len(missing_indices), replace=True)
                df_hotdeck.loc[missing_indices, f'{col}_imputed_stochastic'] = imputed_values
                print(f"Column '{col}' imputed with Hot Deck using {len(missing_indices)} sampled values.")
            else:
                print(f"Warning: Column '{col}' has only NaN values, cannot apply Hot Deck.")

    # Normalize after imputation
    df_hotdeck = normalize_dataframe(df_hotdeck, numerical_cols)
    output_filename_hotdeck = 'P2_Imputation_Norm/house_prices_imputed_stochastic.csv'
    df_hotdeck.to_csv(output_filename_hotdeck, index=False)
    print(f"\nSuccessfully created and saved the new CSV file: '{output_filename_hotdeck}'")
    
except FileNotFoundError:
    print("Error: The file 'P2_Imputation_Norm/house_prices_mod.csv' was not found. Please ensure the path is correct.\n")
