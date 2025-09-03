import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress potential warnings from seaborn
warnings.filterwarnings("ignore")

try:
    # Load original and imputed datasets
    original_data = pd.read_csv('P2_Imputation_Norm/house_prices_mod.csv', sep=',')
    imputed_data = pd.read_csv('P2_Imputation_Norm/house_prices_imputed.csv', sep=',')

    print("Datasets loaded successfully. Preparing for visualization.")

    # Convert the price columns to numeric, coercing errors
    original_data['Price'] = pd.to_numeric(original_data['Price (in rupees)'], errors='coerce')
    imputed_data['Price_imputed_mean'] = pd.to_numeric(imputed_data['Price (in rupees)_imputed_mean'], errors='coerce')
    
    # --- Create a single DataFrame for easier plotting ---
    original_prices = original_data['Price'].dropna()
    imputed_prices = imputed_data['Price_imputed_mean']

    df_combined = pd.DataFrame({
        'price': pd.concat([original_prices, imputed_prices]),
        'data_source': ['Original'] * len(original_prices) + ['Imputed'] * len(imputed_prices)
    })
    
    # --- Calculate the medians to be plotted ---
    original_median = original_prices.median()
    imputed_median = imputed_prices.median()
    
    # Create the figure and plot the boxplots
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_combined, y='price', x='data_source', 
                palette={'Original': 'blue', 'Imputed': 'red'})

    # --- Add horizontal lines for the medians ---
    plt.axhline(original_median, color='blue', linestyle='--', linewidth=2, label=f'Original Median: {original_median:.2f}')
    plt.axhline(imputed_median, color='red', linestyle=':', linewidth=2, label=f'Imputed Median: {imputed_median:.2f}')

    plt.title('Comparison of Original vs. Imputed Data Distributions', fontsize=16)
    plt.ylabel('Price (in rupees)')
    plt.xlabel('Data Source')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the figure
    plot_filename = 'price_imputation_boxplot_comparison_with_medians.png'
    plt.savefig(plot_filename)
    print(f"\nBoxplots saved as '{plot_filename}'")
    plt.show()

except FileNotFoundError:
    print("Error: One or both files were not found. Please ensure 'house_prices_mod.csv' and 'house_prices_imputed.csv' are in the specified directory.")