import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress potential warnings from seaborn
warnings.filterwarnings("ignore")

try:
    # --- Load datasets ---
    original_data = pd.read_csv('P2_Imputation_Norm/house_prices_mod.csv', sep=',')
    mean_mode_data = pd.read_csv('P2_Imputation_Norm/house_prices_imputed_mean_mode.csv', sep=',')
    knn_data = pd.read_csv('P2_Imputation_Norm/house_prices_imputed_knn.csv', sep=',')
    stochastic_data = pd.read_csv('P2_Imputation_Norm/house_prices_imputed_stochastic.csv', sep=',')

    print("Datasets loaded successfully. Preparing for visualization.")

    # --- Convert to numeric ---
    original_prices = pd.to_numeric(original_data['Price (in rupees)'], errors='coerce').dropna()
    mean_mode_prices = pd.to_numeric(mean_mode_data['Price (in rupees)_imputed_mean'], errors='coerce')
    knn_prices = pd.to_numeric(knn_data['Price (in rupees)_imputed_knn'], errors='coerce')
    stochastic_prices = pd.to_numeric(stochastic_data['Price (in rupees)_imputed_stochastic'], errors='coerce')

    # --- Build combined DataFrame ---
    df_combined = pd.DataFrame({
        'price': pd.concat(
            [original_prices, mean_mode_prices, knn_prices, stochastic_prices],
            ignore_index=True
        ),
        'data_source': (['Original'] * len(original_prices)) +
                       (['Mean/Mode'] * len(mean_mode_prices)) +
                       (['KNN'] * len(knn_prices)) +
                       (['Stochastic'] * len(stochastic_prices))
    })

    # --- Calculate quartiles (Q1, Q2, Q3) ---
    quartiles = df_combined.groupby("data_source")['price'].quantile([0.25, 0.5, 0.75]).unstack()

    # --- Plot boxplots ---
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df_combined, y='price', x='data_source',
                palette={'Original': 'blue', 'Mean/Mode': 'green', 'KNN': 'orange', 'Stochastic': 'red'})

    # Add horizontal lines for quartiles
    colors = {'Original': 'blue', 'Mean/Mode': 'green', 'KNN': 'orange', 'Stochastic': 'red'}
    for source in quartiles.index:
        q1, q2, q3 = quartiles.loc[source]
        plt.axhline(q1, color=colors[source], linestyle=':', linewidth=1.5,
                    label=f'{source} Q1: {q1:.2f}')
        plt.axhline(q2, color=colors[source], linestyle='--', linewidth=2,
                    label=f'{source} Median: {q2:.2f}')
        plt.axhline(q3, color=colors[source], linestyle='-.', linewidth=1.5,
                    label=f'{source} Q3: {q3:.2f}')

    plt.title('Comparison of Original vs. Different Imputation Methods\nwith Quartile Lines', fontsize=16)
    plt.ylabel('Price (in rupees)')
    plt.xlabel('Data Source')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save figure
    plot_filename = 'price_imputation_boxplot_comparison_with_quartiles.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"\nBoxplots saved as '{plot_filename}'")
    plt.show()

except FileNotFoundError:
    print("Error: One or more files were not found. Please ensure all CSV files are in the specified directory.")
