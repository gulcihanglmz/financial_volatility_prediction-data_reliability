import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_and_visualize_data(csv_path):
    """
    1. Descriptive Statistics Table
    2. Correlation Heatmap
    """
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Please check the file path.")
        return
        
    print(f"Loading data: {csv_path}")
    # Read the data by setting the Date column as the index
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    
    # Exclude text columns etc. since we will only do numerical analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # ---------------------------------------------------------
    # 1. DESCRIPTIVE STATISTICS TABLE
    # ---------------------------------------------------------
    print("\n--- 1. Descriptive Statistics Table (Summary) ---")
    statistics = numeric_df.describe().T
    statistics = statistics[['count', 'mean', 'std', 'min', 'max']]
    statistics.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
    
    print(statistics.round(2))
    statistics.round(2).to_csv("thesis_descriptive_statistics.csv")
    print("\n[SUCCESS] Descriptive statistics saved as 'thesis_descriptive_statistics.csv'.")
    
    # ---------------------------------------------------------
    # 2. CORRELATION MATRIX HEATMAP
    # ---------------------------------------------------------
    print("\n--- 2. Generating Correlation Heatmap ---")
    plt.figure(figsize=(12, 10))
    correlation = numeric_df.corr()
    
    # Drawing the heatmap (Most suitable color palette for academic theses: coolwarm)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap Between Variables', fontsize=16, pad=20)
    plt.tight_layout()
    
    plt.savefig("thesis_correlation_heatmap.png", dpi=300)
    print("[SUCCESS] Correlation heatmap saved as 'thesis_correlation_heatmap.png' in high resolution.\n")

if __name__ == "__main__":
    # ENTER YOUR CSV FILE PATH HERE
    analyze_and_visualize_data("D:\\fintech\\data\\nasdq.csv")