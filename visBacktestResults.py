import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files for the two strategies
both_good_df = pd.read_csv("bothGoodtrading_strategy_comparison.csv")
just_coint_df = pd.read_csv("justCointtrading_strategy_comparison.csv")
just_corr_df = pd.read_csv("justCorrtrading_strategy_comparison.csv")

# Add a 'group' column to identify each dataset
both_good_df['group'] = 'Both Good'
just_coint_df['group'] = 'Just Coint'
just_corr_df['group'] = 'Just Corr'

# Display the first few rows to verify the data
print("Both Good Data:")
print(both_good_df.head())

print("\nJust Coint Data:")
print(just_coint_df.head())

print("\nJust Corr Data:")
print(just_corr_df.head())

# Combine all three DataFrames into a single DataFrame for analysis
combined_df = pd.concat([both_good_df, just_coint_df, just_corr_df], ignore_index=True)

# Display the combined DataFrame
print("\nCombined Data:")
print(combined_df.head())

# Summary Statistics Function
def display_summary_stats(df):
    summary = df.groupby('group').agg({
        'final_capital_no_model': ['mean', 'std'],
        'avg_return_no_model': ['mean', 'std'],
        'final_capital_with_model': ['mean', 'std'],
        'avg_return_with_model': ['mean', 'std']
    })
    print("\nSummary Statistics:")
    print(summary)

# Display the summary statistics
display_summary_stats(combined_df)

# Visualization Functions
# Final Capital Comparison Plot
def plot_final_capital(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='group', y='final_capital_no_model', errorbar=None, label='No Model')
    sns.barplot(data=df, x='group', y='final_capital_with_model', errorbar=None, label='With Model', alpha=0.7)
    plt.title('Final Capital Comparison by Group')
    plt.ylabel('Final Capital ($)')
    plt.xlabel('Group')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Average Return Comparison Plot
def plot_average_return(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='group', y='avg_return_no_model', errorbar=None, label='No Model')
    sns.barplot(data=df, x='group', y='avg_return_with_model', errorbar=None, label='With Model', alpha=0.7)
    plt.title('Average Return Comparison by Group')
    plt.ylabel('Average Return')
    plt.xlabel('Group')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Boxplots for Final Capital and Average Return
def plot_boxplots(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='group', y='final_capital_no_model')
    sns.boxplot(data=df, x='group', y='final_capital_with_model')
    plt.title('Distribution of Final Capital by Group')
    plt.ylabel('Final Capital ($)')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='group', y='avg_return_no_model')
    sns.boxplot(data=df, x='group', y='avg_return_with_model')
    plt.title('Distribution of Average Return by Group')
    plt.ylabel('Average Return')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.show()

# Call the plotting functions
plot_final_capital(combined_df)
plot_average_return(combined_df)
plot_boxplots(combined_df)


# Save the summary statistics to a new CSV file
def save_summary_stats(df, filename="combined_strategy_summary.csv"):
    summary = df.groupby('group').agg({
        'final_capital_no_model': ['mean', 'std'],
        'avg_return_no_model': ['mean', 'std'],
        'final_capital_with_model': ['mean', 'std'],
        'avg_return_with_model': ['mean', 'std']
    })
    summary.to_csv(filename)
    print(f"\nSummary statistics saved to {filename}")

# Save the summary statistics
save_summary_stats(combined_df)