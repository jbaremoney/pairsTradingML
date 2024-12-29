'''
making dataframe via cleanCorrResults.csv, doing stuff with it
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

totalResultsDf = pd.read_csv('cleanCorrResults.csv')

def plot_corr_heatmap(data, title):
    heatmap_data = data.pivot(index='Coin1', columns='Coin2', values='Correlation')
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()


def plot_pVal_heatmap(data, title):
    p_value_matrix = data.pivot(index='Coin1', columns='Coin2', values='p-value')
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_value_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, cbar_kws={'label': 'p-value'})

    plt.title(title)
    plt.show()


midpoint = len(totalResultsDf)//2
dfPart1 = totalResultsDf[:midpoint]
dfPart2 = totalResultsDf[midpoint:]



plot_corr_heatmap(dfPart1, "1st half df correlation")
plot_corr_heatmap(dfPart2, "2nd half df correlation")

plot_pVal_heatmap(dfPart1, "1st half of df cointegration")


# this has a bunch of stablecoins in it so gonna filter those out
'''
Gonna have 3 classes, cointegrated pairs, and correlated pairs, and both pairs

presumably the both pairs will perform the best but will see
'''

# filtering out the stable coins
filteredDf = totalResultsDf.query("Coin1 != 'USDC' and Coin2 != 'USDC' and Coin1 != 'DAI' and Coin2 != 'DAI'")

# getting just the good correlation bad coint ... (use backticks around hyphenated key)
justCorrDf = filteredDf.query("abs(Correlation) >= .75 and `p-value` >= .051") #76 pairs

#good coint bad corr
justCointDf = filteredDf.query("abs(Correlation) < .75 and `p-value` <= .05") #33 pairs

# good both
bothGoodDf = filteredDf.query("abs(Correlation) >=.75 and `p-value` <= .05") #87 pairs



