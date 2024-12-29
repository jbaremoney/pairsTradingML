'''
Testing each currency to see cointegrated pairs to trade

job id 115388

'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('top_50_crypto_data.csv') #creating df

# Convert 'timestamp' to datetime and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# testing for correlation using Pearson
pearsSufficientPairsList = []
cointSufficientPairsList = []
pairsList = []

PEARSCORRTHRESHOLD = .75
corrResults =[]
def corrTesting():
    for i, coin1 in enumerate(df.columns):
        for j, coin2 in enumerate(df.columns):
            if i<j:
                # only want times where both coins have data
                filtered_df = df[[coin1, coin2]].dropna()

                if len(filtered_df) > 0:  # Ensure there is data to process
                    # Pearson Correlation
                    pearsCorrCoeff = filtered_df[coin1].corr(filtered_df[coin2], method='pearson')

                    # Cointegration Test (Engle-Granger)
                    X = sm.add_constant(filtered_df[coin2])
                    model = sm.OLS(filtered_df[coin1], X).fit()
                    residuals = model.resid

                    # Perform ADF test on residuals
                    adf_test = adfuller(residuals, regresults=True)

                    # Store results (correlation coefficient and ADF test stats)
                    corrResults.append((coin1, coin2, pearsCorrCoeff, adf_test[0], adf_test[1]))

    return corrResults

print(corrTesting())






