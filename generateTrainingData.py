'''
making training data to train classification model
'''

import pandas as pd
import numpy as np
from tradingLogic import tradingStrat
from outputAnalysis import bothGoodDf, justCorrDf, justCointDf, totalResultsDf

def generate_training_data(dataCoin1, dataCoin2, signals, window=30, maxHoldTime=50):
    features = []
    labels = []
    tradeReturns = []

    # Calculate rolling statistics and z-score
    ratio = (dataCoin1 / dataCoin2).dropna()
    ratioMean = ratio.rolling(window=window).mean()
    ratioStd = ratio.rolling(window=window).std()
    zScore = (ratio - ratioMean) / ratioStd

    # Iterate through signals to extract features and labels
    position = 0
    for i in range(1, len(signals)):
        if position == 0:
            if signals['long'].iloc[i]:
                entryPrice1 = dataCoin1.iloc[i]
                entryPrice2 = dataCoin2.iloc[i]
                entryZScore = zScore.iloc[i]
                position = 1
                entryIndex = i

            elif signals['short'].iloc[i]:
                entryPrice1 = dataCoin1.iloc[i]
                entryPrice2 = dataCoin2.iloc[i]
                entryZScore = zScore.iloc[i]
                position = -1
                entryIndex = i

        elif position != 0:
            # Track how long the position is held
            timeInPosition = i - entryIndex

            # Close position based on maxHoldTime or exit signal
            if signals['exit'].iloc[i] or timeInPosition >= maxHoldTime:
                exitPrice1 = dataCoin1.iloc[i]
                exitPrice2 = dataCoin2.iloc[i]

                # Calculate trade return
                if position == 1:
                    tradeReturn = (exitPrice1 - entryPrice1) / entryPrice1 - (exitPrice2 - entryPrice2) / entryPrice2
                else:
                    tradeReturn = (entryPrice2 - exitPrice2) / entryPrice2 - (entryPrice1 - exitPrice1) / entryPrice1

                # Create features for this trade
                feature = {
                    'entry_zscore': entryZScore,
                    'time_in_position': timeInPosition,
                    'price_ratio': entryPrice1 / entryPrice2,
                    'rolling_mean': ratioMean.iloc[entryIndex],
                    'rolling_std': ratioStd.iloc[entryIndex],
                }
                features.append(feature)
                labels.append(1 if tradeReturn > 0 else 0)  # 1 for profit, 0 for loss
                tradeReturns.append(tradeReturn)

                # Reset position
                position = 0

    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    labels_df = pd.Series(labels, name='label')

    return features_df, labels_df, tradeReturns


df = pd.read_csv('top_50_crypto_data.csv', index_col='timestamp')

def save_to_csv(all_features, all_labels, filename):
    """
    Combines features and labels, then saves the resulting DataFrame to a CSV file.

    Parameters:
    - all_features: list of DataFrames containing features for each pair.
    - all_labels: list of Series containing labels for each pair.
    - filename: string, the name of the CSV file to save the data.
    """
    # Check if there are any features and labels to save
    if all_features and all_labels:
        # Combine all features and labels into DataFrames
        features_df = pd.concat(all_features, ignore_index=True)
        labels_df = pd.concat(all_labels, ignore_index=True)

        # Add the labels as a new column in the features DataFrame
        features_df['label'] = labels_df

        # Save the combined DataFrame to a CSV file
        features_df.to_csv(filename, index=False)
        print(f"Saved training data to {filename}")
    else:
        print("No data to save.")


# Concatenate all features and labels into single DataFrame
all_features = []
all_labels =[]

def generateBothGoodTrainingData():
    # output csv
    for i in range(1, len(bothGoodDf)):
        pair = (bothGoodDf['Coin1'].iloc[i], bothGoodDf['Coin2'].iloc[i])
        print("Processing Pair:", pair)

        # Get price data for the pair
        dataCoin1 = df[pair[0]]
        dataCoin2 = df[pair[1]]

        # Generate trading signals
        signals, zScore, ratio = tradingStrat(dataCoin1, dataCoin2)

        # Generate features and labels
        features, labels, tradeReturns = generate_training_data(dataCoin1, dataCoin2, signals)

        # Add pair metadata to the features DataFrame
        features['pair'] = f"{pair[0]}-{pair[1]}"

        # Append to the combined list
        all_features.append(features)
        all_labels.append(labels)
        save_to_csv(all_features, all_labels, "both_good_training_data.csv")


def generateJustCointTrainingData():
    for i in range(1, len(justCointDf)):
        pair = (justCointDf['Coin1'].iloc[i], justCointDf['Coin2'].iloc[i])
        print("Processing Pair:", pair)

        # Get price data for the pair
        dataCoin1 = df[pair[0]]
        dataCoin2 = df[pair[1]]

        # Generate trading signals
        signals, zScore, ratio = tradingStrat(dataCoin1, dataCoin2)

        # Generate features and labels
        features, labels, tradeReturns = generate_training_data(dataCoin1, dataCoin2, signals)

        # Add pair metadata to the features DataFrame
        features['pair'] = f"{pair[0]}-{pair[1]}"

        # Append to the combined list
        all_features.append(features)
        all_labels.append(labels)
        save_to_csv(all_features, all_labels, "just_coint_training_data.csv")


def generateJustCorrTrainingData():
    for i in range(1, len(justCorrDf)):
        pair = (justCorrDf['Coin1'].iloc[i], justCorrDf['Coin2'].iloc[i])
        print("Processing Pair:", pair)

        # Get price data for the pair
        dataCoin1 = df[pair[0]]
        dataCoin2 = df[pair[1]]

        # Generate trading signals
        signals, zScore, ratio = tradingStrat(dataCoin1, dataCoin2)

        # Generate features and labels
        features, labels, tradeReturns = generate_training_data(dataCoin1, dataCoin2, signals)

        # Add pair metadata to the features DataFrame
        features['pair'] = f"{pair[0]}-{pair[1]}"

        # Append to the combined list
        all_features.append(features)
        all_labels.append(labels)
        save_to_csv(all_features, all_labels, "training_data.csv")

