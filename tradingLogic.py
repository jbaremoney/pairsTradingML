'''
file for the actual trading logic being executed

when price(coin1, t) - price(coin2, t) differs by too much we short the high one and buy the low one

we will use machine learning to optimize the parameters for exactly when to enter and close positions
DATA STARTS FROM DECEMBER 31, 2020
'''

#imports
import pandas as pd
import joblib
from outputAnalysis import bothGoodDf, justCorrDf, justCointDf

'''
tradingStrat() PARAMETERS
dataCoin1, dataCoin2: time series data for both coins
window=30: Lookback window for calculating rolling mean and st dev
entryThreshold: Z score minimum for entering positions. default 2.0
exitThreshold: Z score for closing position. default 0.5
maxHoldTime: integer number representing days, if a position remains open for longer than this, close position

tradingStrat() RETURNS
signals: dataframe storing the signals generated at each time
zScore: dataframe storing zScores at each time
ratio: dataframe storing ratio of prices at each time
'''

'''using random forest model because it performed the best'''
model = joblib.load("RandomForestModel.pkl")

def filter_data(dataCoin1, dataCoin2, startDate, endDate):
    """Filter both coin datasets by the adjusted start date and end date."""
    dataCoin1.index = pd.to_datetime(dataCoin1.index)
    dataCoin2.index = pd.to_datetime(dataCoin2.index)

    # find the start dates
    start1 = dataCoin1.dropna().index.min()
    start2 = dataCoin2.dropna().index.min()

    # use the later of the two start dates
    adjustedStartDate = max(start1, start2)

    # convert startDate and endDate to Timestamps if they aren't already
    startDate = pd.to_datetime(startDate)
    endDate = pd.to_datetime(endDate)

    # filter the data based on adjusted start date and end date
    if adjustedStartDate > startDate:
        dataCoin1 = dataCoin1[adjustedStartDate:endDate]
        dataCoin2 = dataCoin2[adjustedStartDate:endDate]
    else:
        dataCoin1 = dataCoin1[startDate:endDate]
        dataCoin2 = dataCoin2[startDate:endDate]

    return dataCoin1, dataCoin2, adjustedStartDate



def calculate_zscore(dataCoin1, dataCoin2, window):
    """Calculate the ratio, rolling mean, standard deviation, and z-score."""
    ratio = (dataCoin1 / dataCoin2).dropna()
    ratioMean = ratio.rolling(window=window).mean()
    ratioStd = ratio.rolling(window=window).std()
    zScore = (ratio - ratioMean) / ratioStd
    return ratio, zScore


def generate_signals(zScore, entryThreshold, exitThreshold):
    """Generate long, short, and exit signals based on z-score thresholds."""
    longSignal = (zScore < -entryThreshold)
    shortSignal = (zScore > entryThreshold)
    exitSignal = (abs(zScore) < exitThreshold)

    signals = pd.DataFrame(index=zScore.index)
    signals['long'] = longSignal
    signals['short'] = shortSignal
    signals['exit'] = exitSignal

    return signals


### Main Functions

def tradingStrat(dataCoin1, dataCoin2, window=30, entryThreshold=2.0, exitThreshold=0.5, startDate="2020-01-01",
                 endDate="2024-11-01"):
    """Generate trading signals based on the z-score of the price ratio."""
    # filter the data by date range
    dataCoin1, dataCoin2, adjustedStartDate = filter_data(dataCoin1, dataCoin2, startDate, endDate)

    #Calculate z-score
    ratio, zScore = calculate_zscore(dataCoin1, dataCoin2, window)

    #generate signals
    signals = generate_signals(zScore, entryThreshold, exitThreshold)

    return signals, zScore, ratio


def backtest(pair, dataCoin1, dataCoin2, signals, initialCapital=10000, timeLimit=1000, startDate="2020-01-01",
             endDate="2024-01-01"):
    """Backtest the trading strategy and return the results and final capital."""
    # filter the data by date range
    dataCoin1, dataCoin2, adjustedStartDate = filter_data(dataCoin1, dataCoin2, startDate, endDate)
    signals = signals[adjustedStartDate:endDate]

    # combine and drop NaN values
    combined_df = pd.concat([dataCoin1, dataCoin2, signals], axis=1).dropna()
    if combined_df.empty:
        print("No data available after filtering.")
        return pd.DataFrame(), initialCapital, []

    # separate data again
    dataCoin1 = combined_df.iloc[:, 0]
    dataCoin2 = combined_df.iloc[:, 1]
    signals = combined_df.iloc[:, 2:]

    # initialize variables
    capital = initialCapital
    position = 0
    capitalHistory = []
    tradeReturns = []

    for i in range(1, len(signals)):
        priceCoin1 = dataCoin1.iloc[i]
        priceCoin2 = dataCoin2.iloc[i]
        currentTimeIndex = i

        # enter long position
        if position == 0 and signals['long'].iloc[i]:
            position = 1
            entryPriceCoin1 = priceCoin1
            entryPriceCoin2 = priceCoin2
            investment = 0.25 * capital
            entryTimeIndex = i

        # enter short position
        elif position == 0 and signals['short'].iloc[i]:
            position = -1
            entryPriceCoin1 = priceCoin1
            entryPriceCoin2 = priceCoin2
            investment = 0.25 * capital
            entryTimeIndex = i

        # exit long position
        elif position == 1 and (signals['exit'].iloc[i] or (currentTimeIndex - entryTimeIndex > timeLimit)):
            exitPriceCoin1 = priceCoin1
            exitPriceCoin2 = priceCoin2
            tradeReturn = (exitPriceCoin1 - entryPriceCoin1) / entryPriceCoin1 * investment - \
                          (exitPriceCoin2 - entryPriceCoin2) / entryPriceCoin2 * investment
            capital += tradeReturn
            tradeReturns.append(tradeReturn)
            position = 0

        # exit short position
        elif position == -1 and (signals['exit'].iloc[i] or (currentTimeIndex - entryTimeIndex > timeLimit)):
            exitPriceCoin1 = priceCoin1
            exitPriceCoin2 = priceCoin2
            tradeReturn = (entryPriceCoin2 - exitPriceCoin2) / entryPriceCoin2 * investment - \
                          (entryPriceCoin1 - exitPriceCoin1) / entryPriceCoin1 * investment
            capital += tradeReturn
            tradeReturns.append(tradeReturn)
            position = 0

        capitalHistory.append(capital)

    # create results DataFrame
    results = pd.DataFrame(index=signals.index[1:])
    results['Capital'] = capitalHistory

    return results, capital, tradeReturns



def tradingStratWithModel(dataCoin1, dataCoin2, model, window=30, entryThreshold=2.0, exitThreshold=0.5, maxHoldTime=50, startDate="2021-01001"):
    # Generate trading signals based on z-score
    ratio = (dataCoin1 / dataCoin2).dropna()
    ratioMean = ratio.rolling(window=window).mean()
    ratioStd = ratio.rolling(window=window).std()
    zScore = (ratio - ratioMean) / ratioStd

    longSignal = (zScore < -entryThreshold)
    shortSignal = (zScore > entryThreshold)
    exitSignal = (abs(zScore) < exitThreshold)

    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=ratio.index)
    signals['long'] = longSignal
    signals['short'] = shortSignal
    signals['exit'] = exitSignal

    # Generate features and store their indices
    features_list = []
    feature_indices = []

    for i in range(len(signals)):
        if longSignal.iloc[i] or shortSignal.iloc[i]:
            feature = {
                'entry_zscore': zScore.iloc[i],
                'time_in_position': 0,  # Placeholder, time in position will be tracked during backtesting
                'rolling_mean': ratioMean.iloc[i],
                'rolling_std': ratioStd.iloc[i]
            }
            features_list.append(feature)
            feature_indices.append(signals.index[i])

    # Convert the features list to a DataFrame
    features_df = pd.DataFrame(features_list)

    # Use the model to predict profitable trades
    if not features_df.empty:
        predictions = model.predict(features_df.values)

        # Create a Series with the same index as feature_indices
        predictions_series = pd.Series(predictions, index=feature_indices)
        # Add the predictions to the signals DataFrame (only at the corresponding indices)
        signals['predicted_profitable'] = predictions_series
        # Fill NaN values with False for all other rows
        signals['predicted_profitable'] = signals['predicted_profitable'].fillna(False)

    else:
        signals['predicted_profitable'] = False

    # Filter signals based on model predictions
    signals['long'] = signals['long'] & (signals['predicted_profitable'] == 1)
    signals['short'] = signals['short'] & (signals['predicted_profitable'] == 1)

    return signals, zScore, ratio


# Load the crypto data
df = pd.read_csv('top_50_crypto_data.csv', parse_dates=['timestamp'], index_col='timestamp')

def outputBacktestResults(pairSetDf, string):
    # Define start and end dates
    startDate = '2021-12-31'
    endDate = '2024-11-01'

    # Lists to store results
    results_no_model = []
    results_with_model = []

    # Iterate through the pairs in bothGoodDf
    for i in range(len(pairSetDf)):
        pair = (pairSetDf['Coin1'].iloc[i], pairSetDf['Coin2'].iloc[i])
        print("Processing Pair:", pair)

        # Get price data for the pair
        dataCoin1 = df[pair[0]]
        dataCoin2 = df[pair[1]]

        ### Backtest Without Model
        signals_no_model, _, _ = tradingStrat(dataCoin1, dataCoin2, startDate=startDate, endDate=endDate)
        _, finalCapital_no_model, tradeReturns_no_model = backtest(pair, dataCoin1, dataCoin2, signals_no_model,
                                                                   startDate=startDate, endDate=endDate)
        avg_return_no_model = sum(tradeReturns_no_model) / len(tradeReturns_no_model) if tradeReturns_no_model else 0
        results_no_model.append(
            {'pair': f"{pair[0]}-{pair[1]}", 'final_capital': finalCapital_no_model, 'avg_return': avg_return_no_model})

        ### Backtest With Model
        signals_with_model, _, _ = tradingStratWithModel(dataCoin1, dataCoin2, model, startDate=startDate)
        _, finalCapital_with_model, tradeReturns_with_model = backtest(pair, dataCoin1, dataCoin2, signals_with_model,
                                                                       startDate=startDate, endDate=endDate)
        avg_return_with_model = sum(tradeReturns_with_model) / len(
            tradeReturns_with_model) if tradeReturns_with_model else 0
        results_with_model.append({'pair': f"{pair[0]}-{pair[1]}", 'final_capital': finalCapital_with_model,
                                   'avg_return': avg_return_with_model})

    # Convert results to DataFrames
    results_no_model_df = pd.DataFrame(results_no_model)
    results_with_model_df = pd.DataFrame(results_with_model)

    # Merge the results for comparison
    comparison_df = pd.merge(results_no_model_df, results_with_model_df, on='pair',
                                      suffixes=('_no_model', '_with_model'))

    # Display the comparison


    # Save to CSV for further analysis
    comparison_df.to_csv(f'{string}trading_strategy_comparison.csv', index=False)
    return comparison_df


bothGoodResults = outputBacktestResults(bothGoodDf, "bothGood")
justCorrResults = outputBacktestResults(justCorrDf, "justCorr")
justCointResults = outputBacktestResults(justCointDf, "justCoint")
