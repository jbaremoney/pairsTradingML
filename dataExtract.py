'''
Data extraction for cryptos to trade.

Using binance api

'''
import binance.exceptions
from binance.client import Client
import pandas as pd
from datetime import datetime
import time
import requests.exceptions

APIKEY = 'rRI5OHVf1CASnQT9QrUjFwRCgLt0qgertRjh0KDvG04epO2vtVoVlHl7yaB3c5gg'
SECRETKEY = '0uidY3UYIXIir8SBgcJUMAXhH73aXI0spGtEbu0CH9E2rCn8FTDysx5fYOjL8uBf'

client = Client(APIKEY, SECRETKEY, tld="us")


try:
    accountInfo = client.get_account()
    print("successfully fetched account info")
    print("Current account balance:", accountInfo['balances'])
except Exception as e:
    print("Unable to fix because:", e)


def getDataForCoin(symbol, interval, startDate, retries=3, wait_time=5):
    attempt = 0
    while attempt < retries:
        try:
            # Attempt to fetch the historical data
            candles = client.get_historical_klines(symbol, interval, startDate)

            # Process the data if successful
            historicalData = []
            for candle in candles:
                candleData = {
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                }
                historicalData.append(candleData)

            # Return the DataFrame if the request was successful
            return pd.DataFrame(historicalData)

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Handle timeout or connection errors
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            attempt += 1
            time.sleep(wait_time)  # Wait before retrying

    # Raise an exception if all retries failed
    raise Exception(f"Max retries reached for {symbol}. API call failed.")


marketCapCoinsFile = 'top50coinsMarketCap.txt'
top50MCCoins = []
with open(marketCapCoinsFile, 'r') as file:
    for line in file:
        line = line.strip()
        top50MCCoins.append(line)


MCdataDictionary = {}
for symbol in top50MCCoins:
    try:
        df = getDataForCoin(symbol + "USDT", Client.KLINE_INTERVAL_1HOUR, "Jan 1, 2021")
        df.set_index('timestamp', inplace=True)  # Set timestamp as the index
        df.rename(columns={'close': symbol}, inplace=True)  # Rename 'close' column to symbol
        MCdataDictionary[symbol] = df[[symbol]]  # Keep only the symbol column for merging
        print("Done pulling: ", symbol)
        time.sleep(1)
    except binance.exceptions.BinanceAPIException:
        print("invalid symbol:" + symbol)

# Preprocess each DataFrame to ensure unique and clean index
for symbol, df in MCdataDictionary.items():
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Sort by timestamp for consistency
    df.sort_index(inplace=True)

    # Store back in the dictionary
    MCdataDictionary[symbol] = df

# Concatenate all DataFrames
combined_df = pd.concat(MCdataDictionary.values(), axis=1)

combined_df.to_csv("top_50_crypto_data.csv")

# View the combined DataFrame
print(combined_df.head())


