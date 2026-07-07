# ── Imports ──────────────────────────────────────────────────
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi']        = 120
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── Configuration (change here to re-run for any ticker) ─────
TICKER      = 'TSLA'
ALL_TICKERS = ['TSLA', 'BND', 'SPY']   
START_DATE  = '2015-01-01'
END_DATE    = '2026-06-30'
TRAIN_END   = '2024-12-31'             

print(f'Ticker        : {TICKER}')
print(f'Portfolio     : {ALL_TICKERS}')
print(f'Full period   : {START_DATE} → {END_DATE}')
print(f'Training ends : {TRAIN_END}')
print('-' * 40)

# ── Data Loader ──────────────────────────────────────────────
def fetch_and_save_data(tickers, start, end, save_dir='data/raw'):
    """
    Fetches historical data for given tickers from Yahoo Finance 
    and saves them as CSVs in the specified directory.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure tickers is a list for iteration
    if isinstance(tickers, str):
        tickers = [tickers]
        
    fetched_data = {}
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Download data using yfinance
            df = yf.download(ticker, start=start, end=end, progress=False)
            
            if df.empty:
                print(f"  ↳ Warning: No data found for {ticker}.")
                continue
                
            # Define file path and save to CSV
            file_path = os.path.join(save_dir, f"{ticker}_historical.csv")
            df.to_csv(file_path)
            print(f"  ↳ Successfully saved to {file_path}")
            
            # Store in dictionary to return
            fetched_data[ticker] = df
            
        except Exception as e:
            print(f"  ↳ Error fetching {ticker}: {e}")
            
    return fetched_data

if __name__ == "__main__":
    # Execute the function for the entire portfolio
    portfolio_dataframes = fetch_and_save_data(
        tickers=ALL_TICKERS, 
        start=START_DATE, 
        end=END_DATE
    )