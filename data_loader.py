import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os

def get_data(tickers, start='2009-01-01', end='2025-01-01'):
    """
    TO_DO: Download stock data for given tickers
    Return a dictionary of ticker -> dataframe 
    """
    stock_data = {}
    for ticker in tickers: 
        df = yf.download(ticker, start=start, end=end) 
        stock_data[ticker] = df
    return stock_data

def save_data_to_csv(stock_data, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    for ticker, df in stock_data.items():  
        df.to_csv(f'{data_dir}/{ticker}.csv')


def load_data_from_csv(tickers, data_dir='data'):
    stock_data = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{data_dir}/{ticker}.csv', skiprows=2, index_col='Date', parse_dates=True)
            # Rename columns to match training data format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            stock_data[ticker] = df
        except Exception as e:
            print(f"Failed to load {ticker}: {str(e)[:50]}")
    return stock_data


def split_data(stock_data, training_range=('2009-01-01', '2019-12-31'), 
                validation_range=('2020-01-01', '2020-12-31'),
                test_range=('2021-01-01', '2025-01-01')):

    training_data = {}
    validation_data = {}
    test_data = {}

    """
    TO_DO: Split stock data into training, validation, and test sets
    Returns: training_data, validation_data, test_data (all dicts of ticker -> dataframe)
    """
    
    for ticker, df in stock_data.items():
        training_data[ticker] = df.loc[training_range[0]:training_range[1]]
        validation_data[ticker] = df.loc[validation_range[0]:validation_range[1]]
        test_data[ticker] = df.loc[test_range[0]:test_range[1]]

    return # TO_DO 

# Technical Indicators
def RSI(df, window=14): 
    """
    Relative Strength Index
    """
    delta = df['Close'].diff()  
    up = delta.where(delta > 0, 0)   
    down = -delta.where(delta < 0, 0)   
    rs = up.rolling(window=window).mean() / down.rolling(window=window).mean()    
    return 100 - 100 / (1 + rs)   


def CCI(df): 
    """
    Commodity Channel Index
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    MA = typical_price.rolling(window=20).mean() 
    mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (typical_price - MA) / (0.015 * mean_deviation) 


def MACD(df): 
    """
    Moving Average Convergence Divergence
    """
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    return ema12 - ema26  


def Signal(df): 
    """
    MACD Signal line
    """
    return MACD(df).ewm(span=9, adjust=False).mean() 


def ADX(df): 
    """
    Average Directional Index
    """
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(span=14, adjust=False).mean() / atr)
    df['-DI'] = 100 * (df['-DM'].ewm(span=14, adjust=False).mean() / atr)
    dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    return dx.ewm(span=14, adjust=False).mean()


def add_technical_indicators(df):  
    """
    Add technical indicators to dataframe
    
    Args:
        df: dataframe with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
        dataframe with technical indicators added
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['RSI'] = RSI(df)
    df['CCI'] = CCI(df)
    df['MACD'] = MACD(df)
    df['Signal'] = Signal(df)
    df['ADX'] = ADX(df)
    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'CCI', 'MACD', 'Signal', 'ADX']]
    return df


def process_data_with_indicators(stock_data):
    """
    Returns: dict of ticker -> dataframe with technical indicators
    """
    processed_data = {}
    # TO_DO: Add technical indicators to all stocks in stock_data
    return processed_data


def save_processed_data(training_data, validation_data, test_data, data_dir='data'):
    """
    Save processed datasets to pickle files
    
    Args:
        training_data: dict of ticker -> dataframe
        validation_data: dict of ticker -> dataframe
        test_data: dict of ticker -> dataframe
        data_dir: directory to save pickle files (default: 'data')
    """
    os.makedirs(data_dir, exist_ok=True)
    with open(f'{data_dir}/training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    with open(f'{data_dir}/validation_data.pkl', 'wb') as f:
        pickle.dump(validation_data, f)
    with open(f'{data_dir}/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)


def load_processed_data(data_dir='data'):
    """
    Load processed datasets from pickle files
    
    Args:
        data_dir: directory containing pickle files (default: 'data')
    
    Returns:
        training_data, validation_data, test_data (all dicts of ticker -> dataframe)
    """
    with open(f'{data_dir}/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)
    with open(f'{data_dir}/validation_data.pkl', 'rb') as f:
        validation_data = pickle.load(f)
    with open(f'{data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return training_data, validation_data, test_data
