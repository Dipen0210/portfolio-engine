import pandas as pd
import numpy as np


def SMA(df, period=20, column='Close'):
    """Simple Moving Average"""
    df[f'SMA_{period}'] = df[column].rolling(window=period).mean()
    return df


def EMA(df, period=20, column='Close'):
    """Exponential Moving Average"""
    df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
    return df


def RSI(df, period=14, column='Close'):
    """Relative Strength Index"""
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df


def MACD(df, short=12, long=26, signal=9, column='Close'):
    """MACD Indicator"""
    df['EMA_short'] = df[column].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df[column].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df


def Bollinger_Bands(df, period=20, column='Close'):
    """Bollinger Bands"""
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    df['BB_upper'] = sma + (2 * std)
    df['BB_lower'] = sma - (2 * std)
    return df
