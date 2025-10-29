import pandas as pd
import numpy as np

def compute_expected_returns(stock_data_dict, window=60):
    """
    Compute expected (forecasted) returns for each stock using rolling mean of log returns.

    Parameters
    ----------
    stock_data_dict : dict
        Dictionary of {ticker: DataFrame} with OHLCV data for each stock.
        Each DataFrame must have a 'Close' column and be sorted by date.
    window : int
        Rolling window length (in trading days) for computing expected return.
        e.g. 60 ≈ 3 months, 20 ≈ 1 month.

    Returns
    -------
    expected_returns_df : pd.DataFrame
        Columns: ['Ticker', 'Expected_Return']
    expected_returns_series : pd.Series
        Indexed by Ticker for use in optimization (μ vector)
    """

    results = []

    for ticker, df in stock_data_dict.items():
        # Skip if not enough data
        if 'Close' not in df.columns or len(df) < window:
            continue

        if 'Date' in df.columns:
            working = df.sort_values('Date').copy()
        else:
            working = df.sort_index().copy()

        # Compute log returns
        working['Log_Return'] = np.log(working['Close'] / working['Close'].shift(1))

        # Rolling mean expected return
        working['Rolling_Mean_Return'] = working['Log_Return'].rolling(window=window).mean()

        # Latest expected return value
        expected_return = working['Rolling_Mean_Return'].iloc[-1]

        results.append({'Ticker': ticker, 'Expected_Return': expected_return})

    expected_returns_df = pd.DataFrame(results).dropna()

    # Convert to Series for optimization
    expected_returns_series = expected_returns_df.set_index('Ticker')['Expected_Return']

    return expected_returns_df, expected_returns_series
