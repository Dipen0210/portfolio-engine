# signals_generation/signal_logger.py
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "logs/trade_signals_log.csv"


def log_signals(signals_df):
    """
    Append generated signals (BUY, SELL, REBALANCE, HOLD) to log file.
    """
    os.makedirs("logs", exist_ok=True)
    signals_df = signals_df.copy()
    if 'Timestamp' not in signals_df.columns:
        signals_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        combined = pd.concat([old, signals_df], ignore_index=True)
    else:
        combined = signals_df

    combined.to_csv(LOG_PATH, index=False)
