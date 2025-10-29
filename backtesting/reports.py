# backtesting/reports.py
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(portfolio_df):
    """
    Plot cumulative portfolio value over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_df.index,
             portfolio_df['Portfolio Value'], label='Portfolio Value', linewidth=2)
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_backtest_report(metrics):
    """
    Print a formatted summary of performance metrics.
    """
    print("\n=== Backtest Performance Summary ===\n")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")
