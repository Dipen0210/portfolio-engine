# portfolio/portfolio_summary.py

import pandas as pd


def summarize_portfolio(portfolio):
    """
    Create a clean summary DataFrame for reporting or dashboard display.
    """

    summary = {
        "Expected Return (daily)": [portfolio["Expected_Return"]],
        "Volatility (daily)": [portfolio["Volatility"]],
        "Sharpe Ratio": [portfolio["Sharpe"]],
    }

    summary_df = pd.DataFrame(summary)

    if portfolio["Sector_Allocation"] is not None:
        sector_df = portfolio["Sector_Allocation"].reset_index()
        sector_df.columns = ["Sector", "Total Weight"]
        return summary_df, sector_df

    return summary_df, None
