# signals/rank_generator.py
import pandas as pd


def generate_ranked_stocks(strategy_df, score_column='Strategy_Score'):
    """
    Generate ranked list of stocks based on strategy score.

    Parameters:
        strategy_df: DataFrame
            Contains tickers and their strategy scores
            e.g. columns = ['Ticker', 'Strategy_Score']
        score_column: str
            Name of the score column to rank by

    Returns:
        DataFrame: ranked stocks with added 'Rank' column
    """
    # Ensure the score column exists
    if score_column not in strategy_df.columns:
        raise ValueError(f"{score_column} not found in strategy DataFrame")

    # Sort by score descending (high score = rank 1)
    ranked_df = strategy_df.sort_values(
        by=score_column, ascending=False).reset_index(drop=True)

    # Assign rank starting from 1
    ranked_df['Rank'] = ranked_df.index + 1

    return ranked_df
