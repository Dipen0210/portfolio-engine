# optimization/hybrid_allocator.py

import pandas as pd


def allocate_hybrid_weights(rank_df, opt_weights, alpha=0.6):
    """
    Combine strategy rank-based weights with optimized weights.
    w* = α * w_rank + (1 - α) * w_opt
    """
    rank_df = rank_df.copy()
    rank_df['Rank_Weight'] = (rank_df['Rank'].max() - rank_df['Rank'] + 1)
    rank_df['Rank_Weight'] /= rank_df['Rank_Weight'].sum()

    w_rank = rank_df.set_index('Ticker')['Rank_Weight']
    w_opt = opt_weights.reindex(w_rank.index).fillna(0)

    w_final = alpha * w_rank + (1 - alpha) * w_opt
    w_final /= w_final.sum()

    return pd.DataFrame({
        'Ticker': w_final.index,
        'Final_Weight': w_final.values,
        'Rank_Weight': w_rank.values,
        'Opt_Weight': w_opt.values
    })
