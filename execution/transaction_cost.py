# execution/transaction_costs.py
def calc_commission(order_value: float, per_trade: float = 1.0, bps: float = 0.0) -> float:
    """
    Commission as flat fee + basis points of notional.
    """
    return per_trade + (bps / 10_000.0) * abs(order_value)


def apply_slippage(mid_price: float, side: str, slippage_bps: float = 5.0) -> float:
    """
    Shift fill price by slippage (bps). Buy -> worse (higher), Sell -> worse (lower).
    """
    shift = (slippage_bps / 10_000.0) * mid_price
    return mid_price + shift if side.upper() == "BUY" else mid_price - shift
