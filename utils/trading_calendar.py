from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

US_HOLIDAY_CAL = USFederalHolidayCalendar()


def is_us_business_day(ts) -> bool:
    """
    Return True when the timestamp falls on a U.S. business day (excludes weekends/holidays).
    """
    normalized = pd.Timestamp(ts).normalize()
    if normalized.weekday() >= 5:
        return False
    holidays = US_HOLIDAY_CAL.holidays(start=normalized, end=normalized)
    return normalized not in holidays


def next_trading_day(start_dt) -> datetime:
    """
    Advance to the next available U.S. trading day on or after the supplied datetime.
    """
    dt = pd.Timestamp(start_dt).normalize()
    guard = 0
    while not is_us_business_day(dt):
        dt += timedelta(days=1)
        guard += 1
        if guard > 366:
            raise ValueError("Unable to resolve the next trading day within one year.")
    return dt.to_pydatetime()


def previous_trading_day(start_dt) -> datetime:
    """
    Roll backward to the prior U.S. trading day before the supplied datetime.
    """
    dt = pd.Timestamp(start_dt).normalize() - timedelta(days=1)
    guard = 0
    while not is_us_business_day(dt):
        dt -= timedelta(days=1)
        guard += 1
        if guard > 366:
            raise ValueError("Unable to resolve the previous trading day within one year.")
    return dt.to_pydatetime()
