from __future__ import annotations
import numpy as np
import pandas as pd

def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_series().groupby(idx.to_period("M")).agg("last").values   

def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    m = series.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    s = series.rolling(lookback, min_periods=max(5, lookback // 3)).std(ddof=1)
    z = (series - m) / s.replace(0, np.nan)
    return z

def drawdown_curve(curve: pd.Series) -> pd.Series:
    cummax = curve.cummax()
    return (curve / cummax) - 1.0