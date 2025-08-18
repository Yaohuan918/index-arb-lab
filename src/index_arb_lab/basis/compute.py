from __future__ import annotations 
import numpy as np 
import pandas as pd 


def compute_basis_series(f_mkt: pd.Series, f_theo: pd.Series) -> pd.Series: 
    aligned = pd.concat([f_mkt.rename("f_mkt"), f_theo.rename("f_theo")], axis=1).dropna()
    aligned["basis"] = aligned["f_mkt"] - aligned["f_theo"]
    return aligned["basis"]

def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series: 
    roll_mean = series.rolling(lookback, min_periods=max(2, lookback // 3)).mean()
    roll_std = series.rolling(lookback, min_periods=max(2, lookback // 3)).std(ddof=1)
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z 