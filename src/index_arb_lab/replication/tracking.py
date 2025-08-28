import pandas as pd 
import numpy as np 

def tracking_error(portfolio: pd.Series, etf: pd.Series) -> float: 
    active = portfolio - etf 
    return np.std(active, ddof=1) 

def drift(portfolio: pd.Series, etf: pd.Series) -> float:
    return portfolio.iloc[-1] - etf.iloc[-1] 

def turnover(weights_t: pd.Series, weights_prev: pd.Series) -> float: 
    return 0.5 * np.sum(np.abs(weights_t - weights_prev))
