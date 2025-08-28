#Replication optimizer:
#Given ETF returns and stock returns, find weights to minimize tracking error

from __future__ import annotations
import cvxpy as cp 
import pandas as pd 

def replicate_etf(etf_returns: pd.Series, stock_returns: pd.DataFrame, allow_short: bool = False, weight_cap: float | None = None, ridge_lambda: float = 0.0) -> pd.Series:
    aligned = pd.concat([etf_returns.rename("etf"), stock_returns], axis=1).dropna()
    y = aligned["etf"].values  # ETF returns
    X = aligned.drop(columns=["etf"]).values  # Stock returns 
    cols = aligned.drop(columns=["etf"]).columns  # Stock names 
    n = X.shape[1]  # Number of stocks 
    w = cp.Variable(n)  # Weights for each stock 
    
    loss = cp.sum_squares(X @ w - y) + ridge_lambda * cp.sum_squares(w) # Loss function: squared error 
    
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
    if weight_cap is not None:
        constraints.append(w <= weight_cap) 
    
    prob = cp.Problem(cp.Minimize(loss), constraints) 
    prob.solve()  
    
    weights = pd.Series(w.value, index=cols)
    return weights 
    