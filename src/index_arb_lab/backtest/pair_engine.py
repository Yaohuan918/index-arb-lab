from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from src.index_arb_lab.data.yahoo_adapter import YahooAdapter
from src.index_arb_lab.replication.optimizer import replicate_etf
from src.index_arb_lab.backtest.utils import month_end_dates, rolling_zscore, drawdown_curve

@dataclass
class BTConfig:
    etf: str
    constituents: list[str]
    start: str 
    end: str
    lookback_days: int
    z_lookback: int
    entry: float
    exit: float
    notional: float
    allow_short: bool
    weight_cap: float | None
    ridge_lambda: float
    one_way_bps: float
    out_dir: str
    trades_csv: str
    perf_csv: str
    pnl_plot: str
    dd_plot: str
    
    target_ann_vol: float = 0.10
    max_dd: float = 0.05
    time_stop_days: int = 15
    
    adf_lookback: int = 120
    adf_pval: float = 0.05
    adf_close_on_fail: bool = False 
    
    
class PairBacktester:
    def __init__(self, cfg: BTConfig):
        self.cfg = cfg 
        self.ya = YahooAdapter()
        
    def load_prices(self) -> pd.DataFrame:
        bars_etf = self.ya.get_bars(self.cfg.etf, self.cfg.start, self.cfg.end)

        if "Adj Close" in bars_etf.columns:
            etf_ser = bars_etf[["Adj Close"]].squeeze()   # 1-col DF -> Series
        else:
            etf_ser = bars_etf[["Close"]].squeeze()
        etf = etf_ser.rename(self.cfg.etf)                # rename Series -> column name after concat

        stock_series = []
        for s in self.cfg.constituents:
            bars_s = self.ya.get_bars(s, self.cfg.start, self.cfg.end)
            if "Adj Close" in bars_s.columns:
                s_ser = bars_s[["Adj Close"]].squeeze()
            else:
                s_ser = bars_s[["Close"]].squeeze()
            stock_series.append(s_ser.rename(s))          

        stocks = pd.concat(stock_series, axis=1)          # DataFrame with ticker columns
        df = pd.concat([etf, stocks], axis=1).dropna()    # align on common dates
        return df



    def fit_weights_monthly(self, ret: pd.DataFrame) -> dict:
        rebs = month_end_dates(ret.index)
        schedule = {}
        for reb in rebs:
            window = ret.loc[:reb].tail(self.cfg.lookback_days)
            if len(window) < max(10, self.cfg.lookback_days // 3):
                continue
            etf_in = window[self.cfg.etf]
            stk_in = window.drop(columns=[self.cfg.etf])
            w = replicate_etf(
                etf_returns=etf_in,
                stock_returns=stk_in,
                allow_short=self.cfg.allow_short,
                weight_cap=self.cfg.weight_cap,
                ridge_lambda=self.cfg.ridge_lambda
            )
            schedule[pd.Timestamp(reb)] = w
        return schedule  # We use this function fit_weights_monthly to get weights at each month end rebalance date, and do the rebalance, as use the replicate_etf to re-calculate the weights
    
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg                      # local alias
        df = self.load_prices()             # df is our price data
        ret = df.pct_change().dropna()      # daily returns
        etf_ret = ret[cfg.etf]              # etf returns
        stock_ret = ret.drop(columns=[cfg.etf])
        
        weights_schedule = self.fit_weights_monthly(ret)  # rebalance schedule
        rep_ret = pd.Series(index=ret.index, dtype=float) # replication returns
        current_w = None # current weights
        reb_dates = sorted(weights_schedule.keys())  # rebalance dates
        
        for i, reb in enumerate(reb_dates):
            current_w = weights_schedule[reb]   # update current weights
            start_idx = ret.index.get_indexer([reb], method='nearest')[0]   # start from the rebalance date
            stop_idx = (ret.index.get_indexer([reb_dates[i+1]], method='nearest')[0] if i + 1 < len(reb_dates) else None)  # up to next rebalance date (exclusive)
            oos_idx = ret.index[start_idx + 1 : stop_idx]   # out-of-sample slice
            if len(oos_idx) == 0:   
                continue
            rep_ret.loc[oos_idx] = (stock_ret.loc[oos_idx] * current_w).sum(axis=1)   # replication returns in the out-of-sample period
            
        rep_ret = rep_ret.ffill().bfill()   # fill any NaN values
        
        #Curves for spread/z-score
        etf_curve = (1 + etf_ret).cumprod()
        rep_curve = (1 + rep_ret).cumprod()
        spread = np.log(etf_curve) - np.log(rep_curve) # log spread
        spread_ret = spread.diff().dropna()  # spread returns
        roll_sig = spread_ret.rolling(60).std().reindex(ret.index).ffill().clip(lower=1e-6)  # rolling volatility of spread returns
        z = rolling_zscore(spread, cfg.z_lookback)  # rolling z-score of the spread
        
        target_ann_vol = getattr(self.cfg, "target_ann_vol", 0.10) if hasattr(self.cfg, "target_ann_vol") else 0.10
        vol_per_day = target_ann_vol / math.sqrt(252)
        size_scale = (vol_per_day / roll_sig).clip(upper=3.0)  # cap max size scale to avoid extreme leverage
        
        from statsmodels.tsa.stattools import adfuller
        
        # Evaluate ADF once on each rebalance date
        adf_p = pd.Series(index=reb_dates, dtype=float)
        
        for d in sorted(weights_schedule.keys()):
            win = spread.loc[:d].tail(cfg.adf_lookback).dropna()
            if len(win) >= int(cfg.adf_lookback * 0.6):
                try:
                    pval = adfuller(win, autolag="AIC")[1]
                except Exception:
                    pval = np.nan
            else:
                pval = np.nan
            adf_p.loc[d] = pval
            
        adf_ok = (adf_p <= cfg.adf_pval).reindex(ret.index).ffill().fillna(False)

        # daily loop: position & PnL
        pos = 0                # -1, 0, +1
        etf_shares = 0.0
        stock_shares = pd.Series(0.0, index=stock_ret.columns)
        
        
        equity = pd.Series(index=ret.index, dtype=float)
        equity.iloc[0] = 1.0 
        peak_equity = equity.iloc[0]
        
        prev_z = None
        days_in_trade = 0
        reb_dates = sorted(weights_schedule.keys())
        if not reb_dates:
            raise RuntimeError("No rebalance dates")
        
        
        trades = []
        pos = 0
        etf_shares = 0.0
        stock_shares = pd.Series(0.0, index=stock_ret.columns)
        equity = pd.Series(index=ret.index, dtype=float)
        equity.iloc[0] = 1.0
        peak_equity = float(equity.iloc[0])
        prev_z = None
        days_in_trade = 0
        reb_dates = sorted(weights_schedule.keys())
        if not reb_dates:
            raise RuntimeError("No rebalance dates")

        for t in ret.index[:-1]:
            next_t = ret.index[ret.index.get_loc(t) + 1]
            
            # weights in effect at t 
            current_w = weights_schedule[max([d for d in reb_dates if d <= t], default=reb_dates[0])]
            # signal at close of t
            z_t = z.get(t, np.nan)
            target = pos
            
            #update peak and drawdown
            peak_equity = max(peak_equity, float(equity.loc[t]))
            dd = equity.loc[t] / peak_equity - 1.0
            
            #circuit breaker: if DD > 5%, flatten
            if dd <= -cfg.max_dd and pos != 0:
                target = 0
            else:
                # Only change position if we have a defined signal
                if not np.isnan(z_t):
                    # entry on crossing entry bands
                    if pos == 0 and abs(z_t) >= cfg.entry:
                        target = -1 if z_t >= cfg.entry else +1
                    else:
                        if prev_z is not None:
                            if (prev_z < cfg.entry <= z_t):
                                target = -1  # cross above upper entry band
                            elif (prev_z > -cfg.entry >= z_t):
                                target = +1  # cross below lower entry band
                    
                    # exit band
                    if abs(z_t) <= cfg.exit:
                        target = 0
                        
                    # time stop
                    if pos != 0 and days_in_trade >= cfg.time_stop_days:
                        target = 0
                    
                    # stationarity gate(ADF)
                    # Block opening new positions if the spread is not stationary
                    if pos == 0 and target in (-1, +1) and not adf_ok.loc[t]:
                        target = 0
                    # Optionally close existing positions when stationarity fails
                    if pos != 0 and cfg.adf_close_on_fail and not adf_ok.loc[t]:
                        target = 0
                
            # rebalance if target changed
            if target != pos:
                # compute desired shares for each leg at close(t)
                px_etf = df.at[t, cfg.etf]  # ETF price at time t
                px_stk = df.loc[t, stock_ret.columns]  # stock prices at time t
                
                # notional for this trade interval, with vol targeting
                size_t = float(size_scale.loc[t])
                dyn_notional = cfg.notional * size_t
                
                target_etf_shares = (target * dyn_notional) / px_etf   # target shares of ETF to hold
                target_stock_shares = (-target * dyn_notional) * (current_w / px_stk) # target shares of each stock to hold
                
                
                # order sizes = target - current
                # Delta = target minus current holdings, That's my child orders at time t
                etf_delta = target_etf_shares - etf_shares
                stock_delta = (target_stock_shares - stock_shares).fillna(0.0)
                
                # simple bps cost on turnover notional
                turn_notional = abs(etf_delta) * px_etf + (abs(stock_delta) * px_stk).sum()
                cost = (2.0 * cfg.one_way_bps / 1e4) * turn_notional   # round-trip
                
                # update positions
                etf_shares = target_etf_shares
                stock_shares = target_stock_shares
                pos = target
                days_in_trade = 0  # reset holding clock
                
                trades.append({
                    "date": t,
                    "z": float(z_t),
                    "target": target,
                    "etf_delta_shares": float(etf_delta),
                    "basket_turnover_notional": float(turn_notional),
                    "cost": float(cost),
                })
                
                # book cost against equity immediately (as return in units of equity)
                equity.loc[t] = equity.loc[t] - cost / max(1e-12, dyn_notional)
            else:
                # still in the same position -> increment holding days
                if pos != 0:
                    days_in_trade += 1
            # hold returns into t+1
            # hold returns into t+1 using ACTIVE RETURN (no share math)
            size_t = float(size_scale.loc[t])
            if pos == 0:
                active_ret = 0.0
            else:
                etf_r = etf_ret.loc[next_t]
                basket_r = (stock_ret.loc[next_t] * current_w).sum()
                active_ret = pos * size_t * (etf_r - basket_r)
                
            equity.loc[next_t] = equity.loc[t] * (1.0 + active_ret)
            prev_z = z_t    # remember for cross logic
            
        equity = equity.ffill().bfill()
        # outputs
        trades_df = pd.DataFrame(trades).set_index("date") if trades else pd.DataFrame(columns=["date"])
        perf = pd.DataFrame({
            "equity": equity,
            "drawdown": drawdown_curve(equity),
            "zscore": z.reindex(equity.index),
            "spread": spread.reindex(equity.index),
        })
        
        # save
        out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        trades_path = out / cfg.trades_csv
        perf_path = out / cfg.perf_csv
        perf.to_csv(perf_path)
        trades_df.to_csv(trades_path)
        
        # plots
        fig = plt.figure()
        perf["equity"].plot(title="Pair Backtest - Equity Curve")
        plt.xlabel("Date"); plt.ylabel("Equity"); plt.tight_layout()
        fig.savefig(out / cfg.pnl_plot, dpi=140); plt.close(fig)
        
        fig2 = plt.figure()
        (100*perf["drawdown"]).plot(title="Pair Backtest - Drawdown (%)")
        plt.xlabel("Date"); plt.ylabel("Drawdown (%)"); plt.tight_layout()
        fig2.savefig(out / cfg.dd_plot, dpi=140); plt.close(fig2)
        
        return trades_df, perf