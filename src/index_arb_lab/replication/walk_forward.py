from __future__ import annotations
import argparse 
from dataclasses import dataclass 
from datetime import date, datetime
from pathlib import Path 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml 

from src.index_arb_lab.data.yahoo_adapter import YahooAdapter
from src.index_arb_lab.replication.optimizer import replicate_etf
from src.index_arb_lab.replication.tracking import tracking_error, drift, turnover

@dataclass
class WalkForwardConfig:
    etf_symbol: str
    constituents: list[str]
    start: str 
    end: str 
    lookback_days: int
    allow_short: bool
    weight_cap: float | None 
    freq: str
    one_way_bps: float
    out_dir: str 
    weights_csv_prefix: str 
    perf_csv_prefix: str 
    overlay_plot: str 
    active_plot: str
    ridge_lambda: float
    report_gross: bool
    
def _parse_date(s: str) -> str:
    return date.today().isoformat() if isinstance(s, str) and s.lower() == "today" else s

def load_config(path: str) -> WalkForwardConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
        
    
    rep, win, opt, costs, repcfg = (
        raw["replication"],
        raw["window"],
        raw["optimizer"],
        raw["costs"],
        raw["reporting"]
    )
    return WalkForwardConfig(
        etf_symbol=rep["etf_symbol"],
        constituents=list(rep["constituents"]),
        start=_parse_date(win["start"]),
        end=_parse_date(win["end"]),
        lookback_days=int(win["lookback_days"]),
        allow_short=bool(opt["allow_short"]),
        weight_cap=None if opt["weight_cap"] in (None, "None") else float(opt["weight_cap"]),
        freq=rep.get("frequency", "monthly"),
        one_way_bps=float(costs.get("one_way_bps", 0.0)),
        out_dir=repcfg["out_dir"],
        weights_csv_prefix=repcfg.get("weights_csv_prefix", "replication_weights"),
        perf_csv_prefix=repcfg.get("perf_csv_prefix", "replication_perf"),
        overlay_plot=repcfg.get("overlay_plot", "replication_overlay.png"),
        active_plot=repcfg.get("active_plot", "replication_active.png"),
        ridge_lambda=float(opt.get("ridge_lambda", 0.0)),
        report_gross=bool(costs.get("report_gross", False)),
    )
    
    
def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_series().groupby(idx.to_period("M")).agg("last").values

def apply_tc_bps_to_first_day(ret: pd.Series, tc_bps: float) -> pd.Series:
    if len(ret) == 0:
        return ret 
    hit = (tc_bps / 1e4) # bps -> fraction
    ret = ret.copy() 
    ret.iloc[0] -= hit  
    return ret

def main():
    ap = argparse.ArgumentParser(description="Monthly walk-forward replication: fetch -> optimize -> evaluate -> save.")
    ap.add_argument("--config", required=True, help="Path to configs/replication.yaml")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # (1) Data (Adjusted Close for total-return style)
    ya = YahooAdapter()
    etf_px = ya.get_bars(cfg.etf_symbol, cfg.start, cfg.end)[["Adj Close"]].squeeze("columns")
    etf_px.name = cfg.etf_symbol
    stock_px = pd.concat(
        [ya.get_bars(s, cfg.start, cfg.end)[["Adj Close"]].squeeze("columns").rename(s) for s in cfg.constituents],
        axis=1
    )
    df = pd.concat([etf_px, stock_px], axis=1).dropna()
    ret = df.pct_change().dropna() 
    etf_ret = ret[cfg.etf_symbol]
    stock_ret = ret.drop(columns=[cfg.etf_symbol]) 
    
    # (2) Rebalance schedule (month-ends)
    rebs = month_end_dates(ret.index) 
    if len(rebs) == 0:
        raise RuntimeError("No month ends found in the requested window")
    
    # (3) Curves & bookkeeping
    first_ts = ret.index[0]
    port_curve_gross = pd.Series(index=ret.index, dtype=float)  # gross returns
    port_curve_net = pd.Series(index=ret.index, dtype=float)  # net returns after transaction costs
    port_curve_gross.iloc[0] = 1.0  # start at 1.0
    port_curve_net.iloc[0] = 1.0  # start at 1.0
    
    total_cost_bps = 0.0
    turnovers = []
    weights_hist: dict[pd.Timestamp, pd.Series] = {}
    w_prev: pd.Series | None = None  # previous weights for turnover calc
    
    # (4) For each rebalance date: fit on *prior* lookback_days, hold for next month 005
    for i, reb in enumerate(rebs):
        # Fit window: strictly before the rebalance date
        in_sample = ret.loc[:reb].tail(cfg.lookback_days)
        if len(in_sample) < max(10, cfg.lookback_days // 3):
            # not enough history yes
            continue
        
        etf_in = in_sample[cfg.etf_symbol]
        stk_in = in_sample.drop(columns=[cfg.etf_symbol])
        
        # Solve weights
        w = replicate_etf(
            etf_returns=etf_in,
            stock_returns=stk_in,
            allow_short=cfg.allow_short,
            weight_cap=cfg.weight_cap,
            ridge_lambda=cfg.ridge_lambda,
        )
        weights_hist[pd.Timestamp(reb)] = w
        
        # Turnover + transaction costs (charged on rebalance)
        tc_bps = 0.0 
        if w_prev is not None:
            tv = turnover(w, w_prev)    # 0.5*sum..
            turnovers.append(tv)
            tc_bps = (2.0 * cfg.one_way_bps) * tv    # round-trip = 2 * one-way * turnover
            total_cost_bps += tc_bps
        w_prev = w
        
        # 00S period: from next trading day after reb up to the next rebalance date (exclusive)
        start_idx = ret.index.get_indexer([reb], method="nearest")[0]
        stop_idx = (ret.index.get_indexer([rebs[i+1]], method="nearest")[0] if i + 1 < len(rebs) else None)
        oos_slice = ret.index[start_idx + 1 : stop_idx]
        oos_stock = stock_ret.loc[oos_slice]
        # Replication 00S returns (constant weights)
        rep_oos_gross = (oos_stock * w).sum(axis=1)
        rep_oss_net = apply_tc_bps_to_first_day(rep_oos_gross, tc_bps)
        
        # Apply transaction cost as a return hit on first day of 00S
        for t, (rg, rn) in zip(rep_oos_gross.index, zip(rep_oos_gross, rep_oss_net)):
            last_g = port_curve_gross.loc[port_curve_gross.last_valid_index()]
            last_n = port_curve_net.loc[port_curve_net.last_valid_index()]
            port_curve_gross.loc[t] = last_g * (1 + rg)
            port_curve_net.loc[t] = last_n * (1 + rn)
            
    # Forward-fill any initial NaNs from the first assigned point 
    port_curve_gross = port_curve_gross.ffill().bfill()
    port_curve_net = port_curve_net.ffill().bfill()
    
    # (5) Metrics(00S)
    etf_curve = (1 + etf_ret).cumprod()
    active_gross = port_curve_gross.pct_change().fillna(0.0) - etf_ret.loc[port_curve_gross.index].fillna(0.0)
    active_net = port_curve_net.pct_change().fillna(0.0) - etf_ret.loc[port_curve_net.index].fillna(0.0)
    
    # daily stdev of active returns
    TE_gross = np.std(active_gross.dropna(), ddof=1)
    TE_net = np.std(active_net.dropna(), ddof=1)
    drift_gross = port_curve_gross.iloc[-1] - etf_curve.loc[port_curve_gross.index].iloc[-1]
    drift_net = port_curve_net.iloc[-1] - etf_curve.loc[port_curve_net.index].iloc[-1]
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    
    # (6) Save artifacts 
    today = date.today().isoformat() 
    weights_csv = out_dir / f"{cfg.weights_csv_prefix}_{today}.csv"
    perf_csv = out_dir / f"{cfg.perf_csv_prefix}_{today}.csv"
    
    pd.DataFrame(weights_hist).T.sort_index().to_csv(weights_csv)
    
    perf = pd.DataFrame({
        "etf_curve": etf_curve.loc[port_curve_net.index].values,
        "rep_curve_gross": port_curve_gross.values,
        "rep_curve_net": port_curve_net.values,
        "active_gross": active_gross.values,
        "active_net": active_net.values,
    }, index=port_curve_net.index)
    perf.to_csv(perf_csv)
    
    # Overlay plot (curves)
    fig = plt.figure()
    etf_curve.loc[port_curve_net.index].plot(label=f"{cfg.etf_symbol}")
    port_curve_net.plot(label="Replication (WF, net)")
    if cfg.report_gross:
        port_curve_gross.plot(label="Replication (WF, gross)")
    
    plt.title(f"WF replication vs ETF |  TE gross={TE_gross:.4f},TE net={TE_net:.4f} | "
              f"avg T0={avg_turnover:.3f}, total cost={total_cost_bps:.1f} bps")
    plt.xlabel("Date"); plt.ylabel("Index Level (start=1)"); plt.legend(); plt.tight_layout()
    overlay_path = out_dir / ("wf_" + cfg.overlay_plot)
    fig.savefig(overlay_path, dpi=140); plt.close(fig)
    
    # Active (cumulative)
    fig2 = plt.figure()
    active_net.cumsum().plot(title="Cumulative Active Returns (WF net)")
    plt.xlabel("Date"); plt.ylabel("Cumulative Returns"); plt.tight_layout()
    active_path = out_dir / ("wf_" + cfg.active_plot)
    fig2.savefig(active_path, dpi=140); plt.close(fig2)
    
    print("Walk-forward replication completed.")
    print(f"TE gross={TE_gross:.6f}  TE net={TE_net:.6f}")
    print(f"Drift gross={drift_gross:.6f}  Drift net={drift_net:.6f}")
    print(f"Avg turnover per rebalance={avg_turnover:.4f}")
    print(f"Total costs charged (sum over rebalances) ≈ {total_cost_bps:.1f} bps")

    print(f"Wrote weights: {weights_csv}")
    print(f"Wrote performance: {perf_csv}")
    print(f"Wrote overlay plot: {overlay_path}")
    print(f"Wrote active returns plot: {active_path}")
    
if __name__ == "__main__":
    main()