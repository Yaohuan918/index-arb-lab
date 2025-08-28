from __future__ import annotations 
import argparse 
from dataclasses import dataclass 
from datetime import date, datetime, timedelta 
from pathlib import Path 

import pandas as pd 
import matplotlib.pyplot as plt 
import yaml  

from src.index_arb_lab.data.yahoo_adapter import YahooAdapter 
from src.index_arb_lab.replication.optimizer import replicate_etf 
from src.index_arb_lab.replication.tracking import tracking_error, drift 

@dataclass 
class ReplicationConfig:
    etf_symbol: str
    constituents: list[str]
    start: str 
    end: str 
    allow_short: bool
    weight_cap: float | None 
    out_dir: str 
    weights_csv_prefix: str 
    perf_csv_prefix: str 
    overlay_plot: str 
    active_plot: str 
    
def _parse_date(s: str) -> str: 
    if s.lower() == "today":
        return date.today().isoformat()
    return s 

def load_replication_config(path: str) -> ReplicationConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) 
    
    rep = raw["replication"] 
    win = raw["window"] 
    opt = raw["optimizer"] 
    repcfg = raw["reporting"] 
    
    return ReplicationConfig(
        etf_symbol=rep["etf_symbol"],
        constituents=list(rep["constituents"]),
        start=_parse_date(win["start"]),
        end=_parse_date(win["end"]),
        allow_short=bool(opt["allow_short"]),
        weight_cap=None if opt["weight_cap"] in (None, "None") else float(opt["weight_cap"]),
        out_dir=repcfg["out_dir"],
        weights_csv_prefix=repcfg.get("weights_csv_prefix", "replication_weights"),
        perf_csv_prefix= repcfg.get("perf_csv_prefix", "replication_perf"),
        overlay_plot=repcfg.get("overlay_plot", "replication_overlay.png"),
        active_plot=repcfg.get("active_plot", "replication_active_returns.png"),
    )
    
def main():
    ap = argparse.ArgumentParser(description="Replication & TE: fetch -> optimize -> evaluate -> save.")
    ap.add_argument("--config", required=True, help="Path to configs/replication.yaml")
    args = ap.parse_args() 
        
    cfg = load_replication_config(args.config) 
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        
    # (1) Fetch Adj Close (split/dividend adjusted)
    ya = YahooAdapter()
    etf = ya.get_bars(cfg.etf_symbol, cfg.start, cfg.end)[["Adj Close"]].squeeze("columns")
    etf.name = cfg.etf_symbol
    stocks = []
    for s in cfg.constituents:
        s_df = ya.get_bars(s, cfg.start, cfg.end)[["Adj Close"]].squeeze("columns")
        s_df.name = s
        stocks.append(s_df) 
    stocks = pd.concat(stocks, axis=1)
        
    # Align all series 
    df = pd.concat([etf, stocks], axis=1).dropna() 
        
    # (2) Compute daily returns 
    ret = df.pct_change().dropna()
    etf_ret = ret[cfg.etf_symbol] 
    stocks_ret = ret.drop(columns=[cfg.etf_symbol]) 
        
    # (3) Optimize weights 
    w = replicate_etf(
        etf_returns=etf_ret,
        stock_returns=stocks_ret,
        allow_short=cfg.allow_short,
        weight_cap=cfg.weight_cap
    )
        
    # (4) Reconstruct replication and evaluate 
    replica_ret = (stocks_ret * w).sum(axis=1) 
    # Turn daily returns into index levels (start at 1.0)
    etf_curve = (1 + etf_ret).cumprod() 
    rep_curve = (1 + replica_ret).cumprod() 
        
    te = tracking_error(replica_ret, etf_ret) 
    end_drift = drift(rep_curve, etf_curve) 
        
    # (5) Save artifacts 
    today = date.today().isoformat() 
    weights_csv = out_dir / f"{cfg.weights_csv_prefix}_{today}.csv"
    perf_csv = out_dir / f"{cfg.perf_csv_prefix}_{today}.csv" 
        
    w.round(6).to_frame("weight").to_csv(weights_csv)
    perf = pd.DataFrame({
        "date": rep_curve.index,
        "etf_curve": etf_curve.values,
        "rep_curve": rep_curve.values,
        "active_ret": (replica_ret - etf_ret).values,
    }).set_index("date")
    perf.to_csv(perf_csv) 
        
    # Overlay plot 
    fig = plt.figure() 
    etf_curve.plot(label=f"{cfg.etf_symbol}")
    rep_curve.plot(label="Replication") 
    plt.title(f"Replication vs ETF (TE={te:.4f}, Drift={end_drift:.4f})")
    plt.xlabel("Date"); plt.ylabel("Index Level (start=1)")
    plt.legend(); plt.tight_layout() 
    overlay_plot_path = out_dir / cfg.overlay_plot 
    fig.savefig(overlay_plot_path, dpi=140); plt.close(fig) 
        
    # Active returns plot 
    fig2 = plt.figure()
    perf["active_ret"].cumsum().plot(title="Cumulative Active Returns")
    plt.xlabel("Date"); plt.ylabel("Cumulative Returns")
    plt.tight_layout()
    active_path = out_dir / cfg.active_plot
    fig2.savefig(active_path, dpi=140); plt.close(fig2)
        
    # Console summary 
    print(f"Optimal weights:\n{w.sort_values(ascending=False)}")
    print(f"\nTracking Error (daily stdev of active): {te:.6f}")
    print(f"End Drift (rep - etf): {end_drift:.6f}")
    print(f"Wrote: {weights_csv}")
    print(f"Wrote: {perf_csv}")
    print(f"Wrote: {overlay_plot_path}")
    print(f"Wrote: {active_path}")
        
if __name__ == "__main__":
        main()
            
        