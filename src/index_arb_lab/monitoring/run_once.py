from __future__ import annotations 
import argparse 
from datetime import datetime, timedelta 
from pathlib import Path 

import pandas as pd 
import matplotlib.pyplot as plt 

from src.index_arb_lab.utils.config import load_config 
from src.index_arb_lab.utils.logging_setup import setup_logging 
from src.index_arb_lab.data.yahoo_adapter import YahooAdapter 
from src.index_arb_lab.index.fair_value import time_to_expiry_yr, fair_value 
from src.index_arb_lab.basis.compute import rolling_zscore 

def main():
    parser = argparse.ArgumentParser(description="Run index")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args() 
    
    cfg = load_config(args.config) 
    logger = setup_logging(cfg.log_path) 
    
    # Date range: enough history to compute rolling z-scores
    end = datetime.utcnow().date()
    start = end - timedelta(days=max(365, cfg.lookback_days + 30))
    
    # (1) Fetch data
    ya = YahooAdapter()
    cash = ya.get_bars(cfg.cash_symbol, start.isoformat(), end.isoformat())["Close"]
    fut = ya.get_bars(cfg.futures_symbol, start.isoformat(), end.isoformat())["Close"]
    cash = pd.Series(cash.squeeze(), name="cash")
    fut = pd.Series(fut.squeeze(), name="fut")
    
    # Align on common dates (important: cash and futures calendars differ)
    df = pd.concat([cash, fut], axis=1).dropna()
    
    # (2) Compute theoretical futures from cash, day by day (time-to-expiry changes daily)
    f_theo = []
    for ts, row in df.iterrows():
        T = time_to_expiry_yr(ts, cfg.expiry_dt) # years to expiry 
        f_theo.append(fair_value(row["cash"], cfg.annual_rate, cfg.annual_div_yield, T))
    df["f_theo"] = pd.Series(f_theo, index=df.index)
    
    # (3) Compute basis and z-scores 
    df["basis"] = df["fut"] - df["f_theo"] 
    df["zscore"] = rolling_zscore(df["basis"], cfg.lookback_days)
    
    # (4) Latest snapshot + simple alerting 
    latest = df.dropna().iloc[-1]
    if abs(latest["zscore"]) >= cfg.z_entry:
        # Rule-of-thumb: if futures > theoretical (positive basis), short fut/long cash 
        side = "SHORT futures / LONG cash" if latest["zscore"] > 0 else "LONG futures / SHORT cash"
        logger.info(f"ENTRY signal: |z|={latest['zscore']:.2f} ({side})")
    elif abs(latest["zscore"]) <= cfg.z_exit:
        logger.info(f"EXIT signal zone for any open position (|z| small)")
    
    logger.info(f"cash={latest['cash']:.2f} fut={latest['fut']:.2f} "
                f"f_theo={latest['f_theo']:.2f} basis={latest['basis']:.2f} "
                f"z={latest['zscore']:.2f}")
    
    # (5) Save a daily CSV snapshot and plot 
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(cfg.out_dir) / f"basis_daily_{end.isoformat()}.csv"
    latest[["cash", "fut", "f_theo", "basis", "zscore"]].to_frame().T.to_csv(out_csv, index_label="date")
    
    fig = plt.figure() # one chart per figure (simple and clean)
    df["basis"].tail(cfg.lookback_days).plot(title="Basis (F_mkt - F_theo)")
    plt.xlabel("Date") 
    plt.ylabel("Index Points")
    plot_path = Path(cfg.out_dir) / cfg.plot_filename 
    plt.tight_layout() 
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    
    logger.info(f"Wrote CSV: {out_csv}") 
    logger.info(f"Wrote plot: {plot_path}")
    
if __name__ == "__main__":
    main()


# THe whole pipeline fits together:
# 1. Config -> symbols, expiry, rates, thresholds, paths
# 2. Data -> cash & futures closes(aligned)
# 3. Model -> F_theo(t) = St * exp((r - d) * T) 
# 4. Basis -> F_mkt(t) - F_theo(t) 
# 5. Z-score -> detect rich/cheap relative to history 
# 6. Signal -> log entry/exit suggestion 
# 7. Persist -> daily CSV + chart for monitoring/alerts
