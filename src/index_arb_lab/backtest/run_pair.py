from __future__ import annotations
import argparse, yaml
from datetime import date, timedelta
from dataclasses import dataclass
from src.index_arb_lab.backtest.pair_engine import BTConfig, PairBacktester


def _parse_date(s: str) -> str:
    if isinstance(s, str) and s.lower() == "today":
        return (date.today() - timedelta(days=1)).isoformat()
    return s
def load_cfg(path: str) -> BTConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    b, o, e, r = raw["backtest"], raw["optimizer"], raw["execution"], raw["reporting"]
    risk = raw.get("risk", {})
    stat = raw.get("stationarity", {})
    return BTConfig(
        etf=b["etf"], constituents=list(b["constituents"]),
        start=_parse_date(b["start"]), end=_parse_date(b["end"]), lookback_days=int(b["lookback_days"]),
        z_lookback=int(b["z_lookback"]), entry=float(b["entry"]), exit=float(b["exit"]),
        notional=float(b["notional"]), allow_short=bool(o["allow_short"]),
        weight_cap=None if o["weight_cap"] in (None, "None") else float(o["weight_cap"]),
        ridge_lambda=float(o["ridge_lambda"]), 
        one_way_bps=float(e["one_way_bps"]),
        out_dir=r["out_dir"], 
        trades_csv=r["trades_csv"],
        perf_csv=r["perf_csv"],
        pnl_plot=r["pnl_plot"],
        dd_plot=r["dd_plot"],
        
        target_ann_vol=float(risk.get("target_ann_vol")),
        max_dd=float(risk.get("max_dd")),
        time_stop_days=int(risk.get("time_stop_days")),
        
        adf_lookback=int(stat.get("adf_lookback")),
        adf_pval=float(stat.get("adf_pval")),
        adf_close_on_fail=bool(stat.get("adf_close_on_fail")),
    )

def main():
    ap = argparse.ArgumentParser(description="Pair Backtest: fetch -> backtest -> save.")
    ap.add_argument("--config", required=True, help="Path to configs/pair_backtest.yaml")
    args = ap.parse_args()
    
    cfg = load_cfg(args.config)
    bt = PairBacktester(cfg)
    trades, perf = bt.run()
    print("Backtest complete.")
    print(f"Trades saved to: {cfg.out_dir}/{cfg.trades_csv}")
    print(f"Performance saved to: {cfg.out_dir}/{cfg.perf_csv}")
    print(f"PNL plot saved to: {cfg.out_dir}/{cfg.pnl_plot}")
    
if __name__ == "__main__":
    main()