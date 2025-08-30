from __future__ import annotations
import argparse, yaml, math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.index_arb_lab.data.yahoo_adapter import YahooAdapter
from src.index_arb_lab.replication.optimizer import replicate_etf
from src.index_arb_lab.backtest.utils import month_end_dates
from .broker import PaperBroker, CostSpec
from .models import Order, Position

@dataclass
class ExecCfg:
    start: str
    end: str
    base_notional: float
    use_vol_target: bool
    vol_lookback: int
    vol_target_ann: float
    vol_floor: float
    vol_cap: float
    one_way_bps: float
    slip_bps: float
    out_dir: str
    paper_csv: str
    paper_plot: str
    read_trades_csv: str
    
def _parse_date(s: str) -> str:
    if isinstance(s, str) and s.lower() == "today":
        return (date.today() - timedelta(days=1)).isoformat()
    return s

def load_cfg(path: str) -> ExecCfg:
    raw = yaml.safe_load(open(path, "r"))
    e = raw["execution"]
    return ExecCfg(
        start = _parse_date(e["start"]),
        end = _parse_date(e["end"]),
        base_notional = float(e["base_notional"]),
        use_vol_target = bool(e.get("use_vol_target", True)),
        vol_lookback=int(e.get("vol_lookback", 60)),
        vol_target_ann=float(e.get("vol_target_ann", 0.10)),
        vol_floor=float(e.get("vol_floor", 0.5)),
        vol_cap=float(e.get("vol_cap", 2.0)),
        one_way_bps=float(e["one_way_bps"]),
        slip_bps=float(e["slip_bps"]),
        out_dir=e["out_dir"],
        paper_csv=e["paper_csv"],
        paper_plot=e["paper_plot"],
        read_trades_csv=e["read_trades_csv"],
    )
    
def load_prices(etf: str, stocks: list[str], start: str, end: str) -> pd.DataFrame:
    ya = YahooAdapter()
    def px(sym):
        df = ya.get_bars(sym, start, end)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return df[[col]].squeeze().rename(sym)
    cols = [px(etf)] + [px(s) for s in stocks]
    return pd.concat(cols, axis=1).dropna()

def fit_weights_monthly(df_ret: pd.DataFrame, etf: str, allow_short: bool, weight_cap: float | None, ridge_lambda: float, lookback_days: int) -> dict[pd.Timestamp, pd.Series]:
    rebs = month_end_dates(df_ret.index)
    sched = {}
    for d in rebs:
        win = df_ret.loc[:d].tail(lookback_days)
        if len(win) < max(10, lookback_days // 3):
            continue
        w = replicate_etf(
            etf_returns=win[etf],
            stock_returns=win.drop(columns=[etf]),
            allow_short=allow_short,
            weight_cap=weight_cap,
            ridge_lambda=ridge_lambda,
        )
        sched[pd.Timestamp(d)] = w
    return sched

def main():
    ap = argparse.ArgumentParser(description="Paper trade runner for index-arb pair engine.")
    ap.add_argument("--config", required=True, help="configs/paper_execution.yaml")
    ap.add_argument("--etf", default="SPY")
    ap.add_argument("--constituents", nargs="+", default=["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "COST", "LLY", "JPM"])
    ap.add_argument("--allow_short", action="store_true", default=False)
    ap.add_argument("--weight_cap", type=float, default=0.40)
    ap.add_argument("--ridge_lambda", type=float, default=0.01)
    ap.add_argument("--rep_lookback_days", type=int, default=180)
    args = ap.parse_args()
    
    cfg = load_cfg(args.config)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)

    # (1) Data
    df = load_prices(args.etf, args.constituents, cfg.start, cfg.end)
    ret = df.pct_change().dropna()
    etf = args.etf
    stocks = [c for c in df.columns if c != etf]
    
    # (2) Weights schedule (monthly)
    weights_sched = fit_weights_monthly(ret, etf, args.allow_short, args.weight_cap, args.ridge_lambda, args.rep_lookback_days)
    reb_dates = sorted(weights_sched.keys())
    if not reb_dates:
        raise RuntimeError("No rebalance dates; widen your date window")
    
    # (3) Build daily position (+1/0/-1) from your backtester trades
    trades_path = Path(cfg.read_trades_csv)
    if not trades_path.exists():
        raise FileNotFoundError(f"Could not find {trades_path}. Run your backtest first.")
    tdf = pd.read_csv(trades_path, parse_dates=["date"]).set_index("date").sort_index()
    pos = pd.Series(0, index=ret.index)
    for t, row in tdf.iterrows():
        if t in pos.index:
            pos.loc[t:] = int(row["target"])
    # Now pos is the *desired* sign each day
    
    # (4) Vol targeting scale (like your backtest)
    etf_curve = (1 + ret[etf]).cumprod()
    # build a replication curve using weights in force
    rep_ret = pd.Series(index=ret.index, dtype=float)
    for i, d in enumerate(reb_dates):
        w = weights_sched[d]
        start_idx = ret.index.get_indexer([d], method="nearest")[0]
        stop_idx = (ret.index.get_indexer([reb_dates[i+1]], method="nearest")[0] if i+1 <len(reb_dates) else None)
        oos = ret.index[start_idx+1: stop_idx]
        rep_ret.loc[oos] = (ret.loc[oos, stocks] @ w)
    rep_ret = rep_ret.ffill().bfill()
    rep_curve = (1 + rep_ret).cumprod()
    spread = np.log(etf_curve) - np.log(rep_curve)
    spread_ret = spread.diff().dropna()
    roll_sig = spread_ret.rolling(cfg.vol_lookback).std().reindex(ret.index).ffill().clip(lower=1e-6)
    vol_per_day = cfg.vol_target_ann / math.sqrt(252)
    size_scale = (vol_per_day / roll_sig) if cfg.use_vol_target else pd.Series(1.0, index=ret.index)
    size_scale = size_scale.clip(lower=cfg.vol_floor, upper=cfg.vol_cap)
    
    # (5) Sim loop with the broker - MOC fills
    broker = PaperBroker(cash=cfg.base_notional, cost_spec=CostSpec(cfg.one_way_bps, cfg.slip_bps))
    nav = pd.Series(index=ret.index, dtype=float)
    nav.iloc[0] = 1.0       # start at $1.0 equity
    
    # helper: weights in force at date t
    def w_at(t: pd.Timestamp) -> pd.Series:
        d = max([d for d in reb_dates if d <= t], default=reb_dates[0])
        return weights_sched[d]
    
    for t in ret.index[:-1]:
        next_t = ret.index[ret.index.get_loc(t)+1]
        
        dyn = cfg.base_notional * float(size_scale.loc[t])
        target = int(pos.loc[t])       # -1/0/+1
        
        px_t = df.loc[t]
        w = w_at(t)
        
        etf_notional = target * dyn
        rep_notional = -target * dyn
        target_etf_sh = etf_notional / float(px_t[etf])
        target_basket_sh = (rep_notional * w) / px_t[stocks]
        
        cur_etf_sh = broker.positions.get(etf, Position(etf, 0.0)).shares
        cur_stock_sh = pd.Series({s: broker.positions.get(s, Position(s, 0.0)).shares for s in stocks})
        
        etf_delta = float(target_etf_sh - cur_etf_sh)
        bask_delta = (target_basket_sh - cur_stock_sh).fillna(0.0)
        
        if abs(etf_delta) > 1e-8:
            etf_side = 1 if etf_delta > 0 else -1
            broker.place_order(
                t,
                Order(ts=t, symbol=etf, side=etf_side, shares=abs(etf_delta))
            )
        for s, q in bask_delta.items():
            if abs(q) > 1e-8:
                side = 1 if q > 0 else -1
                broker.place_order(
                    t,
                    Order(ts=t, symbol=s, side=side, shares=abs(q))
                )

        broker.end_of_day_fill(t, df.loc[t])
        
        port_val_next = broker.value(df.loc[next_t])
        nav.loc[next_t] = port_val_next / cfg.base_notional
        
    nav = nav.ffill()
    
    out_csv = out / cfg.paper_csv
    out_plot = out / cfg.paper_plot
    pd.DataFrame({"equity": nav}).to_csv(out_csv)
    
    fig = plt.figure()
    nav.plot(title="Paper Trading Equity (Index starting at 1.0)")
    plt.xlabel("Date"); plt.ylabel("Equity"); plt.tight_layout()
    fig.savefig(out_plot, dpi=140); plt.close(fig)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_plot}")

if __name__ == "__main__":
    main()
        