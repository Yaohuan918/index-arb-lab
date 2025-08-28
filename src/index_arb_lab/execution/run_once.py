from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import yaml
import pandas as pd  

from src.index_arb_lab.execution.simulator import ExecutionSimulator, BasketItem, ExecParams

@dataclass
class ExecConfig:
    basket: list[dict]
    data_start: str
    data_end: str
    adv_window: int 
    style: str
    n_slices: int
    pov: float
    spread_bps: float
    impact_alpha: float
    impact_beta: float
    seed: int | None
    out_dir: str 
    summary_csv_prefix: str 
    slices_csv_prefix: str 
    overlay_plot: str
    is_plot: str  
    
def _parse_date(s: str) -> str:
    return date.today().isoformat() if isinstance(s, str) and s.lower() == "today" else s

def load_config(path: str) -> ExecConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
        
    data = raw["data"]; exe = raw["execution"]; rep = raw["reporting"]
    return ExecConfig(
        basket=list(raw["basket"]),
        data_start=_parse_date(data["start"]),
        data_end=_parse_date(data["end"]),
        adv_window=int(data["adv_window"]),
        style=exe["style"],
        n_slices=int(exe["n_slices"]),
        pov=float(exe["pov"]),
        spread_bps=float(exe["spread_bps"]),
        impact_alpha=float(exe["impact_alpha"]),
        impact_beta=float(exe["impact_beta"]),
        seed=int(exe.get("seed", None)),
        out_dir=rep["out_dir"],
        summary_csv_prefix=rep["summary_csv_prefix"],
        slices_csv_prefix=rep["slices_csv_prefix"],
        overlay_plot=rep["overlay_plot"],
        is_plot=rep["is_plot"],
    )
    
def main():
    ap = argparse.ArgumentParser(description="Execution Simulator: fetch -> simulate -> save.")
    ap.add_argument("--config", required=True, help="Path to configs/execution.yaml")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    sim = ExecutionSimulator(adv_window=cfg.adv_window)
    params = ExecParams(
        style=cfg.style,
        n_slices=cfg.n_slices,
        pov=cfg.pov,
        spread_bps=cfg.spread_bps,
        impact_alpha=cfg.impact_alpha,
        impact_beta=cfg.impact_beta,
        seed=cfg.seed
    )
    
    basket = [BasketItem(**b) for b in cfg.basket]
    summary, slices = sim.simulate_day(
        basket=basket,
        data_start=cfg.data_start,
        data_end=cfg.data_end,
        params=params,
        out_dir=cfg.out_dir,
        overlay_plot=cfg.overlay_plot,
        is_plot=cfg.is_plot
    )
    
    today = date.today().isoformat()
    summary_path = Path(cfg.out_dir) / f"{cfg.summary_csv_prefix}_{today}.csv"
    slices_path = Path(cfg.out_dir) / f"{cfg.slices_csv_prefix}_{today}.csv"

    summary.to_csv(summary_path, index=False)
    slices.to_csv(slices_path, index=False)
    
    print("Execution simulation complete")
    print(summary)
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {slices_path}")
    print(f"Wrote plots to {cfg.out_dir}")
    
if __name__ == "__main__":
    main()