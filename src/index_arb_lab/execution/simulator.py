from __future__ import annotations 
from dataclasses import dataclass 
from datetime import date, timedelta, datetime 
from pathlib import Path 
from typing import List, Dict, Tuple 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from src.index_arb_lab.data.yahoo_adapter import YahooAdapter
from src.index_arb_lab.execution.volume_curves import u_shape_curve

@dataclass 
class BasketItem:
    symbol: str
    qty: float        # + buy, - sell

@dataclass
class ExecParams:
    style: str        # "TWAP, VWAP, POV"
    n_slices: int 
    pov: float         # used for POV only 
    spread_bps: float 
    impact_alpha: float 
    impact_beta: float
    seed: int | None = None 
    
def implementation_shortfall(arrival: float, fills: List[Tuple[float, float]], side: int) -> Tuple[float, float]:
    if not fills:
        return 0.0, 0.0 
    total_qty = sum(q for _, q in fills)
    vwap = sum(p * q for p, q in fills) / total_qty
    is_abs = side * (vwap - arrival)
    is_bps = (is_abs / arrival) * 1e4 
    return is_abs, is_bps 


class ExecutionSimulator:
    
    def __init__(self, adv_window: int = 20):
        self.adv_window = adv_window
        self.ya = YahooAdapter() 
        
    def _adv_and_arrival(self, symbol: str, start: str, end: str) -> Tuple[float, float]:
        df = self.ya.get_bars(symbol, start, end)[["Close", "Volume"]]
        df = df.dropna()
        if len(df) < max(5, self.adv_window):
            raise RuntimeError(f"Not enough data for {symbol} to estimate ADV.")  # ADV = average daily volume over last adv_window days
        adv = float(df["Volume"].tail(self.adv_window).mean()) 
        arrival = float(df["Close"].iloc[-1])
        return adv, arrival 
    
    def _slice_qtys(self, total_qty: float, style: str, n_slices: int, 
                    curve_weights: np.ndarray, pov: float, mkt_volume_est: float) -> np.ndarray:
        side = 1 if total_qty >= 0 else -1      # side define sell or buy, +1 buy, -1 sell
        qty = abs(total_qty)
        if style.upper() == "TWAP":
            per = qty / n_slices
            q = np.full(n_slices, per)
        elif style.upper() == "VWAP":
            q = qty * curve_weights   # Allocate by curve weights
        elif style.upper() == "POV":      
            q = np.minimum(qty, pov * mkt_volume_est * curve_weights)  # Participate at 'POV' of market volume each slice
            residual = qty - q.sum()     # cap last slice to finish the order
            if residual > 0:
                q[-1] += residual
        else:
            raise ValueError(f"Unknown execution style {style}")
        return side * q
    
    def _fill_price(self, arrival: float, slice_qty: float, adv: float,
                    spread_bps: float, alpha: float, beta: float) -> float:
        if slice_qty == 0:
            return arrival
        side = 1 if slice_qty > 0 else -1
        participation = max(1e-12, abs(slice_qty) / max(adv, 1.0))
        impact = alpha * (participation ** beta)      # in *fraction* terms
        spread = (spread_bps / 1e4)
        return arrival * (1 + side * (impact + spread))
    
    def simulate_day(self,
                     basket: List[BasketItem],
                     data_start: str, data_end: str,
                     params: ExecParams,
                     out_dir: str,
                     overlay_plot: str,
                     is_plot: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        n = params.n_slices
        curve = u_shape_curve(n)
        
        summary_rows = []
        slice_rows = []
        
        for item in basket:
            symbol = item.symbol
            total_qty = float(item.qty)
            side = 1 if total_qty >= 0 else -1
            
            adv, arrival = self._adv_and_arrival(symbol, data_start, data_end)
            mkt_vol_est = adv      # estimate market volume of the day as ADV (simplification)
            child_qtys = self._slice_qtys(total_qty, params.style, n, curve, params.pov, mkt_vol_est) # child quantities
            
            fills = []
            for i in range(n):
                qty_i = child_qtys[i]
                fill_px = self._fill_price(arrival, qty_i, adv, params.spread_bps, params.impact_alpha, params.impact_beta)
                fills.append((fill_px, abs(qty_i)))       # store positive qty for VWAP calc
                slice_rows.append({
                    "symbol": symbol,
                    "slice": i + 1,
                    "child_qty": qty_i,
                    "fill_price": fill_px,
                    "arrival": arrival,
                    "adv": adv,
                })
                
            is_abs, is_bps = implementation_shortfall(arrival, fills, side)
            vwap = sum(p * q for p, q in fills) / max(1e-12,sum(q for _, q in fills))
            notional = abs(total_qty) * arrival 
            
            summary_rows.append({
                "symbol": symbol,
                "side": "BUY" if side > 0 else "SELL",
                "qty": total_qty,
                "arrival_px": arrival,
                "vwap_fill": vwap,
                "IS_abs": is_abs,
                "IS_bps": is_bps,
                "adv": adv,
                "est_day_vol": mkt_vol_est,
                "participation": abs(total_qty) / max(1.0, mkt_vol_est),
                "notional": notional,
            })
        summary = pd.DataFrame(summary_rows)
        slices = pd.DataFrame(slice_rows)
    
        if not summary.empty:
            sym_top = summary.sort_values("notional", ascending=False).iloc[0]["symbol"]
            df_sym = slices[slices["symbol"] == sym_top].copy()
            df_sym["cum_qty"] = df_sym["child_qty"].cumsum()
        
            fig = plt.figure()
            ax1 = plt.gca()
            df_sym["cum_qty"].plot(marker="o", label="Cumulative Qty")
            ax1.set_xlabel("Slice"); ax1.set_ylabel("Cum Qty")
            ax2 = ax1.twinx()
            df_sym["fill_price"].plot(marker="x", label="Fill Price", color="C1")
            ax2.set_ylabel("Price")
            plt.title(f"Execution Overlay - {sym_top} ({params.style})")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
            plt.tight_layout()
            fig.savefig(Path(out_dir) / overlay_plot, dpi=140); plt.close(fig)
            
        if not summary.empty:
            # sign-aware IS bps notionally weighted cumulative (toy visualization)
            ser = (summary["IS_bps"] * summary["notional"]).cumsum() / summary["notional"].sum()
            fig2 = plt.figure()
            (ser / 1.0).plot(title=f"Cumulative IS bps - Basket ({params.style})")
            plt.xlabel("Symbols index (execution order)"); plt.ylabel("Cum IS bps")
            plt.tight_layout()
            fig2.savefig(Path(out_dir) / is_plot, dpi=140); plt.close(fig2)
            
        return summary, slices
                      
    