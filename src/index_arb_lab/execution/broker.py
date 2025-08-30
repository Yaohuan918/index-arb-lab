from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List 
import pandas as pd  
from .models import Order, Fill, Position

@dataclass
class CostSpec:
    one_way_bps: float = 2.0
    slip_bps: float = 2.0
    
@dataclass
class PaperBroker:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    pending: Dict[pd.Timestamp, List[Order]] = field(default_factory=dict)
    fills_log: List[Fill] = field(default_factory=list)
    cost_spec: CostSpec = field(default_factory=CostSpec)
    
    def _get_pos(self, sym: str) -> Position:
        if sym not in self.positions:
            self.positions[sym] = Position(sym, 0.0)
        return self.positions[sym]
    
    def place_order(self, t: pd.Timestamp, order: Order) -> None:
        order.ts = getattr(order, "ts", None) or t
        
        if getattr(order, "side", 0) == 0:
            if order.shares > 0:
                order.side = 1
            elif order.shares < 0:
                order.side = -1
            else:
                order.side = 0
            order.shares = abs(float(order.shares))
            
        self.submit(t, order)
    
    def submit(self, t: pd.Timestamp, o: Order) -> None:
        self.pending.setdefault(t, []).append(o)
        
    def end_of_day_fill(self, t: pd.Timestamp, px_close: pd.Series) -> None:
        for o in self.pending.pop(t, []):
            p0 = float(px_close[o.symbol])   # close
            
            #prefer explicit side; fallback to sign(shares)
            side = getattr(o, "side", 1 if o.shares > 0 else (-1 if o.shares < 0 else 0))
            qty = abs(float(o.shares))
            if side == 0 or qty == 0:
                continue
            # side-aware slippage bps: buy worse price, sell worse price
            slip_mult = (1 + side * (self.cost_spec.slip_bps / 1e4))
            fill_price = p0 * slip_mult
            
            dollars = side * qty * fill_price
            fee = abs(dollars) * (self.cost_spec.one_way_bps / 1e4) # one-way fee
            
            # update cash & position
            self.cash -= dollars + fee
            
            pos = self._get_pos(o.symbol)
            pos.shares += side * qty
            
            self.fills_log.append(Fill(ts=t, symbol=o.symbol, shares=o.shares, price=fill_price, fee=fee))
            
    def value(self, px_close: pd.Series) -> float:
        val = self.cash
        for sym, pos in self.positions.items():
            val += pos.shares * float(px_close.get(sym, 0.0))
        return float(val)  