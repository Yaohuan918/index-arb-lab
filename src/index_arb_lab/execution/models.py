from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd  

@dataclass
class Order:
    ts: pd.Timestamp
    symbol: str
    side: int       # +1 buy, -1 sell
    shares: float
    kind: str = "MOC"  # market-on-close
    
@dataclass
class Fill:
    ts: pd.Timestamp
    symbol: str
    shares: float
    price: float          # execution price(include slippage)
    fee: float            # bps costs converted to $ notional
    
@dataclass
class Position:
    symbol: str
    shares: float = 0.0
    
Portfolio = Dict[str, Position]