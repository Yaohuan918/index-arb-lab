from __future__ import annotations 
import math
from datetime import datetime 

def time_to_expiry_yr(now: datetime, expiry: datetime) -> float:
    days = (expiry - now).days 
    return max(days, 0) / 365.0 


def fair_value(index_level: float, r_annual: float, d_annual: float, T_years: float) -> float: 
    return index_level * math.exp((r_annual - d_annual) * T_years)