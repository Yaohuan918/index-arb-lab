from __future__ import annotations 
import yaml
from dataclasses import dataclass 
from datetime import datetime  

@dataclass 
class Config:
    cash_symbol: str 
    futures_symbol: str 
    lookback_days: int 
    expiry: str 
    annual_rate: float 
    annual_div_yield: float 
    z_entry: float 
    z_exit: float 
    out_dir: str 
    plot_filename: str 
    log_path: str 
    
    @property 
    def expiry_dt(self) -> datetime:
        return datetime.fromisoformat(self.expiry) 
    
def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) 
         
    return Config(
        cash_symbol = raw["universe"]["cash_symbol"],
        futures_symbol = raw["universe"]["futures_symbols"],
        lookback_days = int(raw["lookback_days"]),
        expiry = raw["expiry"],
        annual_rate = float(raw["rates"]["annual_rate"]),
        annual_div_yield = float(raw["dividends"]["annual_yield"]),
        z_entry = float(raw["alerts"]["z_entry"]),
        z_exit = float(raw["alerts"]["z_exit"]),
        out_dir = raw["reporting"]["out_dir"],
        plot_filename = raw["reporting"]["plot_filename"],
        log_path = raw["logging"]["path"],
    )
        
