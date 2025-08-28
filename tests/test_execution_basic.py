import numpy as np
from src.index_arb_lab.execution.simulator import implementation_shortfall

def test_is_sign_buy_sell():
    # If fills are at arrival, IS=0 for both sides
    arrival = 100.0
    fills = [(100.0, 50), (100.0, 50)]
    is_abs, is_bps = implementation_shortfall(arrival, fills, side=1)
    assert abs(is_abs) < 1e-12 and abs(is_bps) < 1e-9
    is_abs, is_bps = implementation_shortfall(arrival, fills, side=-1)
    assert abs(is_abs) < 1e-12 and abs(is_bps) < 1e-9
    
def test_vwap_math():
    arrival = 100.0
    fills = [(101.0, 40), (99.0, 60)]  
    # VWAP = 0.4*101 + 0.6*99 = 99.8
    # Buy: side=+1, IS = 99.8 - 100 = -0.2; Sell: IS>0
    is_abs_buy, _ = implementation_shortfall(arrival, fills, side=1)
    is_abs_sell, _ = implementation_shortfall(arrival, fills, side=-1)
    assert is_abs_buy < 0 and is_abs_sell > 0
    