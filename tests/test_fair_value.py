from datetime import datetime, timedelta 
from src.index_arb_lab.index.fair_value import fair_value, time_to_expiry_yr 

def test_time_to_expiry_yr_act365():
    now = datetime(2025, 1, 1)
    exp = now + timedelta(days=365)
    assert abs(time_to_expiry_yr(now, exp)  - 1.0) < 1e-9 
    
def test_fair_value_monotone_when_r_gt_d():
    I0 = 5000.0
    r, d, T = 0.03, 0.01, 0.5 
    f = fair_value(I0, r, d, T) 
    assert f > I0 # if r > d, fair value should exceed cash 
    