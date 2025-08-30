from src.index_arb_lab.execution.broker import PaperBroker, CostSpec
from src.index_arb_lab.execution.models import Order
import pandas as pd   

def test_broker_basic_fill():
    b = PaperBroker(cash=0.0, costs=CostSpec(one_way_bps=0.0, slip_bps=0.0))
    t = pd.Timestamp("2024-01-02")
    px = pd.Series({"AAA": 100.0})
    
    # buy 10 @ 100
    b.submit(t, Order(ts=t, symbol="AAA", side=+1, shares=10))
    b.end_of_day_fill(t, px)
    assert b.positions["AAA"].shares == 10
    assert abs(b.cash + 1000.0) < 1e-6    # spent $1000
    
    # value with same price
    assert abs(b.value(px) - 0) < 1e-6
