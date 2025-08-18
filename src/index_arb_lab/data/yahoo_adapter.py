from __future__ import annotations 
import pandas as pd 
import yfinance as yf 

class YahooAdapter:
    def get_bars(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        df = df.rename(columns=lambda c: c.strip().title())
        df.index = pd.to_datetime(df.index)
        return df