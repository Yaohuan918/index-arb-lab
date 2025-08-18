from __future__ import annotations 
import argparse 
from pathlib import Path 
from datetime import date 
import pandas as pd 

def main():
    parser = argparse.ArgumentParser(description="Build a tiny HTML report for a given date.")
    parser.add_argument("--date", default="today", help="IS0 date (YYYY-MM-DD) or 'today'.")
    parser.add_argument("--out", required=True, help="Output HTML path")
    args = parser.parse_args() 
    
    d = date.today().isoformat() if args.date == "today" else args.date 
    reports = Path("reports")
    csv = reports / f"basis_daily_{d}.csv"
    png = reports / "basis_plot.png" 
    
    if not csv.exists():
        raise FileNotFoundError(f"Cannot find {csv}")
    
    df = pd.read_csv(csv)
    
    html = f""" 
    <html>
    <head>
        <meta charset="utf-8"/> 
        <title>Daily Basis Report -{d}</title> 
        <style> 
            body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }}
            table {{ border-collapse: collapse; }}
            th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
            th {{ background: #f6f6f6; }}
        </style> 
    </head>
    <body> 
        <h2>Daily Basis Report - {d}</h2> 
        <h3>Snapshot</h3> 
        {df.to_html(index=False)}
        <h3>Basis (rolling)</h3> 
        <img src="basis_plot.png" alt="basis plot" style="max_width: 900px; width: 100%;"/> 
        <p style="color: #666;">Educational project. Not investment advice.</p>
    </body>
    </html>
    """
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()