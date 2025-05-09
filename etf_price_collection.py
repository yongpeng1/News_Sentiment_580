# Add this to your existing price-fetching script using curl_cffi
sector_etfs = [
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV",
    "XLI", "XLB", "XLRE", "XLK", "XLU", "^VIX"
]

from curl_cffi import requests
import pandas as pd
import time
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

def fetch_yahoo_price(ticker, range_days="60d", interval="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "range": range_days,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits"
    }

    r = requests.get(url, headers=headers, impersonate="chrome", params=params, timeout=10)
    if r.status_code != 200:
        print(f"[{ticker}] ❌ Error {r.status_code}")
        return None

    data = r.json()
    chart = data.get("chart", {}).get("result", [{}])[0]
    timestamps = chart.get("timestamp", [])
    prices = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

    df = pd.DataFrame({
        "date": [datetime.utcfromtimestamp(ts).date() for ts in timestamps],
        "adj_close": prices
    })
    df["ticker"] = ticker
    return df

# === Fetch all ETF/VIX prices ===
etf_data = []

for ticker in sector_etfs:
    print(f"📥 Fetching {ticker}...")
    df = fetch_yahoo_price(ticker)
    if df is not None:
        etf_data.append(df)
    time.sleep(1)

# Combine and save
etf_df = pd.concat(etf_data)
etf_df.to_csv("data/all_sector_etfs_and_vix.csv", index=False)
print("✅ All sector ETFs and VIX saved.")