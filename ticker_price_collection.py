from curl_cffi import requests
import pandas as pd
import json
from datetime import datetime
import time

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

tickers = [
    "AAPL", "MSFT", "UNH", "JNJ", "JPM", "BAC", "AMZN", "TSLA", "GOOGL", "NFLX",
    "RTX", "UNP", "PG", "KO", "XOM", "CVX", "NEE", "DUK", "AMT", "PLD", "LIN", "SHW"
]
all_data = []

for ticker in tickers:
    print(f"📥 Fetching {ticker}...")
    df = fetch_yahoo_price(ticker)
    if df is not None:
        all_data.append(df)
    time.sleep(1)  # avoid hammering

final_df = pd.concat(all_data)
final_df.to_csv("data/yahoo_prices_stealth.csv", index=False)
print("✅ Saved to data/yahoo_prices_stealth.csv")