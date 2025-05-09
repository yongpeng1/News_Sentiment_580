import requests
import pandas as pd
import os

# === CONFIGURATION ===
API_KEY = "# Replace with your actual key"  # Replace with your actual key
TICKERS = [
    "AAPL",  # Information Technology
    "MSFT",  # Information Technology
    "UNH",   # Health Care
    "JNJ",   # Health Care
    "JPM",   # Financials
    "BAC",   # Financials
    "AMZN",  # Consumer Discretionary
    "TSLA",  # Consumer Discretionary
    "GOOGL", # Communication Services
    "NFLX",  # Communication Services
    "RTX",   # Industrials
    "UNP",   # Industrials
    "PG",    # Consumer Staples
    "KO",    # Consumer Staples
    "XOM",   # Energy
    "CVX",   # Energy
    "NEE",   # Utilities
    "DUK",   # Utilities
    "AMT",   # Real Estate
    "PLD",   # Real Estate
    "LIN",   # Materials
    "SHW"    # Materials
]
BASE_URL = "https://stocknewsapi.com/api/v1/stat"
OUTPUT_CSV = "data/stocknewsapi_sentiment_30days.csv"

os.makedirs("data", exist_ok=True)

def get_sentiment_for_ticker(ticker):
    params = {
        "tickers": ticker,
        "date": "01152025-today",
        "page": 1,
        "token": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"[ERROR {response.status_code}] {response.text}")
        return []

    data = response.json()
    daily_data = data.get("data", {})
    parsed = []

    for date, day_result in daily_data.items():
        sentiment_data = day_result.get(ticker)
        if not sentiment_data:
            continue
        parsed.append({
            "ticker": ticker,
            "date": date,
            "positive": sentiment_data.get("Positive", 0),
            "neutral": sentiment_data.get("Neutral", 0),
            "negative": sentiment_data.get("Negative", 0),
            "sentiment_score": sentiment_data.get("sentiment_score", None)
        })

    return parsed

def main():
    all_results = []
    for ticker in TICKERS:
        print(f"📊 Fetching sentiment for {ticker}...")
        ticker_data = get_sentiment_for_ticker(ticker)
        all_results.extend(ticker_data)

    df = pd.DataFrame(all_results)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["ticker", "date"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved sentiment scores to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()