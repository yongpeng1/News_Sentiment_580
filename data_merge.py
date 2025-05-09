import pandas as pd
from datetime import datetime

# === Load files ===
sentiment_df = pd.read_csv("data/stocknewsapi_sentiment_30days.csv", parse_dates=["date"])
price_df = pd.read_csv("data/yahoo_prices_stealth.csv", parse_dates=["date"])
etf_df = pd.read_csv("data/all_sector_etfs_and_vix.csv", parse_dates=["date"])

# === Clean all date fields to just date ===
sentiment_df["date"] = sentiment_df["date"].dt.date
price_df["date"] = price_df["date"].dt.date
etf_df["date"] = etf_df["date"].dt.date

# === Pivot sentiment into ticker/date flat table ===
sentiment_flat = sentiment_df[["ticker", "date", "sentiment_score"]]

# === Compute forward returns ===
price_df.sort_values(["ticker", "date"], inplace=True)
price_df["return_1d"] = price_df.groupby("ticker")["adj_close"].pct_change(periods=1).shift(-1)
price_df["return_3d"] = price_df.groupby("ticker")["adj_close"].pct_change(periods=3).shift(-3)
price_df["return_5d"] = price_df.groupby("ticker")["adj_close"].pct_change(periods=5).shift(-5)

# === Merge sentiment with price returns ===
merged_df = pd.merge(price_df, sentiment_flat, how="left", on=["ticker", "date"])
merged_df["sentiment_score"].fillna(0, inplace=True)

# === Export clean outputs ===
sentiment_flat.to_csv("data/merged_sentiment.csv", index=False)
price_df.to_csv("data/merged_prices.csv", index=False)
etf_df.to_csv("data/etf_prices.csv", index=False)
merged_df.to_csv("data/full_dataset.csv", index=False)

print("✅ All files processed and saved:")
print("- data/merged_sentiment.csv")
print("- data/merged_prices.csv")
print("- data/etf_prices.csv")
print("- data/full_dataset.csv")