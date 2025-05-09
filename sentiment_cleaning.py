import pandas as pd

df = pd.read_csv("data/stocknewsapi_sentiment_30days.csv", parse_dates=["date"])

# Pivot to have dates as rows and tickers as columns
pivot_df = df.pivot(index="date", columns="ticker", values="sentiment_score")

# Replace missing values (NaNs) with neutral sentiment = 0
pivot_df_filled = pivot_df.fillna(0)

# Optional: sort rows by date
pivot_df_filled = pivot_df_filled.sort_index()

# Save to new CSV (optional)
pivot_df_filled.to_csv("data/sentiment_score_matrix.csv")

# Display preview
print(pivot_df_filled.head())
print(pivot_df_filled.tail())