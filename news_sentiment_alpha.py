import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load datasets ===
df = pd.read_csv("data/full_dataset.csv", parse_dates=["date"])
etf_df = pd.read_csv("data/etf_prices.csv", parse_dates=["date"])

# === Map each ticker to its corresponding sector ETF ===
ticker_sector_map = {
    "AAPL": "XLK", "MSFT": "XLK",
    "UNH": "XLV", "JNJ": "XLV",
    "JPM": "XLF", "BAC": "XLF",
    "AMZN": "XLY", "TSLA": "XLY",
    "GOOGL": "XLC", "NFLX": "XLC",
    "RTX": "XLI", "UNP": "XLI",
    "PG": "XLP", "KO": "XLP",
    "XOM": "XLE", "CVX": "XLE",
    "NEE": "XLU", "DUK": "XLU",
    "AMT": "XLRE", "PLD": "XLRE",
    "LIN": "XLB", "SHW": "XLB"
}
df["sector_etf"] = df["ticker"].map(ticker_sector_map)

# === Compute sentiment label ===
def classify_sentiment(score):
    if score > 0: return "positive"
    elif score < 0: return "negative"
    else: return "neutral"

df["sentiment_label"] = df["sentiment_score"].apply(classify_sentiment)

# === Prepare ETF return data ===
etf_df.sort_values(["ticker", "date"], inplace=True)
etf_df["return_1d"] = etf_df.groupby("ticker")["adj_close"].pct_change(periods=1).shift(-1)
etf_df["return_3d"] = etf_df.groupby("ticker")["adj_close"].pct_change(periods=3).shift(-3)
etf_df["return_5d"] = etf_df.groupby("ticker")["adj_close"].pct_change(periods=5).shift(-5)

# === Merge stock data with sector ETF returns ===
etf_returns = etf_df.rename(columns={
    "ticker": "sector_etf",
    "return_1d": "etf_1d",
    "return_3d": "etf_3d",
    "return_5d": "etf_5d"
})[["sector_etf", "date", "etf_1d", "etf_3d", "etf_5d"]]

merged = pd.merge(df, etf_returns, on=["sector_etf", "date"], how="inner")

# === Calculate excess return (alpha) ===
merged["alpha_1d"] = merged["return_1d"] - merged["etf_1d"]
merged["alpha_3d"] = merged["return_3d"] - merged["etf_3d"]
merged["alpha_5d"] = merged["return_5d"] - merged["etf_5d"]

# === Group by sentiment and report alpha ===
# === Group by sector ETF + sentiment and compute mean alpha ===
grouped_sector_sentiment = merged.groupby(
    ["sector_etf", "sentiment_label"]
)[["alpha_1d", "alpha_3d", "alpha_5d"]].mean()

print("📊 Sector-wise alpha vs ETF by sentiment label:")
print(grouped_sector_sentiment)



# Pivot into heatmap format
heatmap_data = merged.groupby(["sector_etf", "sentiment_label"])[
    ["alpha_1d", "alpha_3d", "alpha_5d"]
].mean().reset_index()

# Reshape for heatmap: one plot per sentiment label
# === Alpha by company and horizon ===
for sentiment in ["positive", "neutral", "negative"]:
    company_alpha = merged[merged["sentiment_label"] == sentiment].groupby("ticker")[
        ["alpha_1d", "alpha_3d", "alpha_5d"]
    ].mean()

    plt.figure(figsize=(10, 7))
    sns.heatmap(company_alpha, annot=True, cmap="RdYlGn", center=0, fmt=".4f")
    plt.title(f"Alpha Heatmap — {sentiment.capitalize()} Sentiment")
    plt.ylabel("Ticker")
    plt.xlabel("Time Window")
    plt.tight_layout()
    plt.show()

### Does sentiment affect return
