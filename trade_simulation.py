import pandas as pd
import matplotlib.pyplot as plt
# === Load data ===
df = pd.read_csv("data/full_dataset.csv", parse_dates=["date"])
etf_prices = pd.read_csv("data/etf_prices.csv", parse_dates=["date"])

# === Map stock tickers to their sector ETF ===
stock_to_etf = {
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
df["sector_etf"] = df["ticker"].map(stock_to_etf)

# === Create sector-level sentiment signal (average of the 2 stocks per sector per day) ===
sector_sentiment = (
    df.groupby(["date", "sector_etf"])["sentiment_score"]
    .mean()
    .reset_index()
    .dropna()
)
def generate_positive_sentiment_trades(sentiment_df, price_df, hold_days=1, threshold=0):
    """
    Strategy: Buy on positive sentiment and hold for short-term momentum.
    - sentiment_df: must contain columns ['date', 'sector_etf', 'sentiment_score']
    - price_df: ETF prices with ['date', 'ticker', 'adj_close']
    """
    sentiment_df = sentiment_df.copy()
    price_df = price_df.copy()

    sentiment_df["signal"] = sentiment_df["sentiment_score"] > threshold
    entries = sentiment_df[sentiment_df["signal"]].copy()
    entries = entries.rename(columns={"sector_etf": "ticker"})

    # Merge to get entry price
    entries = pd.merge(entries, price_df, on=["date", "ticker"], how="inner")
    entries = entries.rename(columns={"adj_close": "entry_price"})
    entries["exit_date"] = entries["date"] + pd.Timedelta(days=hold_days)

    # Merge to get exit price
    exit_prices = price_df.rename(columns={"date": "exit_date", "adj_close": "exit_price"})
    trades = pd.merge(entries, exit_prices, on=["exit_date", "ticker"], how="left")

    trades["return"] = trades["exit_price"] / trades["entry_price"] - 1
    trades["capital"] = 10000
    trades["pnl"] = trades["return"] * trades["capital"]

    return trades[["date", "exit_date", "ticker", "entry_price", "exit_price", "return", "pnl"]]

def generate_negative_sentiment_trades(sentiment_df, price_df, hold_days=5, threshold=0):
    sentiment_df["signal"] = sentiment_df["sentiment_score"] < threshold
    entries = sentiment_df[sentiment_df["signal"]].copy()
    entries = entries.rename(columns={"sector_etf": "ticker"})

    # Merge entry price
    entries = pd.merge(entries, price_df, on=["date", "ticker"], how="inner")
    entries = entries.rename(columns={"adj_close": "entry_price"})
    entries["exit_date"] = entries["date"] + pd.Timedelta(days=hold_days)

    # Merge exit price
    exit_prices = price_df.rename(columns={"date": "exit_date", "adj_close": "exit_price"})
    trades = pd.merge(entries, exit_prices, on=["exit_date", "ticker"], how="left")

    trades["return"] = trades["exit_price"] / trades["entry_price"] - 1
    trades["capital"] = 10000
    trades["pnl"] = trades["return"] * trades["capital"]

    return trades[["date", "exit_date", "ticker", "entry_price", "exit_price", "return", "pnl"]]

def generate_negative_sentiment_with_vix_filter(sentiment_df, price_df, vix_df, vix_threshold=25, hold_days=5, threshold=0):
    """
    Buy sector ETF only when sentiment is negative and VIX is calm (below threshold).
    
    Inputs:
        sentiment_df: DataFrame with ['date', 'sector_etf', 'sentiment_score']
        price_df: DataFrame with ['date', 'ticker', 'adj_close']
        vix_df: DataFrame with ['date', 'adj_close'] for ticker == '^VIX'
    """
    # Merge VIX into sentiment signal
    vix_df = vix_df[vix_df["ticker"] == "^VIX"][["date", "adj_close"]].rename(columns={"adj_close": "vix_close"})
    df = pd.merge(sentiment_df, vix_df, on="date", how="left")

    # Define signal: negative sentiment & VIX < threshold
    df["signal"] = (df["sentiment_score"] < threshold) & (df["vix_close"] < vix_threshold)

    entries = df[df["signal"]].copy()
    entries = entries.rename(columns={"sector_etf": "ticker"})

    # Merge for entry price
    entries = pd.merge(entries, price_df, on=["date", "ticker"], how="inner")
    entries = entries.rename(columns={"adj_close": "entry_price"})
    entries["exit_date"] = entries["date"] + pd.Timedelta(days=hold_days)

    # Merge for exit price
    exit_prices = price_df.rename(columns={"date": "exit_date", "adj_close": "exit_price"})
    trades = pd.merge(entries, exit_prices, on=["exit_date", "ticker"], how="left")

    trades["return"] = trades["exit_price"] / trades["entry_price"] - 1
    trades["capital"] = 10000
    trades["pnl"] = trades["return"] * trades["capital"]

    return trades[["date", "exit_date", "ticker", "entry_price", "exit_price", "return", "pnl"]]

def generate_adaptive_vix_sentiment_trades(sentiment_df, price_df, vix_df,
                                            sentiment_threshold=-0.3,
                                            target_return=0.015,
                                            stop_loss=-0.05,
                                            max_hold_days=5):
    """
    Adaptive holding + VIX trend filter + stop-loss strategy.
    Exit if:
        - target return is reached
        - OR stop-loss triggered
        - OR max_hold_days reached
    """

    # 1. Prepare VIX data
    vix_series = vix_df[vix_df["ticker"] == "^VIX"][["date", "adj_close"]].sort_values("date")
    vix_series["vix_yesterday"] = vix_series["adj_close"].shift(1)
    vix_series["vix_falling"] = vix_series["adj_close"] < vix_series["vix_yesterday"]
    vix_series = vix_series.rename(columns={"adj_close": "vix_close"})

    # 2. Merge VIX + sentiment
    signal_df = pd.merge(sentiment_df, vix_series[["date", "vix_falling"]], on="date", how="left")
    signal_df["signal"] = (signal_df["sentiment_score"] < sentiment_threshold) & (signal_df["vix_falling"])

    entries = signal_df[signal_df["signal"]].copy()
    entries = entries.rename(columns={"sector_etf": "ticker"})

    # 3. Get entry price
    entries = pd.merge(entries, price_df, on=["date", "ticker"], how="inner")
    entries = entries.rename(columns={"adj_close": "entry_price"})
    entries["entry_date"] = entries["date"]
    entries = entries[["entry_date", "ticker", "entry_price"]]

    # 4. Adaptive trade execution
    all_trades = []

    for i in range(len(entries)):
        row = entries.iloc[i]
        t0 = row["entry_date"]
        ticker = row["ticker"]
        entry_price = row["entry_price"]

        price_path = price_df[(price_df["ticker"] == ticker) & (price_df["date"] > t0)]
        price_path = price_path.sort_values("date").head(max_hold_days)

        for j, (_, exit_row) in enumerate(price_path.iterrows()):
            ret = exit_row["adj_close"] / entry_price - 1

            if ret >= target_return or ret <= stop_loss or j == max_hold_days - 1:
                all_trades.append({
                    "date": t0,
                    "exit_date": exit_row["date"],
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "exit_price": exit_row["adj_close"],
                    "return": ret,
                    "capital": 10000,
                    "pnl": ret * 10000
                })
                break

    return pd.DataFrame(all_trades)


def simulate_portfolio(trades, start_value=110000):
    pnl_by_day = trades.groupby("exit_date")["pnl"].sum().reset_index()
    pnl_by_day = pnl_by_day.rename(columns={"exit_date": "date"})

    full_dates = pd.date_range(trades["date"].min(), trades["exit_date"].max(), freq="D")
    portfolio = pd.DataFrame({"date": full_dates})
    portfolio = pd.merge(portfolio, pnl_by_day, on="date", how="left")
    portfolio["pnl"].fillna(0, inplace=True)
    portfolio["portfolio_value"] = start_value + portfolio["pnl"].cumsum()

    return portfolio

def simulate_benchmark(price_df, sector_etfs):
    price_df = price_df[price_df["ticker"].isin(sector_etfs)].copy()
    start_prices = price_df.groupby("ticker").first()["adj_close"]
    price_df["start_price"] = price_df["ticker"].map(start_prices)
    price_df["value"] = 10000 * price_df["adj_close"] / price_df["start_price"]
    benchmark = price_df.groupby("date")["value"].sum().reset_index()
    benchmark = benchmark.rename(columns={"value": "benchmark_value"})
    return benchmark

def evaluate_performance(portfolio_df, name="Strategy"):
    """
    Evaluate portfolio performance: total return, annual return, volatility, Sharpe, max drawdown.
    Assumes portfolio_df has columns ['date', 'portfolio_value'].
    """
    df = portfolio_df.copy().sort_values("date")

    # Use portfolio_value regardless of original column name
    if "benchmark_value" in df.columns:
        df = df.rename(columns={"benchmark_value": "portfolio_value"})

    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)

    total_return = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(df)) - 1
    volatility = df["daily_return"].std() * (252 ** 0.5)
    sharpe = ann_return / volatility if volatility > 0 else 0

    # Max drawdown
    cummax = df["portfolio_value"].cummax()
    drawdown = (cummax - df["portfolio_value"]) / cummax
    max_dd = drawdown.max()

    return {
        "Strategy": name,
        "Total Return": round(total_return, 4),
        "Annual Return": round(ann_return, 4),
        "Volatility": round(volatility, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Max Drawdown": round(max_dd, 4)
    }

import matplotlib.pyplot as plt

def plot_comparison(pos_df, adaptive_portfolio, vix_neg_df, benchmark_df):
    plt.figure(figsize=(10, 6))

    # Benchmark = solid navy
    plt.plot(benchmark_df["date"], benchmark_df["benchmark_value"],
             label=" Benchmark", color="navy", linewidth=2)

    # Positive sentiment = dashed green
    plt.plot(pos_df["date"], pos_df["portfolio_value"],
             label=" Positive Sentiment", linestyle="--", color="green", linewidth=2)

    # Negative sentiment = dashed orange
    plt.plot(adaptive_portfolio["date"], adaptive_portfolio["portfolio_value"],
             label=" Negative Sentiment+VIX trend+adaptive holding", linestyle="--", color="orange", linewidth=2)

    # VIX-filtered negative sentiment = dashed red
    plt.plot(vix_neg_df["date"], vix_neg_df["portfolio_value"],
             label="Negative + VIX < 25", linestyle="--", color="red", linewidth=2)

    # Format
    plt.title("📉 Strategy Performance: All Sentiment Variants vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    neg_trades = generate_negative_sentiment_trades(sector_sentiment, etf_prices)
    neg_portfolio = simulate_portfolio(neg_trades)

    pos_trades = generate_positive_sentiment_trades(sector_sentiment, etf_prices)
    pos_portfolio = simulate_portfolio(pos_trades)

    vix_trades = generate_negative_sentiment_with_vix_filter(sector_sentiment, etf_prices,etf_prices)
    vix_neg_portfolio = simulate_portfolio(vix_trades)

    adaptive_trades = generate_adaptive_vix_sentiment_trades(sector_sentiment, etf_prices, etf_prices)
    adaptive_portfolio = simulate_portfolio(adaptive_trades)

    benchmark = simulate_benchmark(etf_prices, sector_sentiment["sector_etf"].unique())
    
    results = [
    evaluate_performance(pos_portfolio, "Positive Sentiment"),
    evaluate_performance(vix_neg_portfolio, "Negative + VIX Filter"),
    evaluate_performance(benchmark, "Benchmark"),
    evaluate_performance(adaptive_portfolio, "Adaptive Holding + VIX trend")
]

    results_df = pd.DataFrame(results)
    print("\n📊 Strategy Performance Summary:")
    print(results_df.to_string(index=False))

    plot_comparison(
    pos_portfolio,
    adaptive_portfolio,
    vix_neg_portfolio,
    benchmark
    )
