import pandas as pd
import numpy as np
from datetime import timedelta

stock_to_etf = {
        "AAPL": "XLK", "MSFT": "XLK", "JNJ": "XLV", "PFE": "XLV",
        "XOM": "XLE", "CVX": "XLE", "JPM": "XLF", "BAC": "XLF",
        "WMT": "XLP", "PG": "XLP", "NEE": "XLU", "DUK": "XLU",
        "UNP": "XLI", "UPS": "XLI", "TMO": "XLV", "ABT": "XLV",
        "HD": "XLY", "MCD": "XLY", "AMT": "XLRE", "PLD": "XLRE",
        "NFLX": "XLC", "GOOG": "XLC"
    }
# ---------------------------
# Agent Class
# ---------------------------
class Agent:
    def __init__(self, name, strategy_fn):
        self.name = name
        self.strategy_fn = strategy_fn
        self.trades = None
        self.portfolio = None

    def run(self, data, etf_prices):
        self.trades = self.strategy_fn(data, etf_prices)
        self.portfolio = simulate_portfolio(self.trades)

# ---------------------------
# Utility: Portfolio Simulation
# ---------------------------
def simulate_portfolio(trades):
    if trades is None or trades.empty:
        return pd.DataFrame()
    trades = trades.copy()
    trades["pnl"] = trades["capital"] * trades["return"]

    pnl_by_date = trades.groupby("exit_date")["pnl"].sum().sort_index()
    portfolio = pnl_by_date.cumsum().rename("portfolio_value").to_frame()
    portfolio["portfolio_value"] += 22000  # initial capital
    return portfolio

# ---------------------------
# Strategy 1: Positive Sentiment
# ---------------------------
def strategy_positive(data, etf_prices):
    df = data.copy()
    df = df[df["sentiment_score"] > 0]
    df["sector_etf"] = df["ticker"].map(stock_to_etf)
    signals = df.groupby(["date", "sector_etf"])["sentiment_score"].mean().reset_index()
    signals["signal"] = True

    return make_trades(signals, etf_prices, hold_days=1)

# ---------------------------
# Strategy 2: Momentum
# ---------------------------
def strategy_momentum(data, etf_prices):
    df = data.copy()
    df["sector_etf"] = df["ticker"].map(stock_to_etf)
    signals = df.groupby(["date", "sector_etf"])["return_5d"].mean().reset_index()
    signals["signal"] = signals["return_5d"] > 0
    return make_trades(signals, etf_prices, hold_days=5)

# ---------------------------
# Strategy 3: Reversal
# ---------------------------
def strategy_reversal(data, etf_prices):
    df = data.copy()
    df["sector_etf"] = df["ticker"].map(stock_to_etf)
    signals = df.groupby(["date", "sector_etf"])[["return_5d", "sentiment_score"]].mean().reset_index()
    signals["signal"] = (signals["return_5d"] < 0) & (signals["sentiment_score"] < 0)
    return make_trades(signals, etf_prices, hold_days=5)

# ---------------------------
# Strategy 4: Value
# ---------------------------
def strategy_value(data, etf_prices):
    price = etf_prices[etf_prices["ticker"] != "^VIX"].copy()
    price["ma20"] = price.groupby("ticker")["adj_close"].transform(lambda x: x.rolling(20).mean())
    price["signal"] = price["adj_close"] < price["ma20"]
    signals = price[["date", "ticker", "signal"]].copy()
    return make_trades(signals, price, ticker_col="ticker", hold_days=5)

# ---------------------------
# Strategy 5: Volatility Aversion
# ---------------------------
def strategy_vix_guard(data, etf_prices):
    vix = etf_prices[etf_prices["ticker"] == "^VIX"][["date", "adj_close"]].rename(columns={"adj_close": "vix"})
    df = data.merge(vix, on="date")
    low_vix = df[df["vix"] < 18].copy()
    low_vix["sector_etf"] = low_vix["ticker"].map(stock_to_etf)
    signals = low_vix.groupby(["date", "sector_etf"])["return_5d"].mean().reset_index()
    signals["signal"] = signals["return_5d"] > 0
    return make_trades(signals, etf_prices, hold_days=5)

# ---------------------------
# Strategy 6: Adaptive VIX + Negative Sentiment
# ---------------------------
def strategy_adaptive_vix_neg(data, etf_prices):
    vix = etf_prices[etf_prices["ticker"] == "^VIX"][["date", "adj_close"]].rename(columns={"adj_close": "vix"})
    vix["vix_falling"] = vix["vix"].diff() < 0
    df = data.merge(vix, on="date")
    df = df[(df["sentiment_score"] < -0.3) & (df["vix_falling"])]
    df["sector_etf"] = df["ticker"].map(stock_to_etf)

    signals = df.groupby(["date", "sector_etf"])["sentiment_score"].mean().reset_index()
    signals["signal"] = True
    return make_trades(signals, etf_prices, hold_days=5)

# ---------------------------
# Shared Trade Generator
# ---------------------------
def make_trades(signal_df, price_df, ticker_col="sector_etf", hold_days=5):
    entries = signal_df[signal_df["signal"]].copy()
    entries = entries.rename(columns={ticker_col: "ticker"})
    entries = pd.merge(entries, price_df, on=["date", "ticker"], how="inner")
    entries = entries.rename(columns={"adj_close": "entry_price"})
    entries["exit_date"] = entries["date"] + pd.Timedelta(days=int(hold_days))

    exits = price_df.rename(columns={"date": "exit_date", "adj_close": "exit_price"})
    trades = pd.merge(entries, exits, on=["exit_date", "ticker"], how="left")
    trades["return"] = trades["exit_price"] / trades["entry_price"] - 1
    trades["capital"] = 10000
    return trades[["date", "exit_date", "ticker", "entry_price", "exit_price", "return", "capital"]]



# ---------------------------
# Execution
# ---------------------------
if __name__ == "__main__":
    full = pd.read_csv("data/full_dataset.csv", parse_dates=["date"])
    prices = pd.read_csv("data/all_sector_etfs_and_vix.csv", parse_dates=["date"])

    stock_to_etf = {
        "AAPL": "XLK", "MSFT": "XLK", "JNJ": "XLV", "PFE": "XLV",
        "XOM": "XLE", "CVX": "XLE", "JPM": "XLF", "BAC": "XLF",
        "WMT": "XLP", "PG": "XLP", "NEE": "XLU", "DUK": "XLU",
        "UNP": "XLI", "UPS": "XLI", "TMO": "XLV", "ABT": "XLV",
        "HD": "XLY", "MCD": "XLY", "AMT": "XLRE", "PLD": "XLRE",
        "NFLX": "XLC", "GOOG": "XLC"
    }

    agents = [
        Agent("Positive Sentiment", strategy_positive),
        Agent("Momentum", strategy_momentum),
        Agent("Reversal", strategy_reversal),
        Agent("Value", strategy_value),
        Agent("Volatility Aversion", strategy_vix_guard),
        Agent("Adaptive VIX + Negative", strategy_adaptive_vix_neg)
    ]

    for agent in agents:
        agent.run(full, prices)
        print(f"{agent.name} Final Value: {agent.portfolio['portfolio_value'].iloc[-1] if not agent.portfolio.empty else 'N/A'}")
