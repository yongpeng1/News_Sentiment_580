import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stock_to_etf = {
        "AAPL": "XLK", "MSFT": "XLK", "JNJ": "XLV", "PFE": "XLV",
        "XOM": "XLE", "CVX": "XLE", "JPM": "XLF", "BAC": "XLF",
        "WMT": "XLP", "PG": "XLP", "NEE": "XLU", "DUK": "XLU",
        "UNP": "XLI", "UPS": "XLI", "TMO": "XLV", "ABT": "XLV",
        "HD": "XLY", "MCD": "XLY", "AMT": "XLRE", "PLD": "XLRE",
        "NFLX": "XLC", "GOOG": "XLC"
    }


# ---------------------------
# Utility: Evaluate Performance
# ---------------------------
def evaluate_performance(portfolio):
    if portfolio.empty:
        return {"Sharpe": np.nan, "Max Drawdown": np.nan, "Total Return": np.nan}

    df = portfolio.copy()
    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    sharpe = np.mean(df["daily_return"]) / np.std(df["daily_return"]) * np.sqrt(252)
    df["cum_max"] = df["portfolio_value"].cummax()
    drawdown = df["portfolio_value"] / df["cum_max"] - 1
    max_dd = drawdown.min()
    total_return = df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0] - 1
    return {"Sharpe": sharpe, "Max Drawdown": max_dd, "Total Return": total_return}

# ---------------------------
# Agent of Agents (Dynamic Selector)
# ---------------------------

# ---------------------------
# Execution
# ---------------------------
if __name__ == "__main__":
    from multi_agent import Agent, strategy_positive, strategy_momentum, strategy_reversal, strategy_value, strategy_vix_guard, strategy_adaptive_vix_neg
    
    full = pd.read_csv("data/full_dataset.csv", parse_dates=["date"])
    prices = pd.read_csv("data/all_sector_etfs_and_vix.csv", parse_dates=["date"])
    # Set global variable for mapping
    agents = {
        "Positive": Agent("Positive", strategy_positive),
        "Momentum": Agent("Momentum", strategy_momentum),
        "Reversal": Agent("Reversal", strategy_reversal),
        "Value": Agent("Value", strategy_value),
        "VIX Filter": Agent("VIX Filter", strategy_vix_guard),
        "Adaptive": Agent("Adaptive", strategy_adaptive_vix_neg)
    }

    for agent in agents.values():
        agent.run(full, prices)
    # Step 1: Extract sector ETF tickers
    benchmark_etfs = list(set(stock_to_etf.values()))

    # Step 2: Filter ETF prices for benchmark ETFs
    benchmark = prices[prices["ticker"].isin(benchmark_etfs)].copy()

    # Step 3: Pivot so each ETF is a column, indexed by date
    benchmark = benchmark.pivot(index="date", columns="ticker", values="adj_close").ffill()

    # Step 4: Equal-weighted benchmark portfolio (mean of ETF prices per day)
    benchmark["portfolio_value"] = benchmark.mean(axis=1)

    # Keep only the portfolio value
    benchmark = benchmark[["portfolio_value"]]


    print("\n--- Individual Performance ---")
    for name, agent in agents.items():
        perf = evaluate_performance(agent.portfolio)
        print(f"{name}: {perf}")
    plt.figure(figsize=(12, 6))
    for name, agent in agents.items():
        if not agent.portfolio.empty:
            plt.plot(agent.portfolio.index, agent.portfolio["portfolio_value"], label=name)
    plt.plot(benchmark.index, benchmark["portfolio_value"], label="Benchmark", linestyle="--", color="black")
    plt.title("Multi-Agent Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()