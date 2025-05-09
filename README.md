## News_Sentiment_580
UIUC FIN 580 Final Project

# 📊 Multi-Agent News Sentiment Trading System

This project investigates the alpha decay of news sentiment across S&P 500 sectors using a modular multi-agent framework. Each agent represents a distinct trading strategy (e.g., sentiment-based, momentum, value), and operates independently. We test whether short-term alpha can be systematically captured and how quickly it fades.

---

## 🧱 Project Structure

### 🗂 Data Collection

- `sentiment_collection_newsapi.py`  
  → Uses StockNewsAPI to collect news headlines by ticker.  
  ⚠️ Free trial limits the number of news items — adjust accordingly.

- `sentiment_cleaning.py`  
  → Applies FinBERT to classify sentiment of collected news headlines (positive, negative, neutral).

- `ticker_price_collection.py`  
  → Downloads daily price data for individual tickers (top 2 from each S&P 500 sector).

- `etf_price_collection.py`  
  → Collects price data for sector ETFs (e.g., XLK, XLE, XLU) and VIX (`^VIX`).

- `data_merge.py`  
  → Merges sentiment and price data into a single unified dataset for modeling.

---

### ⚙️ Strategy & Simulation

- `trade_simulation.py`  
  → Simulates trading strategies based on sentiment signals (positive, VIX-filtered, adaptive holding).  
  ✅ Includes benchmark, portfolio plots, and performance metrics.

- `multi_agent.py`  
  → Defines a class-based framework for agents (strategy wrappers).  
  ✅ Includes strategies for momentum, value, reversal, VIX sentiment, etc.

- `multi_agent_evaluation.py`  
  → Runs multiple agents in parallel, evaluates performance (Sharpe, drawdown, return), and can support ensemble agent logic.

- `news_sentiment_alpha.py`  
  → Tests sentiment alpha decay by comparing stock returns vs. sector ETF benchmarks at 1-day, 3-day, and 5-day intervals.  
  ✅ Outputs alpha heatmaps across sectors.

---

## ⚠️ Notes & Warnings

- ✅ All data files must be merged into `full_dataset.csv` and `etf_prices.csv` for consistent usage.
- ❗️Yahoo Finance (`yfinance`) may enforce rate limits. Use `sleep()` between requests if needed.
- ❗️NewsAPI’s free tier has strict query limits. Consider reducing tickers or sampling fewer days if limited.
- ✅ FinBERT model must be pre-downloaded or loaded via Hugging Face Transformers.

---

## 📈 Output & Visualization

- Agent portfolio values plotted over time (vs. benchmark)
- Alpha decay heatmaps (by sector and horizon)
- Tabular summary of Sharpe, Drawdown, Return for each strategy
- Optional: Dynamic agent-of-agents switching logic (future work)

---

## 📚 Reference

- [Tetlock, 2007] News tone predicts returns  
- [Loughran & McDonald, 2011] Finance-specific sentiment dictionary  
- [Araci, 2021] FinBERT: Transformer for financial sentiment

---

## 👨‍💻 Author
Yongpeng Fu

This project was completed as part of an academic research study in algorithmic trading, machine learning, and financial data analysis with AI assitance
